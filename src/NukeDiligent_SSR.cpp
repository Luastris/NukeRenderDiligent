#include "NukeDiligentImpl.h"


// G-buffer targets for the SSR prepass: RGBA16F colour (octN.xy, rough, metal) + D32 depth, both 1x (single
// sample) and shader-readable. Resized to the current camera target.
void NukeDiligent::Impl::EnsureGBuffer(int w, int h)
{
	if (w <= 0 || h <= 0) return;
	if (gbufColor && gbufW == w && gbufH == h) return;   // already the active set

	const uint64_t key = ((uint64_t)(uint32_t)w << 32) | (uint32_t)h;
	auto it = gbufCache.find(key);
	if (it == gbufCache.end())
	{
		// Miss: build a fresh set for this size. Nothing is released — other sizes stay
		// cached, so no in-use buffer is freed mid-frame (that was the device-removed race).
		GBufferSet s;
		TextureDesc cd; cd.Name = "GBuffer"; cd.Type = RESOURCE_DIM_TEX_2D; cd.Width = (Uint32)w; cd.Height = (Uint32)h;
		cd.Format = TEX_FORMAT_RGBA16_FLOAT; cd.BindFlags = BIND_RENDER_TARGET | BIND_SHADER_RESOURCE;
		device->CreateTexture(cd, nullptr, &s.color);
		if (s.color) { s.rtv = s.color->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET); s.srv = s.color->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE); }

		TextureDesc vd; vd.Name = "GBuffer Velocity"; vd.Type = RESOURCE_DIM_TEX_2D; vd.Width = (Uint32)w; vd.Height = (Uint32)h;
		vd.Format = TEX_FORMAT_RG16_FLOAT; vd.BindFlags = BIND_RENDER_TARGET | BIND_SHADER_RESOURCE;   // screen-space motion (TAA)
		device->CreateTexture(vd, nullptr, &s.vel);
		if (s.vel) { s.velRTV = s.vel->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET); s.velSRV = s.vel->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE); }

		// Generic per-OBJECT id: one flat value per draw (pivot hash, written by gbuffer.vs).
		// The G-buffer carries NO effect semantics — consumers (musicvis note tint; outlines /
		// per-object masks later) derive their own meaning from the id.
		TextureDesc nd; nd.Name = "GBuffer ObjectId"; nd.Type = RESOURCE_DIM_TEX_2D; nd.Width = (Uint32)w; nd.Height = (Uint32)h;
		nd.Format = TEX_FORMAT_R8_UNORM; nd.BindFlags = BIND_RENDER_TARGET | BIND_SHADER_RESOURCE;
		device->CreateTexture(nd, nullptr, &s.objId);
		if (s.objId) { s.objIdRTV = s.objId->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET); s.objIdSRV = s.objId->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE); }

		TextureDesc dd; dd.Name = "GBuffer Depth"; dd.Type = RESOURCE_DIM_TEX_2D; dd.Width = (Uint32)w; dd.Height = (Uint32)h;
		dd.Format = TEX_FORMAT_D32_FLOAT; dd.BindFlags = BIND_DEPTH_STENCIL | BIND_SHADER_RESOURCE;
		device->CreateTexture(dd, nullptr, &s.depth);
		if (s.depth) { s.dsv = s.depth->GetDefaultView(TEXTURE_VIEW_DEPTH_STENCIL); s.depthSRV = s.depth->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE); }

		it = gbufCache.emplace(key, std::move(s)).first;
	}

	GBufferSet& set = it->second;
	set.lastUsed = ++gbufFrameCtr;
	gbufCurKey = key;

	// Repoint the live members at the active set — consumers (SSR/RT/Post) read these fields.
	gbufColor = set.color;  gbufDepth = set.depth;  gbufVel = set.vel;  gbufObjId = set.objId;
	gbufRTV = set.rtv; gbufSRV = set.srv; gbufDSV = set.dsv; gbufDepthSRV = set.depthSRV;
	gbufVelRTV = set.velRTV; gbufVelSRV = set.velSRV;
	gbufObjIdRTV = set.objIdRTV; gbufObjIdSRV = set.objIdSRV;
	gbufW = w; gbufH = h;

	EvictGBufferCache();
}

// Bound the cache (main size + preview size + a little slack for drag-resize transients).
// Eviction only drops the RefCntAutoPtrs; Diligent frees the underlying D3D12 memory after
// the frame fence, so evicting a size used a frame ago is safe. The active set is never evicted.
void NukeDiligent::Impl::EvictGBufferCache()
{
	const size_t CAP = 4;
	while (gbufCache.size() > CAP)
	{
		uint64_t lruKey = 0, lru = ~0ull; bool found = false;
		for (auto& kv : gbufCache)
		{
			if (kv.first == gbufCurKey) continue;   // never evict the set in use this frame
			if (kv.second.lastUsed < lru) { lru = kv.second.lastUsed; lruKey = kv.first; found = true; }
		}
		if (!found) break;
		gbufCache.erase(lruKey);
	}
}

// G-buffer prepass pipeline: world.vs (shares worldCB) + gbuffer.ps (shares worldMatCB). Single-sample, depth
// write on; outputs the packed surface buffer. Rebuild-safe.
bool NukeDiligent::Impl::BuildGBufferPipe()
{
	gbufPSO.Release(); gbufSRB.Release(); gbufMRVar = nullptr;
	std::string vsSrc = shaderSource("gbuffer.vs"), psSrc = shaderSource("gbuffer.ps");   // velocity-aware VS (motion vectors)
	if (vsSrc.empty() || psSrc.empty()) return false;
	ShaderCreateInfo sci; sci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
	RefCntAutoPtr<IShader> vs, ps;
	sci.Desc = {"GBuffer VS", SHADER_TYPE_VERTEX, true}; sci.Source = vsSrc.c_str(); device->CreateShader(sci, &vs);
	sci.Desc = {"GBuffer PS", SHADER_TYPE_PIXEL, true};  sci.Source = psSrc.c_str(); device->CreateShader(sci, &ps);
	if (!vs || !ps) return false;

	GraphicsPipelineStateCreateInfo ci; ci.PSODesc.Name = "GBuffer PSO";
	auto& gp = ci.GraphicsPipeline;
	gp.NumRenderTargets = 3; gp.RTVFormats[0] = TEX_FORMAT_RGBA16_FLOAT; gp.RTVFormats[1] = TEX_FORMAT_RG16_FLOAT;   // gbuffer + velocity
	gp.RTVFormats[2] = TEX_FORMAT_R8_UNORM;   // generic per-object id
	gp.DSVFormat = TEX_FORMAT_D32_FLOAT;
	gp.PrimitiveTopology = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	gp.RasterizerDesc.CullMode = CULL_MODE_BACK;
	gp.DepthStencilDesc.DepthEnable = True; gp.DepthStencilDesc.DepthWriteEnable = True;
	gp.SmplDesc.Count = 1;   // 1x — its own depth, no MSAA resolve needed for SSR
	LayoutElement layout[] = { {0, 0, 3, VT_FLOAT32}, {1, 1, 3, VT_FLOAT32}, {2, 2, 2, VT_FLOAT32} };
	gp.InputLayout.NumElements = 3; gp.InputLayout.LayoutElements = layout;

	ShaderResourceVariableDesc vars[] = {
		{SHADER_TYPE_PIXEL, "g_MetalRough", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
		{SHADER_TYPE_PIXEL, "g_Normal",     SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
	};
	ci.PSODesc.ResourceLayout.Variables = vars; ci.PSODesc.ResourceLayout.NumVariables = 2;
	SamplerDesc samp; samp.MinFilter = FILTER_TYPE_LINEAR; samp.MagFilter = FILTER_TYPE_LINEAR; samp.MipFilter = FILTER_TYPE_LINEAR;
	samp.AddressU = TEXTURE_ADDRESS_WRAP; samp.AddressV = TEXTURE_ADDRESS_WRAP;
	ImmutableSamplerDesc imm[] = {   // per-texture samplers pair by name (D3D12-strict combined samplers)
		{SHADER_TYPE_PIXEL, "g_MetalRough", samp},
		{SHADER_TYPE_PIXEL, "g_Normal",     samp},
	};
	ci.PSODesc.ResourceLayout.ImmutableSamplers = imm; ci.PSODesc.ResourceLayout.NumImmutableSamplers = 2;
	ci.pVS = vs; ci.pPS = ps;
	device->CreateGraphicsPipelineState(ci, &gbufPSO);
	if (!gbufPSO) { cout << "[NukeDiligent]\tgbuffer PSO build failed" << endl; return false; }
	if (auto* c = gbufPSO->GetStaticVariableByName(SHADER_TYPE_VERTEX, "CB"))    c->Set(worldCB);
	if (auto* m = gbufPSO->GetStaticVariableByName(SHADER_TYPE_PIXEL,  "MatCB")) m->Set(worldMatCB);
	gbufPSO->CreateShaderResourceBinding(&gbufSRB, true);
	gbufMRVar  = gbufSRB->GetVariableByName(SHADER_TYPE_PIXEL, "g_MetalRough");
	gbufNrmVar = gbufSRB->GetVariableByName(SHADER_TYPE_PIXEL, "g_Normal");
	return true;
}

// Screen-space reflections pass: ray-march the prepass G-buffer/depth, blend onto the chain colour. Falls back
// to a passthrough (handled by the caller) when no prepass ran.
void NukeDiligent::Impl::RunSSR(PostPipe& pp, ITextureView* srcSRV, ITextureView* dstRTV, int w, int h, const std::vector<float>& params)
{
	{
		struct SSRData { float4x4 view, proj, invProj, invView; float res[4]; };
		MapHelper<SSRData> cb(context, ssrCB, MAP_WRITE, MAP_FLAG_DISCARD);
		cb->view = curView; cb->proj = curProjNoJitter; cb->invProj = curProjNoJitter.Inverse();   // unjittered — matches the unjittered gbuffer depth (TAA jitter must not leak into SSR)
		cb->invView = curView.Inverse();   // view -> WORLD (musicvis reconstructs world positions for its note hash)
		cb->res[0] = (float)w; cb->res[1] = (float)h; cb->res[2] = w ? 1.0f / w : 0.0f; cb->res[3] = h ? 1.0f / h : 0.0f;
	}
	{
		MapHelper<float> cbp(context, postParamsCB, MAP_WRITE, MAP_FLAG_DISCARD);
		int n = (int)params.size(); if (n > 64) n = 64;
		for (int k = 0; k < 64; ++k) cbp[k] = (k < n) ? params[k] : 0.0f;
	}
	context->SetRenderTargets(1, &dstRTV, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	Viewport vp; vp.TopLeftX = 0; vp.TopLeftY = 0; vp.Width = (float)w; vp.Height = (float)h; vp.MinDepth = 0; vp.MaxDepth = 1;
	context->SetViewports(1, &vp, w, h);
	if (pp.srcVar)   pp.srcVar->Set(srcSRV);
	if (pp.gbufVar)  pp.gbufVar->Set(gbufSRV);
	if (pp.depthVar) pp.depthVar->Set(gbufDepthSRV);
	if (pp.objIdVar && gbufObjIdSRV) pp.objIdVar->Set(gbufObjIdSRV);   // musicvis: generic per-object id
	context->SetPipelineState(pp.pso);
	context->CommitShaderResources(pp.srb, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	DrawAttribs da{3, DRAW_FLAG_VERIFY_STATES};
	context->Draw(da);
}

// Temporal AA resolve: reproject the per-camera history by the current (unjittered) depth + prev view/proj, clamp
// to the local colour neighbourhood, blend, and copy the result back into the history for next frame.
void NukeDiligent::Impl::RunTAA(PostPipe& pp, ITextureView* srcSRV, ITexture* dstTex, int w, int h, const std::vector<float>& params)
{
	TAAState& st = taaStates[curTarget];
	// The history must MATCH the resolve target's format (it is COPIED from dstTex each
	// frame; a cross-format CopyTextureRegion is invalid in D3D12) — the target can be
	// RGBA8 (scene, HDR off) or RGBA16F (chain scratch), so follow it, don't hardcode.
	const TEXTURE_FORMAT histFmt = dstTex->GetDesc().Format;
	if (!st.hist || st.w != w || st.h != h || st.hist->GetDesc().Format != histFmt)
	{
		st.hist.Release();
		TextureDesc td; td.Name = "TAA history"; td.Type = RESOURCE_DIM_TEX_2D; td.Width = (Uint32)w; td.Height = (Uint32)h;
		td.MipLevels = 1; td.Format = histFmt; td.BindFlags = BIND_SHADER_RESOURCE | BIND_RENDER_TARGET; td.Usage = USAGE_DEFAULT;
		device->CreateTexture(td, nullptr, &st.hist);
		st.w = w; st.h = h; st.valid = false;
	}
	if (!st.hist) return;
	{
		struct TAAData { float4x4 invProj, invView, prevView, prevProj; float res[4]; float flags[4]; };
		MapHelper<TAAData> cb(context, taaCB, MAP_WRITE, MAP_FLAG_DISCARD);
		cb->invProj  = curProjNoJitter.Inverse(); cb->invView = curView.Inverse();
		cb->prevView = st.valid ? st.prevView : curView;
		cb->prevProj = st.valid ? st.prevProj : curProjNoJitter;
		cb->res[0] = (float)w; cb->res[1] = (float)h; cb->res[2] = w ? 1.0f / w : 0.0f; cb->res[3] = h ? 1.0f / h : 0.0f;
		cb->flags[0] = st.valid ? 1.0f : 0.0f; cb->flags[1] = cb->flags[2] = cb->flags[3] = 0.0f;
	}
	{
		MapHelper<float> cbp(context, postParamsCB, MAP_WRITE, MAP_FLAG_DISCARD);
		int n = (int)params.size(); if (n > 64) n = 64;
		for (int k = 0; k < 64; ++k) cbp[k] = (k < n) ? params[k] : 0.0f;
	}
	ITextureView* dstRTV = dstTex->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET);
	context->SetRenderTargets(1, &dstRTV, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	Viewport vp; vp.TopLeftX = 0; vp.TopLeftY = 0; vp.Width = (float)w; vp.Height = (float)h; vp.MinDepth = 0; vp.MaxDepth = 1;
	context->SetViewports(1, &vp, w, h);
	if (pp.srcVar)   pp.srcVar->Set(srcSRV);
	if (pp.depthVar) pp.depthVar->Set(gbufDepthSRV);
	if (pp.histVar)  pp.histVar->Set(st.hist->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
	if (pp.velVar)   pp.velVar->Set(gbufVelSRV);
	context->SetPipelineState(pp.pso);
	context->CommitShaderResources(pp.srb, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	DrawAttribs da{3, DRAW_FLAG_VERIFY_STATES};
	context->Draw(da);

	// Copy the resolved result into the history for next frame; remember this frame's (unjittered) view/proj.
	// dstTex is still bound as the render target — unbind explicitly or Diligent nags every frame.
	context->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE);
	CopyTextureAttribs cp; cp.pSrcTexture = dstTex; cp.pDstTexture = st.hist;
	cp.SrcTextureTransitionMode = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
	cp.DstTextureTransitionMode = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
	context->CopyTexture(cp);
	st.prevView = curView; st.prevProj = curProjNoJitter; st.valid = true;
}

// --- SSR G-buffer prepass (single-sample): normal/roughness/metalness + depth, before the colour pass --------
void NukeDiligent::beginGBufferPass(const NukeCameraDesc& cam)
{
	m_impl->gbufActive = false;
	if (!m_impl->gbufPSO) return;
	int w = 0, h = 0;
	if (!m_impl->CameraSize(cam, w, h)) return;
	m_impl->EnsureGBuffer(w, h);
	if (!m_impl->gbufRTV || !m_impl->gbufDSV) return;
	m_impl->curTarget = cam.target;   // TAA per-camera state (taaStates[curTarget]) keyed correctly during the prepass
	m_impl->SetCameraViewProj(cam, w, h);
	IDeviceContext* ctx = m_impl->context;
	ITextureView* rtvs[3] = { m_impl->gbufRTV, m_impl->gbufVelRTV, m_impl->gbufObjIdRTV };   // MRT: gbuffer + velocity + object id
	Uint32 nrt = 1;
	if (m_impl->gbufVelRTV)  nrt = 2;
	if (m_impl->gbufVelRTV && m_impl->gbufObjIdRTV) nrt = 3;   // slots must be contiguous (PSO declares 3)
	ctx->SetRenderTargets(nrt, rtvs, m_impl->gbufDSV, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	const float clr[4] = { 0, 0, 0, 0 };
	ctx->ClearRenderTarget(m_impl->gbufRTV, clr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	if (m_impl->gbufVelRTV)   ctx->ClearRenderTarget(m_impl->gbufVelRTV, clr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	if (m_impl->gbufObjIdRTV) ctx->ClearRenderTarget(m_impl->gbufObjIdRTV, clr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	ctx->ClearDepthStencil(m_impl->gbufDSV, CLEAR_DEPTH_FLAG, 1.f, 0, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	Viewport vp; vp.TopLeftX = 0; vp.TopLeftY = 0; vp.Width = (float)w; vp.Height = (float)h; vp.MinDepth = 0; vp.MaxDepth = 1;
	ctx->SetViewports(1, &vp, w, h);
	m_impl->gbufActive = true;
}

void NukeDiligent::renderGBufferObject(Mesh* mesh, Material* mat, const float pos[3], const float quat[4], const float scale[3],
                                       const float prevPos[3], const float prevQuat[4], const float prevScale[3])
{
	if (!m_impl->gbufActive || !m_impl->gbufPSO) return;
	Impl::MeshGPU* gp = m_impl->GetMeshGPU(mesh);
	if (!gp) return;
	++m_impl->statDraws;                              // frame stats (status bar)
	m_impl->statTris += mesh ? mesh->numVerts / 3 : 0;
	Impl::MeshGPU& g = *gp;

	auto build = [](const float p[3], const float q[4], const float s[3]) {
		return float4x4::Scale(s[0], s[1], s[2]) * Diligent::Quaternion<float>(q[0], q[1], q[2], q[3]).ToMatrix() * float4x4::Translation(p[0], p[1], p[2]);
	};
	float4x4 world = build(pos, quat, scale);
	float4x4 wvp   = world * m_impl->curView * m_impl->curProj;   // prepass is UNjittered
	// Previous-frame clip: prev object transform * previous camera (from this camera's TAA state). Falls back to
	// the current transforms (zero velocity) when there's no prev transform / no history yet.
	Impl::TAAState& tst = m_impl->taaStates[m_impl->curTarget];
	float4x4 prevWorld = (prevPos && prevQuat && prevScale) ? build(prevPos, prevQuat, prevScale) : world;
	float4x4 prevWVP   = tst.valid ? (prevWorld * tst.prevView * tst.prevProj) : wvp;
	struct CBData { float4x4 wvp; float4x4 world; float4x4 prevWVP; };
	{ MapHelper<CBData> cb(m_impl->context, m_impl->worldCB, MAP_WRITE, MAP_FLAG_DISCARD); cb->wvp = wvp; cb->world = world; cb->prevWVP = prevWVP; }

	float metallic = 0.0f, roughness = 0.6f; ITextureView* mrsrv = nullptr; ITextureView* nsrv = nullptr;
	if (mat) { metallic = mat->metallic; roughness = mat->roughness;
	           if (mat->mr) mrsrv = m_impl->GetTexSRV(mat->mr); if (mat->norm) nsrv = m_impl->GetTexSRV(mat->norm); }
	{
		MapHelper<Uint8> mb(m_impl->context, m_impl->worldMatCB, MAP_WRITE, MAP_FLAG_DISCARD);
		Uint8* p = mb; memset(p, 0, Impl::kMatCBBytes);
		float nrmY = nsrv ? ((mat && mat->norm && !mat->norm->invertGreen) ? -1.0f : 1.0f) : 0.0f;   // sign = green convention
		float prm[4]  = { 0, nrmY, metallic, roughness };   // g_Params (_, hasNormal±greenConv, metallic.z, roughness.w)
		memcpy(p + 16, prm, sizeof(float) * 4);
		float prm2[4] = { mrsrv ? 1.0f : 0.0f, 0, 0, 1.0f };   // g_Params2 (hasMR.x)
		memcpy(p + 32, prm2, sizeof(float) * 4);
	}
	if (m_impl->gbufMRVar)
		m_impl->gbufMRVar->Set(mrsrv ? mrsrv : m_impl->whiteTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
	if (m_impl->gbufNrmVar)
		m_impl->gbufNrmVar->Set(nsrv ? nsrv : m_impl->flatNormTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));

	IDeviceContext* ctx = m_impl->context;
	IBuffer* vbs[] = { g.pos, g.nrm, g.uv }; Uint64 offs[] = { 0, 0, 0 };
	ctx->SetVertexBuffers(0, 3, vbs, offs, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, SET_VERTEX_BUFFERS_FLAG_RESET);
	ctx->SetPipelineState(m_impl->gbufPSO);
	ctx->CommitShaderResources(m_impl->gbufSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	DrawAttribs da{ (Uint32)g.numVerts, DRAW_FLAG_VERIFY_STATES };
	ctx->Draw(da);
}

void NukeDiligent::endGBufferPass() { /* gbufActive stays set so endCamera's SSR pass can sample it; beginCamera rebinds the colour target */ }
