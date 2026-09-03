#include "NukeDiligentImpl.h"
#include <array>


// Make the G-buffer set for w*h current: RGBA16F (octN.xy, rough, metal) + velocity + object id
// + D32 depth, all single-sample and shader-readable. Sets are cached per size.
void NukeDiligent::Impl::EnsureGBuffer(int w, int h)
{
	if (w <= 0 || h <= 0) return;
	if (gbufColor && gbufW == w && gbufH == h) return;   // already the active set

	const uint64_t key = ((uint64_t)(uint32_t)w << 32) | (uint32_t)h;
	auto it = gbufCache.find(key);
	if (it == gbufCache.end())
	{
		// Build a fresh set; never release the other sizes here — freeing an in-use buffer mid-frame removes the device.
		GBufferSet s;
		TextureDesc cd; cd.Name = "GBuffer"; cd.Type = RESOURCE_DIM_TEX_2D; cd.Width = (Uint32)w; cd.Height = (Uint32)h;
		cd.Format = TEX_FORMAT_RGBA16_FLOAT; cd.BindFlags = BIND_RENDER_TARGET | BIND_SHADER_RESOURCE;
		device->CreateTexture(cd, nullptr, &s.color);
		if (s.color) { s.rtv = s.color->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET); s.srv = s.color->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE); }

		TextureDesc vd; vd.Name = "GBuffer Velocity"; vd.Type = RESOURCE_DIM_TEX_2D; vd.Width = (Uint32)w; vd.Height = (Uint32)h;
		vd.Format = TEX_FORMAT_RG16_FLOAT; vd.BindFlags = BIND_RENDER_TARGET | BIND_SHADER_RESOURCE;   // screen-space motion (TAA)
		device->CreateTexture(vd, nullptr, &s.vel);
		if (s.vel) { s.velRTV = s.vel->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET); s.velSRV = s.vel->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE); }

		// Generic per-object id: one flat value per draw (pivot hash from gbuffer.vs); consumers assign it meaning.
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

	gbufColor = set.color;  gbufDepth = set.depth;  gbufVel = set.vel;  gbufObjId = set.objId;
	gbufRTV = set.rtv; gbufSRV = set.srv; gbufDSV = set.dsv; gbufDepthSRV = set.depthSRV;
	gbufVelRTV = set.velRTV; gbufVelSRV = set.velSRV;
	gbufObjIdRTV = set.objIdRTV; gbufObjIdSRV = set.objIdSRV;
	gbufW = w; gbufH = h;

	EvictGBufferCache();
}

// Drop least-recently-used G-buffer sets down to a fixed cap. The active set is never evicted.
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
		// Evict via the trash: an evicted size's SRVs may still sit in recorded draw data.
		auto it = gbufCache.find(lruKey);
		Trash(it->second.color); Trash(it->second.depth); Trash(it->second.vel); Trash(it->second.objId);
		gbufCache.erase(it);
	}
}

// Overlay props + the per-draw condition context into a mapped G-buffer MatCB. The gbuffer
// cbuffer mirrors the world shader's layout, so the engine-parsed offsets apply verbatim.
static void FillGBufOverlays(Diligent::Uint8* p, nuke::Material* mat)
{
	using Impl = NukeDiligent::Impl;
	if (!mat || !mat->shader) return;
	for (const nuke::ShaderProp& sp : mat->shader->props)
	{
		if (sp.name.compare(0, 4, "g_Ov") != 0 && sp.name != "g_Det" && sp.name != "g_Var"
		    && sp.name.compare(0, 6, "g_Brdf") != 0) continue;
		auto pv = mat->props.find(sp.name);
		if (pv == mat->props.end()) continue;
		uint32_t bytes = (uint32_t)sp.components * sizeof(float);
		if (sp.offset + bytes <= Impl::kMatCBBytes) memcpy(p + sp.offset, pv->second.data(), bytes);
	}
	if (mat->liveOvCount <= 0 || !mat->liveDrawSet) return;
	static const auto kOvV = []{ std::array<std::string, Impl::kOvSlots> a; for (int s = 0; s < Impl::kOvSlots; ++s) a[s] = "g_Ov"  + std::to_string(s); return a; }();
	static const auto kOvP = []{ std::array<std::string, Impl::kOvSlots> a; for (int s = 0; s < Impl::kOvSlots; ++s) a[s] = "g_OvP" + std::to_string(s); return a; }();
	static const char* const kOvM[3] = { "g_OvM0", "g_OvM1", "g_OvM2" };
	for (const nuke::ShaderProp& sp : mat->shader->props)
	{
		if (sp.components != 4 || sp.offset + 16 > Impl::kMatCBBytes) continue;
		float* d = (float*)(p + sp.offset);
		for (int sl = 0; sl < Impl::kOvSlots; ++sl)
		{
			if (sp.name == kOvV[sl] && mat->liveDrawValue[sl] >= 0.0f) d[0] = mat->liveDrawValue[sl];
			if (sp.name == kOvP[sl]) d[2] = mat->liveDrawMaskChan[sl];
		}
		if (mat->liveDrawMask3D)
		{
			for (int r = 0; r < 3; ++r)
				if (sp.name == kOvM[r]) memcpy(d, &mat->liveDrawMaskXform[r * 4], 16);
			if (sp.name == "g_OvMQ") { d[0] = mat->liveDrawMaskRes; d[1] = 1.0f; }
		}
	}
}

// Build the G-buffer prepass PSOs (plain + instanced). Single-sample, depth write on.
// Returns true when the plain PSO is ready.
bool NukeDiligent::Impl::BuildGBufferPipe()
{
	gbufPSO.Release(); gbufSRB.Release(); gbufMRVar = nullptr;
	std::string vsSrc = shaderSource("gbuffer.vs"), psSrc = shaderSource("gbuffer.ps");   // velocity-aware VS (motion vectors)
	if (vsSrc.empty() || psSrc.empty()) return false;
	ShaderCreateInfo sci; sci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
	auto sfLocal = ShaderFactory();   // held for the compiles below
	sci.pShaderSourceStreamFactory = sfLocal;
	RefCntAutoPtr<IShader> vs, ps;
	sci.Desc = {"GBuffer VS", SHADER_TYPE_VERTEX, true}; sci.Source = vsSrc.c_str(); CreateShaderCached(sci, &vs);
	sci.Desc = {"GBuffer PS", SHADER_TYPE_PIXEL, true};  sci.Source = psSrc.c_str(); CreateShaderCached(sci, &ps);
	if (!vs || !ps) return false;

	GraphicsPipelineStateCreateInfo ci; ci.PSODesc.Name = "GBuffer PSO";
	auto& gp = ci.GraphicsPipeline;
	gp.NumRenderTargets = 3; gp.RTVFormats[0] = TEX_FORMAT_RGBA16_FLOAT; gp.RTVFormats[1] = TEX_FORMAT_RG16_FLOAT;   // gbuffer + velocity
	gp.RTVFormats[2] = TEX_FORMAT_R8_UNORM;   // generic per-object id
	gp.DSVFormat = TEX_FORMAT_D32_FLOAT;
	gp.PrimitiveTopology = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	// The prepass depth must match what the beauty pass SHOWS — and the world PSO is
	// two-sided (CULL_NONE). Back-culling here made reversed-winding meshes (builtin
	// sphere) write their INNER wall into the depth: decals/SSR then reconstructed the
	// inside of the surface. Nearest-depth wins either way, so two-sided is simply correct.
	gp.RasterizerDesc.CullMode = CULL_MODE_NONE;
	gp.DepthStencilDesc.DepthEnable = True; gp.DepthStencilDesc.DepthWriteEnable = True;
	gp.SmplDesc.Count = 1;   // 1x — its own depth, no MSAA resolve needed for SSR
	LayoutElement layout[] = { {0, 0, 3, VT_FLOAT32}, {1, 1, 3, VT_FLOAT32}, {2, 2, 2, VT_FLOAT32} };
	gp.InputLayout.NumElements = 3; gp.InputLayout.LayoutElements = layout;

	std::vector<ShaderResourceVariableDesc> vars = {
		{SHADER_TYPE_PIXEL, "g_MetalRough", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
		{SHADER_TYPE_PIXEL, "g_Normal",     SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
		{SHADER_TYPE_PIXEL, "g_Tex",        SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},   // base alpha for cutout
		{SHADER_TYPE_PIXEL, "g_WipeMask",   SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},   // luma-wipe holes
	};
	// Overlay slots shape the G-buffer normal/roughness too; one shared sampler on g_Ov0Nrm.
	for (const std::string& n : OvGbufNames())
		vars.push_back({SHADER_TYPE_PIXEL, n.c_str(), SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC});
	ci.PSODesc.ResourceLayout.Variables = vars.data(); ci.PSODesc.ResourceLayout.NumVariables = (Uint32)vars.size();
	SamplerDesc samp; samp.MinFilter = FILTER_TYPE_LINEAR; samp.MagFilter = FILTER_TYPE_LINEAR; samp.MipFilter = FILTER_TYPE_LINEAR;
	samp.AddressU = TEXTURE_ADDRESS_WRAP; samp.AddressV = TEXTURE_ADDRESS_WRAP;
	ImmutableSamplerDesc imm[] = {   // per-texture samplers pair by name (D3D12-strict combined samplers)
		{SHADER_TYPE_PIXEL, "g_MetalRough", samp},
		{SHADER_TYPE_PIXEL, "g_Normal",     samp},
		{SHADER_TYPE_PIXEL, "g_Tex",        samp},
		{SHADER_TYPE_PIXEL, "g_WipeMask",   samp},
		{SHADER_TYPE_PIXEL, "g_Ov0Nrm",     samp},   // shared by the whole overlay block
	};
	ci.PSODesc.ResourceLayout.ImmutableSamplers = imm; ci.PSODesc.ResourceLayout.NumImmutableSamplers = 5;
	ci.pVS = vs; ci.pPS = ps;
	CreateGraphicsPipelineStateCached(ci, &gbufPSO);
	if (!gbufPSO) { cout << "[NukeDiligent]\tgbuffer PSO build failed" << endl; return false; }
	if (auto* c = gbufPSO->GetStaticVariableByName(SHADER_TYPE_VERTEX, "CB"))    c->Set(worldCB);
	if (auto* m = gbufPSO->GetStaticVariableByName(SHADER_TYPE_PIXEL,  "MatCB")) m->Set(worldMatCB);
	gbufPSO->CreateShaderResourceBinding(&gbufSRB, true);
	gbufMRVar  = gbufSRB->GetVariableByName(SHADER_TYPE_PIXEL, "g_MetalRough");
	gbufNrmVar = gbufSRB->GetVariableByName(SHADER_TYPE_PIXEL, "g_Normal");
	gbufTexVar  = gbufSRB->GetVariableByName(SHADER_TYPE_PIXEL, "g_Tex");
	gbufWipeVar = gbufSRB->GetVariableByName(SHADER_TYPE_PIXEL, "g_WipeMask");
	for (int k = 0; k < kOvGbufCount; ++k)
		gbufOvVar[k] = gbufSRB->GetVariableByName(SHADER_TYPE_PIXEL, OvGbufNames()[k].c_str());
	memset(gbufOvLast, 0, sizeof(gbufOvLast));

	// Instanced twin: per-instance world rows; the CB carries current + previous camera view*proj
	// only, so instanced velocity is camera-only (gbuffer.vs.hlsl NUKE_INSTANCED).
	gbufPSOInst.Release(); gbufSRBInst.Release(); gbufMRVarInst = nullptr; gbufNrmVarInst = nullptr;
	{
		const std::string vsI = "#define NUKE_INSTANCED 1\n" + vsSrc;
		const std::string psI = "#define NUKE_INSTANCED 1\n" + psSrc;
		RefCntAutoPtr<IShader> vsi, psi;
		sci.Desc = {"GBuffer VS (inst)", SHADER_TYPE_VERTEX, true}; sci.Source = vsI.c_str(); CreateShaderCached(sci, &vsi);
		sci.Desc = {"GBuffer PS (inst)", SHADER_TYPE_PIXEL, true};  sci.Source = psI.c_str(); CreateShaderCached(sci, &psi);
		if (vsi && psi)
		{
			LayoutElement layoutI[] = {
				{0, 0, 3, VT_FLOAT32}, {1, 1, 3, VT_FLOAT32}, {2, 2, 2, VT_FLOAT32},
				{3, 3, 4, VT_FLOAT32, False, LAYOUT_ELEMENT_AUTO_OFFSET, LAYOUT_ELEMENT_AUTO_STRIDE, INPUT_ELEMENT_FREQUENCY_PER_INSTANCE},
				{4, 3, 4, VT_FLOAT32, False, LAYOUT_ELEMENT_AUTO_OFFSET, LAYOUT_ELEMENT_AUTO_STRIDE, INPUT_ELEMENT_FREQUENCY_PER_INSTANCE},
				{5, 3, 4, VT_FLOAT32, False, LAYOUT_ELEMENT_AUTO_OFFSET, LAYOUT_ELEMENT_AUTO_STRIDE, INPUT_ELEMENT_FREQUENCY_PER_INSTANCE},
				{6, 3, 4, VT_FLOAT32, False, LAYOUT_ELEMENT_AUTO_OFFSET, LAYOUT_ELEMENT_AUTO_STRIDE, INPUT_ELEMENT_FREQUENCY_PER_INSTANCE},
				{7, 3, 4, VT_FLOAT32, False, LAYOUT_ELEMENT_AUTO_OFFSET, LAYOUT_ELEMENT_AUTO_STRIDE, INPUT_ELEMENT_FREQUENCY_PER_INSTANCE},
			};
			gp.InputLayout.LayoutElements = layoutI;
			gp.InputLayout.NumElements    = 8;
			ci.PSODesc.Name = "GBuffer PSO (inst)";
			ci.pVS = vsi; ci.pPS = psi;
			CreateGraphicsPipelineStateCached(ci, &gbufPSOInst);
			if (gbufPSOInst)
			{
				if (auto* c = gbufPSOInst->GetStaticVariableByName(SHADER_TYPE_VERTEX, "CB"))    c->Set(worldCB);
				if (auto* m = gbufPSOInst->GetStaticVariableByName(SHADER_TYPE_PIXEL,  "MatCB")) m->Set(worldMatCB);
				if (auto* b = gbufPSOInst->GetStaticVariableByName(SHADER_TYPE_VERTEX, "BendCB")) b->Set(bendCB);   // depth/velocity must bend like the lit pass (7.4)
				gbufPSOInst->CreateShaderResourceBinding(&gbufSRBInst, true);
				// Vulkan: cbuffers may reflect MUTABLE — bind BendCB via the SRB too.
				if (auto* b = gbufSRBInst->GetVariableByName(SHADER_TYPE_VERTEX, "BendCB")) b->Set(bendCB);
				gbufMRVarInst  = gbufSRBInst->GetVariableByName(SHADER_TYPE_PIXEL, "g_MetalRough");
				gbufNrmVarInst = gbufSRBInst->GetVariableByName(SHADER_TYPE_PIXEL, "g_Normal");
				gbufTexVarInst  = gbufSRBInst->GetVariableByName(SHADER_TYPE_PIXEL, "g_Tex");
				gbufWipeVarInst = gbufSRBInst->GetVariableByName(SHADER_TYPE_PIXEL, "g_WipeMask");
				for (int k = 0; k < kOvGbufCount; ++k)
					gbufOvVarInst[k] = gbufSRBInst->GetVariableByName(SHADER_TYPE_PIXEL, OvGbufNames()[k].c_str());
			}
		}
	}

	// Skinned twin (gbuffer.vs.hlsl NUKE_SKINNED): stream 3 = previous-frame skinned positions
	// from skin.cs — velocity is the TRUE pose delta, not the rigid-transform approximation.
	gbufPSOSkin.Release(); gbufSRBSkin.Release(); gbufMRVarSkin = nullptr; gbufNrmVarSkin = nullptr;
	{
		const std::string vsS = "#define NUKE_SKINNED 1\n" + vsSrc;
		const std::string psS = "#define NUKE_SKINNED 1\n" + psSrc;
		RefCntAutoPtr<IShader> vss, pss;
		sci.Desc = {"GBuffer VS (skin)", SHADER_TYPE_VERTEX, true}; sci.Source = vsS.c_str(); CreateShaderCached(sci, &vss);
		sci.Desc = {"GBuffer PS (skin)", SHADER_TYPE_PIXEL, true};  sci.Source = psS.c_str(); CreateShaderCached(sci, &pss);
		if (vss && pss)
		{
			LayoutElement layoutS[] = {
				{0, 0, 3, VT_FLOAT32}, {1, 1, 3, VT_FLOAT32}, {2, 2, 2, VT_FLOAT32},
				{3, 3, 3, VT_FLOAT32},   // previous-frame skinned position
			};
			gp.InputLayout.LayoutElements = layoutS;
			gp.InputLayout.NumElements    = 4;
			ci.PSODesc.Name = "GBuffer PSO (skin)";
			ci.pVS = vss; ci.pPS = pss;
			CreateGraphicsPipelineStateCached(ci, &gbufPSOSkin);
			if (gbufPSOSkin)
			{
				if (auto* c = gbufPSOSkin->GetStaticVariableByName(SHADER_TYPE_VERTEX, "CB"))    c->Set(worldCB);
				if (auto* m = gbufPSOSkin->GetStaticVariableByName(SHADER_TYPE_PIXEL,  "MatCB")) m->Set(worldMatCB);
				gbufPSOSkin->CreateShaderResourceBinding(&gbufSRBSkin, true);
				gbufMRVarSkin  = gbufSRBSkin->GetVariableByName(SHADER_TYPE_PIXEL, "g_MetalRough");
				gbufNrmVarSkin = gbufSRBSkin->GetVariableByName(SHADER_TYPE_PIXEL, "g_Normal");
				gbufTexVarSkin  = gbufSRBSkin->GetVariableByName(SHADER_TYPE_PIXEL, "g_Tex");
				gbufWipeVarSkin = gbufSRBSkin->GetVariableByName(SHADER_TYPE_PIXEL, "g_WipeMask");
				for (int k = 0; k < kOvGbufCount; ++k)
					gbufOvVarSkin[k] = gbufSRBSkin->GetVariableByName(SHADER_TYPE_PIXEL, OvGbufNames()[k].c_str());
			}
		}
	}
	return true;
}

// Screen-space reflections: ray-march the prepass G-buffer/depth from srcSRV into dstRTV.
void NukeDiligent::Impl::RunSSR(PostPipe& pp, ITextureView* srcSRV, ITextureView* dstRTV, int w, int h, const std::vector<float>& params)
{
	{
		struct SSRData { float4x4 view, proj, invProj, invView; float res[4]; };
		MapHelper<SSRData> cb(context, ssrCB, MAP_WRITE, MAP_FLAG_DISCARD);
		cb->view = curView; cb->proj = curProjNoJitter; cb->invProj = curProjNoJitter.Inverse();   // unjittered: must match the gbuffer depth
		cb->invView = curView.Inverse();   // view -> world
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
	if (pp.objIdVar && gbufObjIdSRV) pp.objIdVar->Set(gbufObjIdSRV);
	context->SetPipelineState(pp.pso);
	context->CommitShaderResources(pp.srb, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	DrawAttribs da{3, DRAW_FLAG_VERIFY_STATES};
	context->Draw(da);
}

// Temporal AA resolve: reproject the per-camera history, neighbourhood-clamp, blend into dstTex,
// then copy the result back into the history for next frame.
void NukeDiligent::Impl::RunTAA(PostPipe& pp, ITextureView* srcSRV, ITexture* dstTex, int w, int h, const std::vector<float>& params)
{
	TAAState& st = taaStates[curCamKey];
	st.lastUsed = frameId;
	// History format must follow dstTex: it is copied from it, and cross-format CopyTexture is invalid in D3D12.
	const TEXTURE_FORMAT histFmt = dstTex->GetDesc().Format;
	if (!st.hist || st.w != w || st.h != h || st.hist->GetDesc().Format != histFmt)
	{
		Trash(st.hist);   // the old history may still be bound in this frame's earlier TAA pass
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

	// Unbind before the copy: dstTex is still bound as the render target.
	context->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE);
	CopyTextureAttribs cp; cp.pSrcTexture = dstTex; cp.pDstTexture = st.hist;
	cp.SrcTextureTransitionMode = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
	cp.DstTextureTransitionMode = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
	context->CopyTexture(cp);
	st.prevView = curView; st.prevProj = curProjNoJitter; st.valid = true;
}

// Begin the single-sample G-buffer prepass (normal/roughness/metalness + depth) for `cam`.
void NukeDiligent::beginGBufferPass(const NukeCameraDesc& cam)
{
	m_impl->GpuPass("gbuffer");   // depth/normal/velocity prepass; "scene" opens in beginCamera
	++m_impl->passSerial;   // invalidate the per-draw redundancy gates (this pass maps the shared CBs itself)
	// The prepass runs BEFORE beginCamera: latch the LOD anchor here too, or the prepass
	// selects LODs against the PREVIOUS camera and mismatches the beauty pass geometry.
	m_impl->lodCamPos[0] = cam.camPos[0]; m_impl->lodCamPos[1] = cam.camPos[1]; m_impl->lodCamPos[2] = cam.camPos[2];
	m_impl->gbufActive = false;
	if (!m_impl->gbufPSO || m_impl->gbufBuilding) return;   // not built yet / being (re)built in the background
	int w = 0, h = 0;
	if (!m_impl->CameraSize(cam, w, h)) return;
	m_impl->EnsureGBuffer(w, h);
	if (!m_impl->gbufRTV || !m_impl->gbufDSV) return;
	m_impl->curTarget = cam.target;   // feedback guard during the prepass
	m_impl->curCamKey = Impl::CamKey(cam);   // keys the per-camera TAA / AO state
	m_impl->SetCameraViewProj(cam, w, h);
	m_impl->gbufView = m_impl->curView; m_impl->gbufProj = m_impl->curProj;
	IDeviceContext* ctx = m_impl->context;
	ITextureView* rtvs[3] = { m_impl->gbufRTV, m_impl->gbufVelRTV, m_impl->gbufObjIdRTV };
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
	m_impl->gbufTarget = cam.target;   // this prepass belongs to THIS camera (see beginCamera)
}

void NukeDiligent::renderGBufferObject(Mesh* mesh, Material* mat, const float pos[3], const float quat[4], const float scale[3],
                                       const float prevPos[3], const float prevQuat[4], const float prevScale[3])
{
	if (!mesh) return;
	uint32_t first = 0, count = 0;
	m_impl->LodRange(mesh, m_impl->SelectLod(mesh, pos, scale), first, count);
	RenderGBufferRange(mesh, mat, pos, quat, scale, prevPos, prevQuat, prevScale, first, count);
}

void NukeDiligent::renderGBufferObjectMulti(Mesh* mesh, Material* const* mats, int matCount,
                                            const float pos[3], const float quat[4], const float scale[3],
                                            const float prevPos[3], const float prevQuat[4], const float prevScale[3],
                                            int blendPass)
{
	if (!mesh) return;
	if (mesh->numIndices <= 0 || mesh->sections.empty())
	{
		Material* m = matCount > 0 ? mats[0] : nullptr;
		const int bm = m ? m->blendMode : 0;
		if (blendPass == 0 && bm != 0) return;
		if (blendPass == 1 && bm == 0) return;
		renderGBufferObject(mesh, m, pos, quat, scale, prevPos, prevQuat, prevScale);
		return;
	}
	MeshLOD L = mesh->Lod(m_impl->SelectLod(mesh, pos, scale));
	for (int s = 0; s < L.sectionCount; ++s)
	{
		MeshSection sec = mesh->Section(L.firstSection + s);
		Material* m = (sec.slot >= 0 && sec.slot < matCount && mats[sec.slot]) ? mats[sec.slot]
		            : (matCount > 0 ? mats[0] : nullptr);
		const int bm = m ? m->blendMode : 0;
		if (blendPass == 0 && bm != 0) continue;
		if (blendPass == 1 && bm == 0) continue;
		RenderGBufferRange(mesh, m, pos, quat, scale, prevPos, prevQuat, prevScale, sec.firstIndex, sec.indexCount);
	}
}

void NukeDiligent::RenderGBufferRange(Mesh* mesh, Material* mat, const float pos[3], const float quat[4], const float scale[3],
                                      const float prevPos[3], const float prevQuat[4], const float prevScale[3],
                                      uint32_t firstIndex, uint32_t indexCount)
{
	if (!m_impl->gbufActive || !m_impl->gbufPSO || indexCount == 0) return;
	Impl::MeshGPU* gp = m_impl->GetMeshGPU(mesh);
	if (!gp) return;
	++m_impl->statDraws;
	m_impl->statTris += (int)indexCount / 3;
	Impl::MeshGPU& g = *gp;

	auto build = [](const float p[3], const float q[4], const float s[3]) {
		return float4x4::Scale(s[0], s[1], s[2]) * Diligent::Quaternion<float>(q[0], q[1], q[2], q[3]).ToMatrix() * float4x4::Translation(p[0], p[1], p[2]);
	};
	float4x4 world = build(pos, quat, scale);
	float4x4 wvp   = world * m_impl->curView * m_impl->curProj;   // prepass is UNjittered
	// Previous-frame clip = prev object transform * previous camera; falls back to current (zero velocity).
	Impl::TAAState& tst = m_impl->taaStates[m_impl->curCamKey];
	float4x4 prevWorld = (prevPos && prevQuat && prevScale) ? build(prevPos, prevQuat, prevScale) : world;
	float4x4 prevWVP   = tst.valid ? (prevWorld * tst.prevView * tst.prevProj) : wvp;
	struct CBData { float4x4 wvp; float4x4 world; float4x4 prevWVP; };
	{ MapHelper<CBData> cb(m_impl->context, m_impl->worldCB, MAP_WRITE, MAP_FLAG_DISCARD); cb->wvp = wvp; cb->world = world; cb->prevWVP = prevWVP; }

	float metallic = 0.0f, roughness = 0.6f; ITextureView* mrsrv = nullptr; ITextureView* nsrv = nullptr;
	ITextureView* texsrv = nullptr; ITextureView* wipesrv = nullptr;
	if (mat) { metallic = mat->metallic; roughness = mat->roughness;
	           if (mat->mr) mrsrv = m_impl->GetTexSRV(mat->mr); if (mat->norm) nsrv = m_impl->GetTexSRV(mat->norm);
	           if (mat->diff) texsrv = m_impl->GetTexSRV(mat->diff);
	           if (mat->wipe) wipesrv = m_impl->GetTexSRV(mat->wipe); }
	{
		MapHelper<Uint8> mb(m_impl->context, m_impl->worldMatCB, MAP_WRITE, MAP_FLAG_DISCARD);
		Uint8* p = mb; memset(p, 0, Impl::kMatCBBytes);
		float col[4] = { 1, 1, 1, mat ? (float)mat->color.a : 1.0f };   // g_Color (alpha feeds the cutout clip)
		memcpy(p + 0, col, sizeof(float) * 4);
		float nrmY = nsrv ? ((mat && mat->norm && !mat->norm->invertGreen) ? -1.0f : 1.0f) : 0.0f;   // sign = green convention
		float prm[4]  = { texsrv ? 1.0f : 0.0f, nrmY, metallic, roughness };   // g_Params (hasBase, hasNormal±greenConv, metallic, roughness)
		memcpy(p + 16, prm, sizeof(float) * 4);
		float prm2[4] = { mrsrv ? 1.0f : 0.0f, 0, 0, 1.0f };   // g_Params2 (hasMR.x)
		memcpy(p + 32, prm2, sizeof(float) * 4);
		// LiveMaterial: UV transform + cutout/wipe thresholds mirror the color pass.
		float uvt[4]  = { mat ? (float)mat->uvTiling.x : 0.0f, mat ? (float)mat->uvTiling.y : 0.0f,
		                  mat ? (float)mat->uvOffset.x + mat->uvAnim[0] : 0.0f,
		                  mat ? (float)mat->uvOffset.y + mat->uvAnim[1] : 0.0f };
		memcpy(p + 64, uvt, sizeof(float) * 4);
		float uvt2[4] = { mat ? mat->uvRotation * 0.01745329f : 0.0f,
		                  (mat && mat->blendMode == 3) ? mat->alphaCutoff : 0.0f,
		                  (mat && mat->wipe && mat->wipeThreshold > 0.0f) ? mat->wipeThreshold : 0.0f, 0.0f };
		memcpy(p + 80, uvt2, sizeof(float) * 4);
		FillGBufOverlays(p, mat);   // overlay slots shape the SSR normal/roughness too
	}
	// GPU-skinned instance: prev-position stream + the NUKE_SKINNED pipeline (true MVs).
	const bool skinnedDraw = g.skinned && g.skinPosPrev && m_impl->gbufPSOSkin && m_impl->gbufSRBSkin;
	ITextureView* whiteSRV = m_impl->whiteTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
	if (skinnedDraw)
	{
		if (m_impl->gbufMRVarSkin)
			m_impl->gbufMRVarSkin->Set(mrsrv ? mrsrv : whiteSRV);
		if (m_impl->gbufNrmVarSkin)
			m_impl->gbufNrmVarSkin->Set(nsrv ? nsrv : m_impl->flatNormTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
		if (m_impl->gbufTexVarSkin)  m_impl->gbufTexVarSkin->Set(texsrv ? texsrv : whiteSRV);
		if (m_impl->gbufWipeVarSkin) m_impl->gbufWipeVarSkin->Set(wipesrv ? wipesrv : whiteSRV);
	}
	else
	{
		if (m_impl->gbufMRVar)
			m_impl->gbufMRVar->Set(mrsrv ? mrsrv : whiteSRV);
		if (m_impl->gbufNrmVar)
			m_impl->gbufNrmVar->Set(nsrv ? nsrv : m_impl->flatNormTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
		if (m_impl->gbufTexVar)  m_impl->gbufTexVar->Set(texsrv ? texsrv : whiteSRV);
		if (m_impl->gbufWipeVar) m_impl->gbufWipeVar->Set(wipesrv ? wipesrv : whiteSRV);
	}
	// Overlay maps (normal/MR/mask per slot + the painted 3D mask), redundancy-gated per SRB.
	{
		const int srbIdx = skinnedDraw ? 2 : 0;
		IShaderResourceVariable** ovVars = skinnedDraw ? m_impl->gbufOvVarSkin : m_impl->gbufOvVar;
		ITextureView* flatN = m_impl->flatNormTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
		ITextureView* ovsrv[Impl::kOvGbufCount] = {};
		if (mat)
		{
			for (int sl = 0; sl < mat->liveOvCount && sl < nuke::Material::kOverlaySlots; ++sl)
			{
				const nuke::Material::OverlayRT& ov = mat->liveOv[sl];
				if (ov.normal) ovsrv[sl * 3 + 0] = m_impl->GetTexSRV(ov.normal);
				if (ov.mrTex)  ovsrv[sl * 3 + 1] = m_impl->GetTexSRV(ov.mrTex);
				if (ov.mask)   ovsrv[sl * 3 + 2] = m_impl->GetTexSRV(ov.mask);
			}
			if (mat->liveDrawSet && mat->liveDrawMask3D) ovsrv[Impl::kOvSlots * 3] = m_impl->GetTexSRV(mat->liveDrawMask3D);
			if (mat->detailNrm) ovsrv[Impl::kOvSlots * 3 + 1] = m_impl->GetTexSRV(mat->detailNrm);
		}
		for (int k = 0; k < Impl::kOvGbufCount; ++k)
		{
			IDeviceObject* o = ovsrv[k] ? (IDeviceObject*)ovsrv[k]
			                            : (((k < Impl::kOvSlots * 3 && (k % 3) == 0) || k == Impl::kOvSlots * 3 + 1) ? (IDeviceObject*)flatN : (IDeviceObject*)whiteSRV);
			if (ovVars[k] && o != m_impl->gbufOvLast[srbIdx][k]) { ovVars[k]->Set(o); m_impl->gbufOvLast[srbIdx][k] = o; }
		}
	}

	IDeviceContext* ctx = m_impl->context;
	if (skinnedDraw)
	{
		IBuffer* vbs[] = { g.pos, g.nrm, g.uv, g.skinPosPrev }; Uint64 offs[] = { 0, 0, 0, 0 };
		ctx->SetVertexBuffers(0, 4, vbs, offs, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, SET_VERTEX_BUFFERS_FLAG_RESET);
		ctx->SetPipelineState(m_impl->gbufPSOSkin);
		ctx->CommitShaderResources(m_impl->gbufSRBSkin, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	}
	else
	{
		IBuffer* vbs[] = { g.PosBuf(), g.NrmBuf(), g.UVBuf() };
		Uint64 offs[] = { g.PosOfs(), g.NrmOfs(), g.UVOfs() };
		ctx->SetVertexBuffers(0, 3, vbs, offs, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, SET_VERTEX_BUFFERS_FLAG_RESET);
		ctx->SetPipelineState(m_impl->gbufPSO);
		ctx->CommitShaderResources(m_impl->gbufSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	}
	if (g.IdxBuf())
	{
		ctx->SetIndexBuffer(g.IdxBuf(), g.IdxOfs(), RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		DrawIndexedAttribs da{ (Uint32)indexCount, VT_UINT32, DRAW_FLAG_VERIFY_STATES };
		da.FirstIndexLocation = (Uint32)firstIndex;
		ctx->DrawIndexed(da);
	}
	else
	{
		DrawAttribs da{ (Uint32)indexCount, DRAW_FLAG_VERIFY_STATES };
		da.StartVertexLocation = (Uint32)firstIndex;
		ctx->Draw(da);
	}
	if (mat && mat->liveDrawSet) mat->liveDrawSet = false;   // consumed; the colour pass re-pushes
}

// Draw [first, first+count) of `instBuf` into the prepass. The CB carries current + previous
// camera view*proj only, so instance velocity is camera-only.
void NukeDiligent::renderGBufferInstanced(Mesh* mesh, Material* mat, uint64_t instBuf, int first, int count)
{
	if (!m_impl->gbufActive || !m_impl->gbufPSOInst || count <= 0) return;
	auto bit = m_impl->instBufs.find(instBuf);
	if (bit == m_impl->instBufs.end() || !bit->second.buf) return;
	if (first < 0 || first + count > bit->second.count) return;
	Impl::MeshGPU* gp = m_impl->GetMeshGPU(mesh);
	if (!gp) return;
	++m_impl->statDraws;
	m_impl->statTris += mesh ? mesh->TriCount() * count : 0;
	Impl::MeshGPU& g = *gp;

	float4x4 vp = m_impl->curView * m_impl->curProj;   // prepass is UNjittered
	Impl::TAAState& tst = m_impl->taaStates[m_impl->curCamKey];
	float4x4 prevVP = tst.valid ? (tst.prevView * tst.prevProj) : vp;
	struct CBData { float4x4 wvp; float4x4 world; float4x4 prevWVP; };
	{
		MapHelper<CBData> cb(m_impl->context, m_impl->worldCB, MAP_WRITE, MAP_FLAG_DISCARD);
		if (cb == nullptr) return;
		cb->wvp = vp; cb->world = float4x4::Identity(); cb->prevWVP = prevVP;
	}

	float metallic = 0.0f, roughness = 0.6f; ITextureView* mrsrv = nullptr; ITextureView* nsrv = nullptr;
	ITextureView* texsrv = nullptr; ITextureView* wipesrv = nullptr;
	if (mat) { metallic = mat->metallic; roughness = mat->roughness;
	           if (mat->mr) mrsrv = m_impl->GetTexSRV(mat->mr); if (mat->norm) nsrv = m_impl->GetTexSRV(mat->norm);
	           if (mat->diff) texsrv = m_impl->GetTexSRV(mat->diff);
	           if (mat->wipe) wipesrv = m_impl->GetTexSRV(mat->wipe); }
	{
		MapHelper<Uint8> mb(m_impl->context, m_impl->worldMatCB, MAP_WRITE, MAP_FLAG_DISCARD);
		if (mb == nullptr) return;
		Uint8* p = mb; memset(p, 0, Impl::kMatCBBytes);
		float col[4] = { 1, 1, 1, mat ? (float)mat->color.a : 1.0f };   // g_Color (alpha feeds the cutout clip)
		memcpy(p + 0, col, sizeof(float) * 4);
		float nrmY = nsrv ? ((mat && mat->norm && !mat->norm->invertGreen) ? -1.0f : 1.0f) : 0.0f;
		float prm[4]  = { texsrv ? 1.0f : 0.0f, nrmY, metallic, roughness };
		memcpy(p + 16, prm, sizeof(float) * 4);
		float prm2[4] = { mrsrv ? 1.0f : 0.0f, 0, 0, 1.0f };
		memcpy(p + 32, prm2, sizeof(float) * 4);
		// LiveMaterial: UV transform + cutout/wipe thresholds mirror the color pass.
		float uvt[4]  = { mat ? (float)mat->uvTiling.x : 0.0f, mat ? (float)mat->uvTiling.y : 0.0f,
		                  mat ? (float)mat->uvOffset.x + mat->uvAnim[0] : 0.0f,
		                  mat ? (float)mat->uvOffset.y + mat->uvAnim[1] : 0.0f };
		memcpy(p + 64, uvt, sizeof(float) * 4);
		float uvt2[4] = { mat ? mat->uvRotation * 0.01745329f : 0.0f,
		                  (mat && mat->blendMode == 3) ? mat->alphaCutoff : 0.0f,
		                  (mat && mat->wipe && mat->wipeThreshold > 0.0f) ? mat->wipeThreshold : 0.0f, 0.0f };
		memcpy(p + 80, uvt2, sizeof(float) * 4);
		FillGBufOverlays(p, mat);
	}
	ITextureView* instWhiteSRV = m_impl->whiteTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
	if (m_impl->gbufMRVarInst)
		m_impl->gbufMRVarInst->Set(mrsrv ? mrsrv : instWhiteSRV);
	if (m_impl->gbufNrmVarInst)
		m_impl->gbufNrmVarInst->Set(nsrv ? nsrv : m_impl->flatNormTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
	if (m_impl->gbufTexVarInst)  m_impl->gbufTexVarInst->Set(texsrv ? texsrv : instWhiteSRV);
	if (m_impl->gbufWipeVarInst) m_impl->gbufWipeVarInst->Set(wipesrv ? wipesrv : instWhiteSRV);
	{
		ITextureView* whiteSRV = m_impl->whiteTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
		ITextureView* flatN = m_impl->flatNormTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
		ITextureView* ovsrv[Impl::kOvGbufCount] = {};
		if (mat)
		{
			for (int sl = 0; sl < mat->liveOvCount && sl < nuke::Material::kOverlaySlots; ++sl)
			{
				const nuke::Material::OverlayRT& ov = mat->liveOv[sl];
				if (ov.normal) ovsrv[sl * 3 + 0] = m_impl->GetTexSRV(ov.normal);
				if (ov.mrTex)  ovsrv[sl * 3 + 1] = m_impl->GetTexSRV(ov.mrTex);
				if (ov.mask)   ovsrv[sl * 3 + 2] = m_impl->GetTexSRV(ov.mask);
			}
			if (mat->liveDrawSet && mat->liveDrawMask3D) ovsrv[Impl::kOvSlots * 3] = m_impl->GetTexSRV(mat->liveDrawMask3D);
			if (mat->detailNrm) ovsrv[Impl::kOvSlots * 3 + 1] = m_impl->GetTexSRV(mat->detailNrm);
		}
		for (int k = 0; k < Impl::kOvGbufCount; ++k)
		{
			IDeviceObject* o = ovsrv[k] ? (IDeviceObject*)ovsrv[k]
			                            : (((k < Impl::kOvSlots * 3 && (k % 3) == 0) || k == Impl::kOvSlots * 3 + 1) ? (IDeviceObject*)flatN : (IDeviceObject*)whiteSRV);
			if (m_impl->gbufOvVarInst[k] && o != m_impl->gbufOvLast[1][k]) { m_impl->gbufOvVarInst[k]->Set(o); m_impl->gbufOvLast[1][k] = o; }
		}
	}

	IDeviceContext* ctx = m_impl->context;
	IBuffer* vbs[] = { g.PosBuf(), g.NrmBuf(), g.UVBuf(), bit->second.buf };
	Uint64 offs[] = { g.PosOfs(), g.NrmOfs(), g.UVOfs(), 0 };
	ctx->SetVertexBuffers(0, 4, vbs, offs, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, SET_VERTEX_BUFFERS_FLAG_RESET);
	ctx->SetPipelineState(m_impl->gbufPSOInst);
	ctx->CommitShaderResources(m_impl->gbufSRBInst, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	if (g.IdxBuf())
	{
		// LOD0 range only: the whole IB also carries the appended LOD shells.
		uint32_t l0First = 0, l0Count = 0;
		m_impl->LodRange(mesh, 0, l0First, l0Count);
		ctx->SetIndexBuffer(g.IdxBuf(), g.IdxOfs(), RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		DrawIndexedAttribs da{ (Uint32)l0Count, VT_UINT32, DRAW_FLAG_VERIFY_STATES };
		da.FirstIndexLocation    = (Uint32)l0First;
		da.NumInstances          = (Uint32)count;
		da.FirstInstanceLocation = (Uint32)first;
		ctx->DrawIndexed(da);
	}
	else
	{
		DrawAttribs da{ (Uint32)g.numVerts, DRAW_FLAG_VERIFY_STATES };
		da.NumInstances          = (Uint32)count;
		da.FirstInstanceLocation = (Uint32)first;
		ctx->Draw(da);
	}
	if (mat && mat->liveDrawSet) mat->liveDrawSet = false;   // consumed; the colour pass re-pushes
}

void NukeDiligent::endGBufferPass() { /* gbufActive stays set so endCamera's SSR pass can sample it; beginCamera rebinds the colour target */ }
