#include "NukeDiligentImpl.h"


// G-buffer targets for the SSR prepass: RGBA16F colour (octN.xy, rough, metal) + D32 depth, both 1x (single
// sample) and shader-readable. Resized to the current camera target.
void NukeDiligent::Impl::EnsureGBuffer(int w, int h)
{
	if (w <= 0 || h <= 0) return;
	if (gbufColor && gbufW == w && gbufH == h) return;
	gbufW = w; gbufH = h;
	gbufColor.Release(); gbufDepth.Release();
	gbufRTV = gbufSRV = gbufDSV = gbufDepthSRV = nullptr;

	TextureDesc cd; cd.Name = "GBuffer"; cd.Type = RESOURCE_DIM_TEX_2D; cd.Width = (Uint32)w; cd.Height = (Uint32)h;
	cd.Format = TEX_FORMAT_RGBA16_FLOAT; cd.BindFlags = BIND_RENDER_TARGET | BIND_SHADER_RESOURCE;
	device->CreateTexture(cd, nullptr, &gbufColor);
	if (gbufColor) { gbufRTV = gbufColor->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET); gbufSRV = gbufColor->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE); }

	TextureDesc dd; dd.Name = "GBuffer Depth"; dd.Type = RESOURCE_DIM_TEX_2D; dd.Width = (Uint32)w; dd.Height = (Uint32)h;
	dd.Format = TEX_FORMAT_D32_FLOAT; dd.BindFlags = BIND_DEPTH_STENCIL | BIND_SHADER_RESOURCE;
	device->CreateTexture(dd, nullptr, &gbufDepth);
	if (gbufDepth) { gbufDSV = gbufDepth->GetDefaultView(TEXTURE_VIEW_DEPTH_STENCIL); gbufDepthSRV = gbufDepth->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE); }
}

// G-buffer prepass pipeline: world.vs (shares worldCB) + gbuffer.ps (shares worldMatCB). Single-sample, depth
// write on; outputs the packed surface buffer. Rebuild-safe.
bool NukeDiligent::Impl::BuildGBufferPipe()
{
	gbufPSO.Release(); gbufSRB.Release(); gbufMRVar = nullptr;
	std::string vsSrc = shaderSource("world.vs"), psSrc = shaderSource("gbuffer.ps");
	if (vsSrc.empty() || psSrc.empty()) return false;
	ShaderCreateInfo sci; sci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
	RefCntAutoPtr<IShader> vs, ps;
	sci.Desc = {"GBuffer VS", SHADER_TYPE_VERTEX, true}; sci.Source = vsSrc.c_str(); device->CreateShader(sci, &vs);
	sci.Desc = {"GBuffer PS", SHADER_TYPE_PIXEL, true};  sci.Source = psSrc.c_str(); device->CreateShader(sci, &ps);
	if (!vs || !ps) return false;

	GraphicsPipelineStateCreateInfo ci; ci.PSODesc.Name = "GBuffer PSO";
	auto& gp = ci.GraphicsPipeline;
	gp.NumRenderTargets = 1; gp.RTVFormats[0] = TEX_FORMAT_RGBA16_FLOAT; gp.DSVFormat = TEX_FORMAT_D32_FLOAT;
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
		struct SSRData { float4x4 view, proj, invProj; float res[4]; };
		MapHelper<SSRData> cb(context, ssrCB, MAP_WRITE, MAP_FLAG_DISCARD);
		cb->view = curView; cb->proj = curProj; cb->invProj = curProj.Inverse();
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
	context->SetPipelineState(pp.pso);
	context->CommitShaderResources(pp.srb, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	DrawAttribs da{3, DRAW_FLAG_VERIFY_STATES};
	context->Draw(da);
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
	m_impl->SetCameraViewProj(cam, w, h);
	IDeviceContext* ctx = m_impl->context;
	ctx->SetRenderTargets(1, &m_impl->gbufRTV, m_impl->gbufDSV, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	const float clr[4] = { 0, 0, 0, 0 };
	ctx->ClearRenderTarget(m_impl->gbufRTV, clr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	ctx->ClearDepthStencil(m_impl->gbufDSV, CLEAR_DEPTH_FLAG, 1.f, 0, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	Viewport vp; vp.TopLeftX = 0; vp.TopLeftY = 0; vp.Width = (float)w; vp.Height = (float)h; vp.MinDepth = 0; vp.MaxDepth = 1;
	ctx->SetViewports(1, &vp, w, h);
	m_impl->gbufActive = true;
}

void NukeDiligent::renderGBufferObject(Mesh* mesh, Material* mat, const float pos[3], const float quat[4], const float scale[3])
{
	if (!m_impl->gbufActive || !m_impl->gbufPSO) return;
	Impl::MeshGPU* gp = m_impl->GetMeshGPU(mesh);
	if (!gp) return;
	Impl::MeshGPU& g = *gp;

	float4x4 world = float4x4::Scale(scale[0], scale[1], scale[2])
	               * Diligent::Quaternion<float>(quat[0], quat[1], quat[2], quat[3]).ToMatrix()
	               * float4x4::Translation(pos[0], pos[1], pos[2]);
	float4x4 wvp = world * m_impl->curView * m_impl->curProj;
	struct CBData { float4x4 wvp; float4x4 world; };
	{ MapHelper<CBData> cb(m_impl->context, m_impl->worldCB, MAP_WRITE, MAP_FLAG_DISCARD); cb->wvp = wvp; cb->world = world; }

	float metallic = 0.0f, roughness = 0.6f; ITextureView* mrsrv = nullptr; ITextureView* nsrv = nullptr;
	if (mat) { metallic = mat->metallic; roughness = mat->roughness;
	           if (mat->mr) mrsrv = m_impl->GetTexSRV(mat->mr); if (mat->norm) nsrv = m_impl->GetTexSRV(mat->norm); }
	{
		MapHelper<Uint8> mb(m_impl->context, m_impl->worldMatCB, MAP_WRITE, MAP_FLAG_DISCARD);
		Uint8* p = mb; memset(p, 0, Impl::kMatCBBytes);
		float prm[4]  = { 0, nsrv ? 1.0f : 0.0f, metallic, roughness };   // g_Params (_, hasNormal.y, metallic.z, roughness.w)
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
