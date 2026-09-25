// Built-in post effects of the R3 tails: depth of field, motion blur, auto-exposure. Each is a
// chain stage declared by its *.post.hlsl (the params) and driven here as a multi-pass over the
// HDR chain; the pipelines are built with the post pipelines (warm-up), never in the draw path.
#include "NukeDiligentImpl.h"
#include <API/Model/Time.h>

// ---- pipelines --------------------------------------------------------------------------------

void NukeDiligent::Impl::CreatePostFXPipelines()
{
	ReloadScope reloadScope("postfx");
	dofCocPSO.Release(); dofGatherPSO.Release(); dofCompPSO.Release();
	mbTilePSO.Release(); mbNeighborPSO.Release(); mbReconPSO.Release();
	expLumPSO.Release(); expAdaptPSO.Release(); expApplyPSO.Release(); upReactPSO.Release();
	dofCocSRB.Release(); dofGatherSRB.Release(); dofCompSRB.Release();   // (a shader reload reruns this)
	mbTileSRB.Release(); mbNeighborSRB.Release(); mbReconSRB.Release();
	expLumSRB.Release(); expAdaptSRB.Release(); expApplySRB.Release(); upReactSRB.Release();
	auto makeCB = [&](const char* name, size_t bytes, RefCntAutoPtr<IBuffer>& cb)
	{
		if (cb) return;
		BufferDesc d; d.Name = name; d.Size = (Uint64)bytes; d.Usage = USAGE_DYNAMIC; d.BindFlags = BIND_UNIFORM_BUFFER; d.CPUAccessFlags = CPU_ACCESS_WRITE;
		device->CreateBuffer(d, nullptr, &cb);
	};
	makeCB("DofCB", sizeof(float) * 12, dofCB);
	makeCB("MbCB",  sizeof(float) * 8,  mbCB);
	makeCB("ExpCB", sizeof(float) * 12, expCB);
	std::string vs = shaderSource("post.vs");
	if (vs.empty() || !dofCB || !mbCB || !expCB) return;

	SamplerDesc lin; lin.MinFilter = FILTER_TYPE_LINEAR; lin.MagFilter = FILTER_TYPE_LINEAR; lin.MipFilter = FILTER_TYPE_LINEAR;
	lin.AddressU = TEXTURE_ADDRESS_CLAMP; lin.AddressV = TEXTURE_ADDRESS_CLAMP; lin.AddressW = TEXTURE_ADDRESS_CLAMP;
	SamplerDesc pnt = lin; pnt.MinFilter = FILTER_TYPE_POINT; pnt.MagFilter = FILTER_TYPE_POINT; pnt.MipFilter = FILTER_TYPE_POINT;

	// A fullscreen graphics pass: `sampled` names get the linear sampler, `loaded` are Load-only.
	struct Tex { const char* name; bool sampled; };
	auto makePS = [&](const char* psName, const char* dbg, TEXTURE_FORMAT fmt, const char* cbName, IBuffer* cb,
	                  std::initializer_list<Tex> texs, RefCntAutoPtr<IPipelineState>& pso, RefCntAutoPtr<IShaderResourceBinding>& srb)
	{
		std::string ps = shaderSource(psName);
		if (ps.empty()) { std::cout << "[NukeDiligent]\tpost fx shader missing: " << psName << std::endl; return; }
		ShaderCreateInfo s; s.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL; s.pShaderSourceStreamFactory = ShaderFactory();
		RefCntAutoPtr<IShader> vv, pp;
		s.Desc = {dbg, SHADER_TYPE_VERTEX, true}; s.Source = vs.c_str(); CreateShaderCached(s, &vv);
		s.Desc = {dbg, SHADER_TYPE_PIXEL,  true}; s.Source = ps.c_str(); CreateShaderCached(s, &pp);
		if (!vv || !pp) return;
		GraphicsPipelineStateCreateInfo ci; ci.PSODesc.Name = dbg;
		auto& gp = ci.GraphicsPipeline;
		gp.NumRenderTargets = 1; gp.RTVFormats[0] = fmt; gp.DSVFormat = TEX_FORMAT_UNKNOWN;
		gp.PrimitiveTopology = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; gp.RasterizerDesc.CullMode = CULL_MODE_NONE;
		gp.DepthStencilDesc.DepthEnable = False; gp.InputLayout.NumElements = 0;
		std::vector<ShaderResourceVariableDesc> vars; std::vector<ImmutableSamplerDesc> imms;
		for (const Tex& t : texs)
		{
			vars.push_back({SHADER_TYPE_PIXEL, t.name, SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC});
			if (t.sampled) imms.push_back({SHADER_TYPE_PIXEL, t.name, lin});
		}
		ci.PSODesc.ResourceLayout.Variables = vars.data(); ci.PSODesc.ResourceLayout.NumVariables = (Uint32)vars.size();
		ci.PSODesc.ResourceLayout.ImmutableSamplers = imms.data(); ci.PSODesc.ResourceLayout.NumImmutableSamplers = (Uint32)imms.size();
		ci.pVS = vv; ci.pPS = pp;
		CreateGraphicsPipelineStateCached(ci, &pso);
		if (!pso) { std::cout << "[NukeDiligent]\tpost fx PSO failed: " << dbg << std::endl; return; }
		if (auto* c = pso->GetStaticVariableByName(SHADER_TYPE_PIXEL, cbName)) c->Set(cb);
		pso->CreateShaderResourceBinding(&srb, true);
	};
	const TEXTURE_FORMAT RG16 = TEX_FORMAT_RG16_FLOAT, R16 = TEX_FORMAT_R16_FLOAT;
	makePS("dof_coc.ps",     "DOF CoC",     HDR_FMT, "DofCB", dofCB, {{"g_Source", true}, {"g_Depth", false}}, dofCocPSO, dofCocSRB);
	makePS("upscale_reactive.ps", "Upscale Reactive", TEX_FORMAT_R8_UNORM, "DofCB", dofCB, {{"g_GBuffer", false}, {"g_Cover", false}, {"g_CoverWorld", false}}, upReactPSO, upReactSRB);   // 4.2: the reactive mask
	makePS("dof_gather.ps",  "DOF Gather",  HDR_FMT, "DofCB", dofCB, {{"g_Source", true}}, dofGatherPSO, dofGatherSRB);
	makePS("dof_comp.ps",    "DOF Comp",    HDR_FMT, "DofCB", dofCB, {{"g_Source", true}, {"g_Depth", false}, {"g_Far", true}, {"g_Near", true}}, dofCompPSO, dofCompSRB);
	makePS("mb_tilemax.ps",  "MB Tile Max", RG16,    "MbCB",  mbCB,  {{"g_Velocity", false}}, mbTilePSO, mbTileSRB);
	makePS("mb_neighbor.ps", "MB Neighbor", RG16,    "MbCB",  mbCB,  {{"g_Source", false}}, mbNeighborPSO, mbNeighborSRB);
	makePS("mb_recon.ps",    "MB Recon",    HDR_FMT, "MbCB",  mbCB,  {{"g_Source", true}, {"g_Velocity", false}, {"g_Depth", false}, {"g_Tiles", true}}, mbReconPSO, mbReconSRB);
	makePS("exposure_lum.ps",   "Exposure Lum",   R16,     "ExpCB", expCB, {{"g_Source", true}}, expLumPSO, expLumSRB);
	makePS("exposure_apply.ps", "Exposure Apply", HDR_FMT, "ExpCB", expCB, {{"g_Source", true}, {"g_Adapted", false}}, expApplyPSO, expApplySRB);
	{   // the adaptation compute: histogram + eye easing, one group
		std::string cs = shaderSource("exposure_adapt.cs");
		if (cs.empty()) { std::cout << "[NukeDiligent]\tpost fx shader missing: exposure_adapt.cs" << std::endl; return; }
		ShaderCreateInfo s; s.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL; s.pShaderSourceStreamFactory = ShaderFactory();
		RefCntAutoPtr<IShader> c;
		s.Desc = {"Exposure Adapt CS", SHADER_TYPE_COMPUTE, true}; s.Source = cs.c_str(); CreateShaderCached(s, &c);
		if (!c) return;
		ComputePipelineStateCreateInfo ci; ci.PSODesc.Name = "Exposure Adapt PSO";
		ShaderResourceVariableDesc vars[] = {
			{SHADER_TYPE_COMPUTE, "ExpCB",  SHADER_RESOURCE_VARIABLE_TYPE_STATIC},
			{SHADER_TYPE_COMPUTE, "g_Lum",  SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
			{SHADER_TYPE_COMPUTE, "g_Prev", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
			{SHADER_TYPE_COMPUTE, "g_Out",  SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
		};
		ci.PSODesc.ResourceLayout.Variables = vars; ci.PSODesc.ResourceLayout.NumVariables = 4;
		ci.pCS = c;
		CreateComputePipelineStateCached(ci, &expAdaptPSO);
		if (!expAdaptPSO) { std::cout << "[NukeDiligent]\tpost fx PSO failed: Exposure Adapt" << std::endl; return; }
		if (auto* v = expAdaptPSO->GetStaticVariableByName(SHADER_TYPE_COMPUTE, "ExpCB")) v->Set(expCB);
		expAdaptPSO->CreateShaderResourceBinding(&expAdaptSRB, true);
	}
}

// ---- shared helpers ---------------------------------------------------------------------------

static void FullscreenPass(IDeviceContext* ctx, IPipelineState* pso, IShaderResourceBinding* srb, ITextureView* rtv, int w, int h)
{
	ctx->SetRenderTargets(1, &rtv, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	Viewport vp; vp.TopLeftX = 0; vp.TopLeftY = 0; vp.Width = (float)w; vp.Height = (float)h; vp.MinDepth = 0; vp.MaxDepth = 1;
	ctx->SetViewports(1, &vp, w, h);
	ctx->SetPipelineState(pso);
	ctx->CommitShaderResources(srb, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	DrawAttribs da{3, DRAW_FLAG_VERIFY_STATES}; ctx->Draw(da);
}
static void SetVar(IShaderResourceBinding* srb, SHADER_TYPE st, const char* name, IDeviceObject* o)
{
	if (auto* v = srb->GetVariableByName(st, name)) if (o) v->Set(o);
}
static float ParamAt(const std::vector<float>& p, size_t i, float def) { return i < p.size() ? p[i] : def; }

// A size-keyed pair of textures from a cache (the bloom pattern): a/b re-created on a size change.
ITexture* NukeDiligent::Impl::SizedTex(std::unordered_map<uint64_t, SizedTexSet>& cache, int w, int h, TEXTURE_FORMAT fmt, const char* name, bool second, Diligent::BIND_FLAGS bind)
{
	const uint64_t key = ((uint64_t)(uint32_t)w << 32) | (uint32_t)h;
	SizedTexSet& s = cache[key];
	RefCntAutoPtr<ITexture>& t = second ? s.b : s.a;
	if (t && t->GetDesc().Format != fmt) { Trash(t); t.Release(); }
	if (!t)
	{
		TextureDesc td; td.Name = name; td.Type = RESOURCE_DIM_TEX_2D; td.Width = (Uint32)std::max(w, 1); td.Height = (Uint32)std::max(h, 1);
		td.Format = fmt; td.BindFlags = bind;
		device->CreateTexture(td, nullptr, &t);
	}
	s.lastUsed = ++sizedClock;
	EvictSized(cache, key);
	return t;
}

// ---- depth of field ---------------------------------------------------------------------------

void NukeDiligent::Impl::RunDOF(ITextureView* srcSRV, ITextureView* dstRTV, int w, int h, const std::vector<float>& params)
{
	if (!dofCocPSO || !dofGatherPSO || !dofCompPSO || !srcSRV || !dstRTV || !gbufDepthSRV) { if (srcSRV && dstRTV) BlitTexture(srcSRV, dstRTV->GetTexture()); return; }
	const float focus = ParamAt(params, 0, 8.0f), range = ParamAt(params, 1, 4.0f), maxCoC = ParamAt(params, 2, 12.0f), nearK = ParamAt(params, 3, 1.0f);
	if (maxCoC <= 0.01f) { BlitTexture(srcSRV, dstRTV->GetTexture()); return; }
	const int hw = std::max(w / 2, 1), hh = std::max(h / 2, 1);
	ITexture* cocT  = SizedTex(dofCache, hw, hh, HDR_FMT, "DOF CoC",  false);
	ITexture* farT  = SizedTex(dofCache, hw, hh, HDR_FMT, "DOF Far",  true);
	ITexture* nearT = SizedTex(dofNearCache, hw, hh, HDR_FMT, "DOF Near", false);
	if (!cocT || !farT || !nearT) return;
	auto fill = [&](float field)
	{
		MapHelper<float> cb(context, dofCB, MAP_WRITE, MAP_FLAG_DISCARD);
		if (cb == nullptr) return;
		cb[0] = curNear; cb[1] = curFar; cb[2] = focus; cb[3] = range;
		cb[4] = maxCoC; cb[5] = nearK; cb[6] = (float)w; cb[7] = (float)h;
		cb[8] = 1.0f / hw; cb[9] = 1.0f / hh; cb[10] = field; cb[11] = 0.0f;
	};
	fill(0.0f);
	SetVar(dofCocSRB, SHADER_TYPE_PIXEL, "g_Source", srcSRV);
	SetVar(dofCocSRB, SHADER_TYPE_PIXEL, "g_Depth", gbufDepthSRV);
	FullscreenPass(context, dofCocPSO, dofCocSRB, cocT->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET), hw, hh);
	SetVar(dofGatherSRB, SHADER_TYPE_PIXEL, "g_Source", cocT->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
	FullscreenPass(context, dofGatherPSO, dofGatherSRB, farT->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET), hw, hh);
	fill(1.0f);
	FullscreenPass(context, dofGatherPSO, dofGatherSRB, nearT->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET), hw, hh);
	SetVar(dofCompSRB, SHADER_TYPE_PIXEL, "g_Source", srcSRV);
	SetVar(dofCompSRB, SHADER_TYPE_PIXEL, "g_Depth", gbufDepthSRV);
	SetVar(dofCompSRB, SHADER_TYPE_PIXEL, "g_Far",  farT->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
	SetVar(dofCompSRB, SHADER_TYPE_PIXEL, "g_Near", nearT->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
	FullscreenPass(context, dofCompPSO, dofCompSRB, dstRTV, w, h);
}

// ---- motion blur ------------------------------------------------------------------------------

void NukeDiligent::Impl::RunMotionBlur(ITextureView* srcSRV, ITextureView* dstRTV, int w, int h, const std::vector<float>& params)
{
	if (!mbTilePSO || !mbNeighborPSO || !mbReconPSO || !srcSRV || !dstRTV || !gbufDepthSRV || !gbufVelSRV) { if (srcSRV && dstRTV) BlitTexture(srcSRV, dstRTV->GetTexture()); return; }
	const float shutter = ParamAt(params, 0, 0.5f), maxBlur = ParamAt(params, 1, 32.0f), samples = ParamAt(params, 2, 12.0f);
	// The velocity target is written against this camera's previous view/projection: keep them
	// (TAA does the same); without TAA the prepass would otherwise see no camera motion.
	TAAState& st = taaStates[curCamKey];
	st.prevView = curView; st.prevProj = curProjNoJitter; st.valid = true; st.lastUsed = frameId;
	if (shutter <= 0.001f || maxBlur < 0.5f) { BlitTexture(srcSRV, dstRTV->GetTexture()); return; }
	const int tile = 20, tw = (w + tile - 1) / tile, th = (h + tile - 1) / tile;
	ITexture* tileT = SizedTex(mbCache, tw, th, TEX_FORMAT_RG16_FLOAT, "MB Tiles", false);
	ITexture* nbrT  = SizedTex(mbCache, tw, th, TEX_FORMAT_RG16_FLOAT, "MB Neighbours", true);
	if (!tileT || !nbrT) return;
	{
		MapHelper<float> cb(context, mbCB, MAP_WRITE, MAP_FLAG_DISCARD);
		if (cb == nullptr) return;
		cb[0] = (float)w; cb[1] = (float)h; cb[2] = (float)tile; cb[3] = maxBlur;
		cb[4] = shutter; cb[5] = samples; cb[6] = 1.0f / tw; cb[7] = 1.0f / th;
	}
	SetVar(mbTileSRB, SHADER_TYPE_PIXEL, "g_Velocity", gbufVelSRV);
	FullscreenPass(context, mbTilePSO, mbTileSRB, tileT->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET), tw, th);
	SetVar(mbNeighborSRB, SHADER_TYPE_PIXEL, "g_Source", tileT->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
	FullscreenPass(context, mbNeighborPSO, mbNeighborSRB, nbrT->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET), tw, th);
	SetVar(mbReconSRB, SHADER_TYPE_PIXEL, "g_Source", srcSRV);
	SetVar(mbReconSRB, SHADER_TYPE_PIXEL, "g_Velocity", gbufVelSRV);
	SetVar(mbReconSRB, SHADER_TYPE_PIXEL, "g_Depth", gbufDepthSRV);
	SetVar(mbReconSRB, SHADER_TYPE_PIXEL, "g_Tiles", nbrT->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
	FullscreenPass(context, mbReconPSO, mbReconSRB, dstRTV, w, h);
}

// ---- auto-exposure ----------------------------------------------------------------------------

void NukeDiligent::Impl::RunExposure(ITextureView* srcSRV, ITextureView* dstRTV, int w, int h, const std::vector<float>& params)
{
	if (!expLumPSO || !expAdaptPSO || !expApplyPSO || !srcSRV || !dstRTV) { if (srcSRV && dstRTV) BlitTexture(srcSRV, dstRTV->GetTexture()); return; }
	ExposureState& st = expStates[curCamKey];
	st.lastUsed = frameId;
	for (int i = 0; i < 2; ++i)
		if (!st.ev[i])
		{
			TextureDesc td; td.Name = "Exposure EV"; td.Type = RESOURCE_DIM_TEX_2D; td.Width = 1; td.Height = 1;
			td.Format = TEX_FORMAT_R32_FLOAT; td.BindFlags = BIND_SHADER_RESOURCE | BIND_UNORDERED_ACCESS;
			device->CreateTexture(td, nullptr, &st.ev[i]);
			st.valid = false;
		}
	ITexture* lumT = SizedTex(expCache, 64, 64, TEX_FORMAT_R16_FLOAT, "Exposure Lum", false);
	if (!st.ev[0] || !st.ev[1] || !lumT) return;
	// The frame's dt for the easing: the game clock's delta, once per frame (a second camera in
	// the same frame eases by zero, not twice).
	const double now = Time::getSingleton() ? Time::getSingleton()->gameDelta : 0.0;
	const float dt = (st.frame == frameId) ? 0.0f : (float)std::min(std::max(now, 0.0), 0.25);
	st.frame = frameId;
	{
		MapHelper<float> cb(context, expCB, MAP_WRITE, MAP_FLAG_DISCARD);
		if (cb == nullptr) return;
		cb[0] = ParamAt(params, 0, -4.0f); cb[1] = ParamAt(params, 1, 10.0f); cb[2] = ParamAt(params, 2, 3.0f); cb[3] = ParamAt(params, 3, 1.0f);
		cb[4] = ParamAt(params, 4, 0.0f); cb[5] = ParamAt(params, 5, 0.0f); cb[6] = ParamAt(params, 6, 0.0f); cb[7] = dt;
		cb[8] = 1.0f / w; cb[9] = 1.0f / h; cb[10] = hdr ? 0.0f : 1.0f; cb[11] = st.valid ? 1.0f : 0.0f;
	}
	// 1) the 64x64 log-luminance grid
	SetVar(expLumSRB, SHADER_TYPE_PIXEL, "g_Source", srcSRV);
	FullscreenPass(context, expLumPSO, expLumSRB, lumT->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET), 64, 64);
	// 2) histogram + adaptation: prev EV -> next EV (targets must be unbound around compute)
	const int nxt = st.cur ^ 1;
	context->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	SetVar(expAdaptSRB, SHADER_TYPE_COMPUTE, "g_Lum",  lumT->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
	SetVar(expAdaptSRB, SHADER_TYPE_COMPUTE, "g_Prev", st.ev[st.cur]->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
	SetVar(expAdaptSRB, SHADER_TYPE_COMPUTE, "g_Out",  st.ev[nxt]->GetDefaultView(TEXTURE_VIEW_UNORDERED_ACCESS));
	context->SetPipelineState(expAdaptPSO);
	context->CommitShaderResources(expAdaptSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	context->DispatchCompute(DispatchComputeAttribs(1, 1, 1));
	st.cur = nxt; st.valid = true;
	// 3) apply
	SetVar(expApplySRB, SHADER_TYPE_PIXEL, "g_Source", srcSRV);
	SetVar(expApplySRB, SHADER_TYPE_PIXEL, "g_Adapted", st.ev[st.cur]->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
	FullscreenPass(context, expApplyPSO, expApplySRB, dstRTV, w, h);
}

// Per-camera exposure states nobody rendered with lately (the TAA/AO states' rule).
void NukeDiligent::Impl::PruneExposureStates()
{
	for (auto it = expStates.begin(); it != expStates.end();)
	{
		if (it->second.lastUsed + 120 < frameId) { Trash(it->second.ev[0]); Trash(it->second.ev[1]); it = expStates.erase(it); }
		else ++it;
	}
}
