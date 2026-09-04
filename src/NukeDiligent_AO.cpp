#include "NukeDiligentImpl.h"

using namespace Diligent;
using namespace std;

// Screen-space ambient occlusion: SSAO / HBAO / GTAO / VBAO (one pass, method per world) over
// the prepass depth + normals at half resolution, then a depth/normal-aware upsample with
// temporal accumulation into a full-res
// visibility texture the world PS multiplies into its ambient/IBL term. Per-target state
// (history) like TAA. Both PSOs build on the builder thread; AO stays off until they land.
namespace {
struct AOData { float4x4 view, proj, invProj, invView; float res[4]; float params[4]; float params2[4]; };
}

bool NukeDiligent::Impl::BuildAOPipes()
{
	const string vs = shaderSource("post.vs"), psA = shaderSource("ao.ps"), psR = shaderSource("aoresolve.ps");
	if (vs.empty() || psA.empty() || psR.empty()) return false;
	SamplerDesc lin; lin.MinFilter = FILTER_TYPE_LINEAR; lin.MagFilter = FILTER_TYPE_LINEAR; lin.MipFilter = FILTER_TYPE_LINEAR;
	lin.AddressU = TEXTURE_ADDRESS_CLAMP; lin.AddressV = TEXTURE_ADDRESS_CLAMP;
	SamplerDesc pt = lin; pt.MinFilter = FILTER_TYPE_POINT; pt.MagFilter = FILTER_TYPE_POINT; pt.MipFilter = FILTER_TYPE_POINT;
	// (texture name, linear?) — depth/normals/velocity must stay point-sampled.
	// RT-AO traces inline (RayQuery): DXC at SM6.5 + RT_ENABLED like the world PS; the other
	// methods compile the same way on those devices, FXC/SM5 elsewhere (method 5 -> GTAO).
	ShaderMacro rtMacro[] = {{"RT_ENABLED", "1"}};
	auto build = [&](const char* name, const string& ps, const vector<pair<const char*, bool>>& texs, bool tlas, int nrt, PostPipe& out) -> bool
	{
		ShaderCreateInfo sci; sci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
		if (rtSupported)
		{
			sci.ShaderCompiler = SHADER_COMPILER_DXC;
			sci.HLSLVersion    = ShaderVersion{6, 5};
			sci.Macros         = ShaderMacroArray{rtMacro, 1};
		}
		RefCntAutoPtr<IShader> v, p;
		sci.Desc = {"AO VS", SHADER_TYPE_VERTEX, true}; sci.Source = vs.c_str(); CreateShaderCached(sci, &v);
		sci.Desc = {name, SHADER_TYPE_PIXEL, true};    sci.Source = ps.c_str(); CreateShaderCached(sci, &p);
		if (!v || !p) return false;
		GraphicsPipelineStateCreateInfo ci; ci.PSODesc.Name = name;
		auto& gp = ci.GraphicsPipeline;
		gp.NumRenderTargets = (Uint8)nrt; gp.RTVFormats[0] = HDR_FMT; gp.RTVFormats[1] = HDR_FMT; gp.DSVFormat = TEX_FORMAT_UNKNOWN;
		gp.PrimitiveTopology = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; gp.RasterizerDesc.CullMode = CULL_MODE_NONE;
		gp.DepthStencilDesc.DepthEnable = False; gp.InputLayout.NumElements = 0;
		vector<ShaderResourceVariableDesc> vars; vector<ImmutableSamplerDesc> imms;
		for (const auto& t : texs)
		{
			vars.push_back({SHADER_TYPE_PIXEL, t.first, SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC});
			imms.push_back({SHADER_TYPE_PIXEL, t.first, t.second ? lin : pt});
		}
		if (tlas && rtSupported) vars.push_back({SHADER_TYPE_PIXEL, "g_TLAS", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC});
		ci.PSODesc.ResourceLayout.Variables = vars.data(); ci.PSODesc.ResourceLayout.NumVariables = (Uint32)vars.size();
		ci.PSODesc.ResourceLayout.ImmutableSamplers = imms.data(); ci.PSODesc.ResourceLayout.NumImmutableSamplers = (Uint32)imms.size();
		ci.pVS = v; ci.pPS = p;
		PostPipe pp;
		CreateGraphicsPipelineStateCached(ci, &pp.pso);
		if (!pp.pso) return false;
		if (auto* c = pp.pso->GetStaticVariableByName(SHADER_TYPE_PIXEL, "AOCB")) c->Set(aoCB);
		pp.pso->CreateShaderResourceBinding(&pp.srb, true);
		pp.srcVar   = pp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Source");
		pp.gbufVar  = pp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_GBuffer");
		pp.depthVar = pp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Depth");
		pp.histVar  = pp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_History");
		pp.velVar   = pp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Velocity");
		out = std::move(pp);
		return true;
	};
	PostPipe a, r;
	const bool ok = build("AO PS", psA, {{"g_GBuffer", false}, {"g_Depth", false}}, true, 1, a)
	             && build("AO resolve PS", psR, {{"g_Source", true}, {"g_Depth", false}, {"g_GBuffer", false}, {"g_Velocity", false}, {"g_History", true}}, false, 2, r);
	if (!ok) { cout << "[NukeDiligent]\tAO pipelines failed to build; ambient occlusion stays off" << endl; return false; }
	aoPipe = std::move(a); aoResolvePipe = std::move(r);
	aoTlasVar = aoPipe.srb ? aoPipe.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_TLAS") : nullptr;
	return true;
}

// The current camera's AO from its prepass, called by beginCamera before the colour targets
// bind. Leaves screenAOSRV = this camera's resolved visibility (null while off / building).
void NukeDiligent::Impl::RunAO(int w, int h)
{
	if (w <= 0 || h <= 0 || !gbufDepthSRV || !gbufSRV) return;
	if (!aoCB)
	{
		BufferDesc d; d.Name = "AOCB"; d.Size = sizeof(AOData); d.Usage = USAGE_DYNAMIC;
		d.BindFlags = BIND_UNIFORM_BUFFER; d.CPUAccessFlags = CPU_ACCESS_WRITE;
		device->CreateBuffer(d, nullptr, &aoCB);
		if (!aoCB) return;
	}
	if (aoBuilding || aoFailed) return;
	if (!aoPipe.pso || !aoResolvePipe.pso)
	{
		if (!aoBuilding.exchange(true))
			EnqueueBuild([this] { if (!BuildAOPipes()) aoFailed = true; }, [this] { aoBuilding = false; }, kPrioGBuffer, "AO pipelines");
		return;
	}

	AOState& st = aoStates[curCamKey];
	st.lastUsed = frameId;
	const int lw = (w + 1) / 2, lh = (h + 1) / 2;
	auto make = [&](RefCntAutoPtr<ITexture>& t, const char* name, int tw, int th)
	{
		Trash(t); t.Release();   // the old one may still be bound by this frame's earlier camera
		TextureDesc td; td.Name = name; td.Type = RESOURCE_DIM_TEX_2D; td.Width = (Uint32)tw; td.Height = (Uint32)th;
		td.MipLevels = 1; td.Format = HDR_FMT; td.BindFlags = BIND_SHADER_RESOURCE | BIND_RENDER_TARGET; td.Usage = USAGE_DEFAULT;
		device->CreateTexture(td, nullptr, &t);
	};
	if (!st.raw || !st.den || st.lw != lw || st.lh != lh)
	{
		make(st.raw, "AO raw", lw, lh); make(st.den, "AO denoised", lw, lh); make(st.den2, "AO denoise scratch", lw, lh);   // den2: the 2-target resolve PSO needs a same-size second RT
		st.lw = lw; st.lh = lh;
	}
	if (!st.resolved || !st.hist[0] || !st.hist[1] || st.w != w || st.h != h)
	{
		make(st.resolved, "AO resolved", w, h); make(st.hist[0], "AO history A", w, h); make(st.hist[1], "AO history B", w, h);
		st.w = w; st.h = h; st.valid = false; st.cur = 0;
	}
	if (!st.raw || !st.den || !st.den2 || !st.resolved || !st.hist[0] || !st.hist[1]) return;

	// Off, SSAO (slices*steps hemisphere samples), HBAO (directions x steps), GTAO, VBAO (slices x
	// steps per side), RT-AO (rays per pixel; GTAO on devices without ray tracing)
	static const int kSlices[6] = { 0, 4, 8, 4, 4, 4 }, kSteps[6] = { 0, 4, 8, 10, 10, 1 };
	int q = aoQuality < 1 ? 1 : (aoQuality > 5 ? 5 : aoQuality);
	if (q == 5 && !rtSupported) q = 3;
	const float4x4 invProj = gbufProj.Inverse(), invView = gbufView.Inverse();
	// params.w = slices for the AO pass, the mode (0 denoise / 1 final) for the resolve pass
	auto fill = [&](int tw, int th, float pw, float p0, float p1, float p2, float p3)
	{
		MapHelper<AOData> cb(context, aoCB, MAP_WRITE, MAP_FLAG_DISCARD);
		cb->view = gbufView; cb->proj = gbufProj; cb->invProj = invProj; cb->invView = invView;
		cb->res[0] = (float)tw; cb->res[1] = (float)th; cb->res[2] = 1.0f / tw; cb->res[3] = 1.0f / th;
		cb->params[0] = aoRadius; cb->params[1] = aoIntensity; cb->params[2] = aoPower; cb->params[3] = pw;
		cb->params2[0] = p0; cb->params2[1] = p1; cb->params2[2] = p2; cb->params2[3] = p3;
	};
	auto draw = [&](PostPipe& pp, ITexture* dst, ITexture* dst2, int tw, int th)
	{
		ITextureView* rtv[2] = { dst->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET), dst2 ? dst2->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET) : nullptr };
		context->SetRenderTargets(dst2 ? 2 : 1, rtv, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		Viewport vp; vp.TopLeftX = 0; vp.TopLeftY = 0; vp.Width = (float)tw; vp.Height = (float)th; vp.MinDepth = 0; vp.MaxDepth = 1;
		context->SetViewports(1, &vp, tw, th);
		context->SetPipelineState(pp.pso);
		context->CommitShaderResources(pp.srb, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		DrawAttribs da{3, DRAW_FLAG_VERIFY_STATES};
		context->Draw(da);
	};

	// 1) raw horizons at the AO resolution
	fill(lw, lh, (float)kSlices[q], (float)kSteps[q], (float)(aoFrame % 16), (float)q, 0.0f);
	if (aoPipe.gbufVar)  aoPipe.gbufVar->Set(gbufSRV);
	if (aoPipe.depthVar) aoPipe.depthVar->Set(gbufDepthSRV);
	if (aoTlasVar) aoTlasVar->Set((rtSceneReady && tlas) ? (IDeviceObject*)tlas.RawPtr() : (IDeviceObject*)fallbackTLAS.RawPtr());
	draw(aoPipe, st.raw, nullptr, lw, lh);

	// 2) 5x5 bilateral denoise, still at the AO resolution
	ITextureView* whiteSRV = whiteTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
	fill(lw, lh, 0.0f, 0.0f, 0.0f, (float)lw, (float)lh);
	if (aoResolvePipe.srcVar)   aoResolvePipe.srcVar->Set(st.raw->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
	if (aoResolvePipe.depthVar) aoResolvePipe.depthVar->Set(gbufDepthSRV);
	if (aoResolvePipe.gbufVar)  aoResolvePipe.gbufVar->Set(gbufSRV);
	if (aoResolvePipe.velVar)   aoResolvePipe.velVar->Set(gbufVelSRV ? gbufVelSRV : whiteSRV);
	ITexture* histPrev = st.hist[st.cur ^ 1]; ITexture* histCur = st.hist[st.cur];
	if (aoResolvePipe.histVar)  aoResolvePipe.histVar->Set(histPrev->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
	draw(aoResolvePipe, st.den, st.den2, lw, lh);
	// 3) bilateral upsample + temporal accumulation at full resolution
	const bool useHist = st.valid && gbufVelSRV;
	fill(w, h, 1.0f, useHist ? 1.0f : 0.0f, 0.9f, (float)lw, (float)lh);
	if (aoResolvePipe.srcVar)   aoResolvePipe.srcVar->Set(st.den->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
	draw(aoResolvePipe, st.resolved, histCur, w, h);
	context->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE);
	st.cur ^= 1;
	st.valid = true;
	++aoFrame;
	// The prepass derives its motion vectors from the TAA state's previous camera; without TAA
	// nobody advances it and the velocity stays zero. Publish this frame's (unjittered) camera
	// the way RunTAA does, so next frame's reprojection follows camera motion.
	{ TAAState& ts = taaStates[curCamKey]; ts.prevView = gbufView; ts.prevProj = gbufProj; ts.valid = true; ts.lastUsed = frameId; }
	screenAOSRV = st.resolved->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
}

// Per-camera temporal state outlives nothing: a camera deleted, a preview closed or a render
// texture dropped simply stops rendering, and its history goes after a couple of seconds.
void NukeDiligent::Impl::PruneCameraStates()
{
	const uint64_t kStale = 120;
	for (auto it = taaStates.begin(); it != taaStates.end();)
	{
		if (frameId - it->second.lastUsed > kStale) { Trash(it->second.hist); it = taaStates.erase(it); }
		else ++it;
	}
	for (auto it = aoStates.begin(); it != aoStates.end();)
	{
		if (frameId - it->second.lastUsed > kStale)
		{
			AOState& s = it->second;
			Trash(s.raw); Trash(s.den); Trash(s.den2); Trash(s.resolved); Trash(s.hist[0]); Trash(s.hist[1]);
			it = aoStates.erase(it);
		}
		else ++it;
	}
	for (auto it = ssgiStates.begin(); it != ssgiStates.end();)
	{
		if (frameId - it->second.lastUsed > kStale)
		{
			SSGIState& s = it->second;
			Trash(s.lit); Trash(s.raw); Trash(s.den); Trash(s.den2); Trash(s.den3); Trash(s.resolved); Trash(s.hist[0]); Trash(s.hist[1]); Trash(s.histZ[0]); Trash(s.histZ[1]);
			it = ssgiStates.erase(it);
		}
		else ++it;
	}
	for (auto it = occlViews.begin(); it != occlViews.end();)
	{
		if (occlFrame - it->second.lastUsed > kStale)
		{
			OcclView& o = it->second;
			Trash(o.hiz); Trash(o.hizScratch); for (auto& r : o.ring) Trash(r.staging);
			it = occlViews.erase(it);
		}
		else ++it;
	}
}
