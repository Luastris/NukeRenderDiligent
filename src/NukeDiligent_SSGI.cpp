#include "NukeDiligentImpl.h"
#include <algorithm>

using namespace Diligent;
using namespace std;

// Screen-space GI: contact-scale diffuse bounce on top of the DDGI probes. Per camera: last
// frame's scene colour is kept as the lit history; ssgi.ps marches the prepass depth at half
// resolution and reads the bounce from that history; ssgi_resolve.ps denoises, upsamples and
// accumulates it (depth-rejected history, second render target). The world PS adds the result
// (g_ScreenGI, E / pi) to its ambient term. Same pipe/state pattern as the AO pass.
namespace {
struct SSGIData { float4x4 view, proj, invProj; float res[4]; float params[4]; float params2[4]; };
}

void NukeDiligent::setScreenGI(int quality, float radius, float intensity)
{
	m_impl->ssgiQuality = quality; m_impl->ssgiRadius = radius; m_impl->ssgiIntensity = intensity;
}

bool NukeDiligent::Impl::BuildSSGIPipes()
{
	const string vs = shaderSource("post.vs"), psT = shaderSource("ssgi.ps"), psR = shaderSource("ssgi_resolve.ps");
	if (vs.empty() || psT.empty() || psR.empty()) return false;
	SamplerDesc lin; lin.MinFilter = FILTER_TYPE_LINEAR; lin.MagFilter = FILTER_TYPE_LINEAR; lin.MipFilter = FILTER_TYPE_LINEAR;
	lin.AddressU = TEXTURE_ADDRESS_CLAMP; lin.AddressV = TEXTURE_ADDRESS_CLAMP;
	SamplerDesc pt = lin; pt.MinFilter = FILTER_TYPE_POINT; pt.MagFilter = FILTER_TYPE_POINT; pt.MipFilter = FILTER_TYPE_POINT;
	auto build = [&](const char* name, const string& ps, const vector<pair<const char*, bool>>& texs, int nrt, PostPipe& out) -> bool
	{
		ShaderCreateInfo sci; sci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
		RefCntAutoPtr<IShader> v, p;
		sci.Desc = {"SSGI VS", SHADER_TYPE_VERTEX, true}; sci.Source = vs.c_str(); CreateShaderCached(sci, &v);
		sci.Desc = {name, SHADER_TYPE_PIXEL, true};      sci.Source = ps.c_str(); CreateShaderCached(sci, &p);
		if (!v || !p) return false;
		GraphicsPipelineStateCreateInfo ci; ci.PSODesc.Name = name;
		auto& gp = ci.GraphicsPipeline;
		gp.NumRenderTargets = (Uint8)nrt; gp.RTVFormats[0] = HDR_FMT; gp.RTVFormats[1] = HDR_FMT; gp.RTVFormats[2] = TEX_FORMAT_R16_FLOAT; gp.DSVFormat = TEX_FORMAT_UNKNOWN;
		gp.PrimitiveTopology = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; gp.RasterizerDesc.CullMode = CULL_MODE_NONE;
		gp.DepthStencilDesc.DepthEnable = False; gp.InputLayout.NumElements = 0;
		vector<ShaderResourceVariableDesc> vars; vector<ImmutableSamplerDesc> imms;
		for (const auto& t : texs)
		{
			vars.push_back({SHADER_TYPE_PIXEL, t.first, SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC});
			imms.push_back({SHADER_TYPE_PIXEL, t.first, t.second ? lin : pt});
		}
		ci.PSODesc.ResourceLayout.Variables = vars.data(); ci.PSODesc.ResourceLayout.NumVariables = (Uint32)vars.size();
		ci.PSODesc.ResourceLayout.ImmutableSamplers = imms.data(); ci.PSODesc.ResourceLayout.NumImmutableSamplers = (Uint32)imms.size();
		ci.pVS = v; ci.pPS = p;
		PostPipe pp;
		CreateGraphicsPipelineStateCached(ci, &pp.pso);
		if (!pp.pso) return false;
		if (auto* c = pp.pso->GetStaticVariableByName(SHADER_TYPE_PIXEL, "SSGICB")) c->Set(ssgiCB);
		pp.pso->CreateShaderResourceBinding(&pp.srb, true);
		pp.srcVar   = pp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Source");
		pp.gbufVar  = pp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_GBuffer");
		pp.depthVar = pp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Depth");
		pp.histVar  = pp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_History");
		pp.velVar   = pp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Velocity");
		pp.objIdVar = pp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_HistoryZ");   // history depth (the objId slot is free here)
		out = std::move(pp);
		return true;
	};
	PostPipe t, r;
	const bool ok = build("SSGI trace PS", psT, {{"g_GBuffer", false}, {"g_Depth", false}, {"g_Velocity", false}, {"g_History", true}}, 1, t)
	             && build("SSGI resolve PS", psR, {{"g_Source", true}, {"g_Depth", false}, {"g_GBuffer", false}, {"g_Velocity", false}, {"g_History", true}, {"g_HistoryZ", true}}, 3, r);
	if (!ok) { cout << "[NukeDiligent]\tSSGI pipelines failed to build; screen-space GI stays off" << endl; return false; }
	ssgiPipe = std::move(t); ssgiResolvePipe = std::move(r);
	return true;
}

// This camera's bounce from its prepass + last frame's colour; beginCamera calls it next to RunAO.
void NukeDiligent::Impl::RunSSGI(int w, int h)
{
	if (w <= 0 || h <= 0 || !gbufDepthSRV || !gbufSRV) return;
	if (!ssgiCB)
	{
		BufferDesc d; d.Name = "SSGICB"; d.Size = sizeof(SSGIData); d.Usage = USAGE_DYNAMIC;
		d.BindFlags = BIND_UNIFORM_BUFFER; d.CPUAccessFlags = CPU_ACCESS_WRITE;
		device->CreateBuffer(d, nullptr, &ssgiCB);
		if (!ssgiCB) return;
	}
	if (ssgiBuilding || ssgiFailed) return;
	if (!ssgiPipe.pso || !ssgiResolvePipe.pso)
	{
		if (!ssgiBuilding.exchange(true))
			EnqueueBuild([this] { if (!BuildSSGIPipes()) ssgiFailed = true; }, [this] { ssgiBuilding = false; }, kPrioGBuffer, "SSGI pipelines");
		return;
	}
	SSGIState& st = ssgiStates[curCamKey];
	st.lastUsed = frameId;
	if (!st.lit || !st.litValid) return;   // no lit history yet (first frame of this camera): nothing to bounce
	const int lw = (w + 1) / 2, lh = (h + 1) / 2;
	auto make = [&](RefCntAutoPtr<ITexture>& t, const char* name, int tw, int th, TEXTURE_FORMAT fmt = HDR_FMT)
	{
		Trash(t); t.Release();
		TextureDesc td; td.Name = name; td.Type = RESOURCE_DIM_TEX_2D; td.Width = (Uint32)tw; td.Height = (Uint32)th;
		td.MipLevels = 1; td.Format = fmt; td.BindFlags = BIND_SHADER_RESOURCE | BIND_RENDER_TARGET; td.Usage = USAGE_DEFAULT;
		device->CreateTexture(td, nullptr, &t);
	};
	if (!st.raw || !st.den || !st.den2 || st.lw != lw || st.lh != lh)
	{
		make(st.raw, "SSGI raw", lw, lh); make(st.den, "SSGI denoised", lw, lh); make(st.den2, "SSGI scratch", lw, lh); make(st.den3, "SSGI scratch Z", lw, lh, TEX_FORMAT_R16_FLOAT);
		st.lw = lw; st.lh = lh;
	}
	if (!st.resolved || !st.hist[0] || !st.hist[1] || st.w != w || st.h != h)
	{
		make(st.resolved, "SSGI resolved", w, h); make(st.hist[0], "SSGI history A", w, h); make(st.hist[1], "SSGI history B", w, h);
		make(st.histZ[0], "SSGI history Z A", w, h, TEX_FORMAT_R16_FLOAT); make(st.histZ[1], "SSGI history Z B", w, h, TEX_FORMAT_R16_FLOAT);
		st.w = w; st.h = h; st.valid = false; st.cur = 0;
	}
	if (!st.raw || !st.den || !st.den2 || !st.den3 || !st.resolved || !st.hist[0] || !st.hist[1] || !st.histZ[0] || !st.histZ[1]) return;

	static const int kRays[4] = { 0, 2, 4, 8 }, kSteps[4] = { 0, 8, 12, 16 };   // Off, Low, Medium, High
	const int q = ssgiQuality < 1 ? 1 : (ssgiQuality > 3 ? 3 : ssgiQuality);
	const float4x4 invProj = gbufProj.Inverse();
	auto fill = [&](int tw, int th, float pw, float p0, float p1, float p2, float p3)
	{
		MapHelper<SSGIData> cb(context, ssgiCB, MAP_WRITE, MAP_FLAG_DISCARD);
		cb->view = gbufView; cb->proj = gbufProj; cb->invProj = invProj;
		cb->res[0] = (float)tw; cb->res[1] = (float)th; cb->res[2] = 1.0f / tw; cb->res[3] = 1.0f / th;
		cb->params[0] = ssgiRadius; cb->params[1] = ssgiIntensity; cb->params[2] = (float)kRays[q]; cb->params[3] = pw;
		cb->params2[0] = p0; cb->params2[1] = p1; cb->params2[2] = p2; cb->params2[3] = p3;
	};
	auto draw = [&](PostPipe& pp, ITexture* dst, ITexture* dst2, ITexture* dst3, int tw, int th)
	{
		ITextureView* rtv[3] = { dst->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET), dst2 ? dst2->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET) : nullptr, dst3 ? dst3->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET) : nullptr };
		context->SetRenderTargets(dst3 ? 3 : (dst2 ? 2 : 1), rtv, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		Viewport vp; vp.TopLeftX = 0; vp.TopLeftY = 0; vp.Width = (float)tw; vp.Height = (float)th; vp.MinDepth = 0; vp.MaxDepth = 1;
		context->SetViewports(1, &vp, tw, th);
		context->SetPipelineState(pp.pso);
		context->CommitShaderResources(pp.srb, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		DrawAttribs da{3, DRAW_FLAG_VERIFY_STATES};
		context->Draw(da);
	};
	ITextureView* whiteSRV = whiteTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
	ITextureView* velSRV = gbufVelSRV ? gbufVelSRV : whiteSRV;

	// 1) trace at half resolution; params.w = steps, params2 = (phase, LDR history, white point, 0)
	fill(lw, lh, (float)kSteps[q], (float)(ssgiFrame % 16), hdr ? 0.0f : 1.0f, sky.whitePoint, 0.0f);
	if (ssgiPipe.gbufVar)  ssgiPipe.gbufVar->Set(gbufSRV);
	if (ssgiPipe.depthVar) ssgiPipe.depthVar->Set(gbufDepthSRV);
	if (ssgiPipe.velVar)   ssgiPipe.velVar->Set(velSRV);
	if (ssgiPipe.histVar)  ssgiPipe.histVar->Set(st.lit->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
	draw(ssgiPipe, st.raw, nullptr, nullptr, lw, lh);
	// 2) denoise (still half res)
	fill(lw, lh, 0.0f, 0.0f, 0.0f, (float)lw, (float)lh);
	ITexture* histPrev = st.hist[st.cur ^ 1]; ITexture* histCur = st.hist[st.cur];
	if (ssgiResolvePipe.srcVar)   ssgiResolvePipe.srcVar->Set(st.raw->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
	if (ssgiResolvePipe.depthVar) ssgiResolvePipe.depthVar->Set(gbufDepthSRV);
	if (ssgiResolvePipe.gbufVar)  ssgiResolvePipe.gbufVar->Set(gbufSRV);
	if (ssgiResolvePipe.velVar)   ssgiResolvePipe.velVar->Set(velSRV);
	if (ssgiResolvePipe.histVar)  ssgiResolvePipe.histVar->Set(histPrev->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
	if (ssgiResolvePipe.objIdVar) ssgiResolvePipe.objIdVar->Set(st.histZ[st.cur ^ 1]->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
	draw(ssgiResolvePipe, st.den, st.den2, st.den3, lw, lh);
	// 3) upsample + temporal at full resolution
	fill(w, h, 1.0f, (st.valid && gbufVelSRV) ? 1.0f : 0.0f, 0.92f, (float)lw, (float)lh);
	if (ssgiResolvePipe.srcVar) ssgiResolvePipe.srcVar->Set(st.den->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
	draw(ssgiResolvePipe, st.resolved, histCur, st.histZ[st.cur], w, h);
	context->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE);
	st.cur ^= 1; st.valid = true;
	++ssgiFrame;
	screenGISRV = st.resolved->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
}

// endCamera: keep this camera's resolved scene colour as next frame's lit history.
void NukeDiligent::Impl::KeepSSGILitHistory(ITextureView* sceneSRV, int w, int h)
{
	if (ssgiQuality <= 0 || !sceneSRV || w <= 0 || h <= 0) return;
	SSGIState& st = ssgiStates[curCamKey];
	st.lastUsed = frameId;
	ITexture* src = sceneSRV->GetTexture();
	if (!src) return;
	const TEXTURE_FORMAT fmt = src->GetDesc().Format;
	if (!st.lit || st.litW != w || st.litH != h || st.lit->GetDesc().Format != fmt)
	{
		Trash(st.lit); st.lit.Release();
		TextureDesc td; td.Name = "SSGI lit history"; td.Type = RESOURCE_DIM_TEX_2D; td.Width = (Uint32)w; td.Height = (Uint32)h;
		td.MipLevels = 1; td.Format = fmt; td.BindFlags = BIND_SHADER_RESOURCE | BIND_RENDER_TARGET; td.Usage = USAGE_DEFAULT;
		device->CreateTexture(td, nullptr, &st.lit);
		st.litW = w; st.litH = h; st.litValid = false;
	}
	if (!st.lit) return;
	context->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE);
	BlitTexture(sceneSRV, st.lit);   // same format: a plain copy
	st.litValid = true;
}
