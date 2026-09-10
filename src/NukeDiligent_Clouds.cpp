// NukeDiligent_Clouds.cpp - volumetric clouds (VL3): a cloud layer as a shell around the planet,
// its medium a weather map x tileable Perlin-Worley base x Worley erosion (Nubis / Horizon-class),
// ray-marched per camera at a reduced resolution with a jittered two-level march, lit by the sun
// (Beer-Lambert, multiple-scattering octaves, dual-lobe phase, powder) and the sky, resolved to
// full resolution by temporal reprojection, composited before the fog (the atmosphere lies over
// the clouds), occluding the sun shafts' source and casting a sweeping shadow map on the ground.
// The description comes from the World's Environment component (iRender::setClouds).
#include "NukeDiligentImpl.h"
#include <chrono>
#include <cmath>
#include <cstring>

using namespace Diligent;
using std::string; using std::vector; using std::cout; using std::endl;

struct CloudCBData
{
	float4x4 invVP, prevVP;
	float cam[4], sunDir[4], sunCol[4], skyTop[4], skyHor[4], layer[4], shape[4], cover[4], phase[4], wind[4], screen[4], misc[4], shadow[4];
};

bool NukeDiligent::Impl::BuildCloudPipes()
{
	static_assert(sizeof(CloudCBData) == kCloudCBSize, "CloudCB size");
	const string csG = shaderSource("clouds_gen.cs"), csM = shaderSource("clouds.cs"), csT = shaderSource("clouds_temporal.cs"), csS = shaderSource("clouds_shadow.cs");
	const string vs = shaderSource("post.vs"), psA = shaderSource("clouds_apply.ps");
	if (csG.empty() || csM.empty() || csT.empty() || csS.empty() || vs.empty() || psA.empty()) return false;
	auto sf = ShaderFactory();   // clouds.hlsli
	auto compile = [&](const string& src, const char* dbg, SHADER_TYPE type, RefCntAutoPtr<IShader>& out)
	{
		ShaderCreateInfo sci; sci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL; sci.pShaderSourceStreamFactory = sf;
		sci.Desc = {dbg, type, true}; sci.Source = src.c_str();
		CreateShaderCached(sci, &out);
		return out != nullptr;
	};
	if (!cloudCB)
	{
		BufferDesc d; d.Name = "CloudCB"; d.Size = kCloudCBSize; d.Usage = USAGE_DYNAMIC;
		d.BindFlags = BIND_UNIFORM_BUFFER; d.CPUAccessFlags = CPU_ACCESS_WRITE;
		device->CreateBuffer(d, nullptr, &cloudCB);
		if (!cloudCB) return false;
	}
	SamplerDesc wrap; wrap.MinFilter = FILTER_TYPE_LINEAR; wrap.MagFilter = FILTER_TYPE_LINEAR; wrap.MipFilter = FILTER_TYPE_LINEAR;
	wrap.AddressU = TEXTURE_ADDRESS_WRAP; wrap.AddressV = TEXTURE_ADDRESS_WRAP; wrap.AddressW = TEXTURE_ADDRESS_WRAP;
	SamplerDesc clampS = wrap; clampS.AddressU = TEXTURE_ADDRESS_CLAMP; clampS.AddressV = TEXTURE_ADDRESS_CLAMP; clampS.AddressW = TEXTURE_ADDRESS_CLAMP;
	SamplerDesc pt = clampS; pt.MinFilter = FILTER_TYPE_POINT; pt.MagFilter = FILTER_TYPE_POINT; pt.MipFilter = FILTER_TYPE_POINT;
	auto buildCS = [&](const char* dbg, IShader* cs, const vector<std::pair<const char*, SamplerDesc>>& samplers, RefCntAutoPtr<IPipelineState>& pso, RefCntAutoPtr<IShaderResourceBinding>& srb)
	{
		ComputePipelineStateCreateInfo ci; ci.PSODesc.Name = dbg;
		ShaderResourceVariableDesc vars[] = { {SHADER_TYPE_COMPUTE, "CloudCB", SHADER_RESOURCE_VARIABLE_TYPE_STATIC} };
		vector<ImmutableSamplerDesc> imms;
		for (const auto& s : samplers) imms.push_back({SHADER_TYPE_COMPUTE, s.first, s.second});
		ci.PSODesc.ResourceLayout.Variables = vars; ci.PSODesc.ResourceLayout.NumVariables = 1;
		ci.PSODesc.ResourceLayout.DefaultVariableType = SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC;
		ci.PSODesc.ResourceLayout.ImmutableSamplers = imms.data(); ci.PSODesc.ResourceLayout.NumImmutableSamplers = (Uint32)imms.size();
		ci.pCS = cs;
		CreateComputePipelineStateCached(ci, &pso);
		if (!pso) return false;
		if (auto* v = pso->GetStaticVariableByName(SHADER_TYPE_COMPUTE, "CloudCB")) v->Set(cloudCB);
		pso->CreateShaderResourceBinding(&srb, true);
		return srb != nullptr;
	};
	RefCntAutoPtr<IShader> sG, sM, sT, sS, sV, sP;
	if (!compile(csG, "Clouds gen CS", SHADER_TYPE_COMPUTE, sG)) return false;
	if (!compile(csM, "Clouds march CS", SHADER_TYPE_COMPUTE, sM)) return false;
	if (!compile(csT, "Clouds temporal CS", SHADER_TYPE_COMPUTE, sT)) return false;
	if (!compile(csS, "Clouds shadow CS", SHADER_TYPE_COMPUTE, sS)) return false;
	if (!compile(vs, "Clouds apply VS", SHADER_TYPE_VERTEX, sV)) return false;
	if (!compile(psA, "Clouds apply PS", SHADER_TYPE_PIXEL, sP)) return false;
	// immutable samplers per pass: only the textures each shader actually samples (Diligent warns on the rest)
	const vector<std::pair<const char*, SamplerDesc>> noiseSamp = { {"g_CloudBase", wrap}, {"g_CloudDetail", wrap}, {"g_CloudWeather", wrap} };
	const vector<std::pair<const char*, SamplerDesc>> coarseSamp = { {"g_CloudBase", wrap}, {"g_CloudWeather", wrap} };
	const vector<std::pair<const char*, SamplerDesc>> tempSamp = { {"g_CloudIn", clampS}, {"g_CloudHist", clampS} };
	const vector<std::pair<const char*, SamplerDesc>> noSamp;
	RefCntAutoPtr<IPipelineState> pG, pM, pT, pS; RefCntAutoPtr<IShaderResourceBinding> bG, bM, bT, bS;
	if (!buildCS("Clouds gen PSO",      sG, noSamp,     pG, bG)) return false;
	if (!buildCS("Clouds march PSO",    sM, noiseSamp,  pM, bM)) return false;
	if (!buildCS("Clouds temporal PSO", sT, tempSamp,   pT, bT)) return false;
	if (!buildCS("Clouds shadow PSO",   sS, coarseSamp, pS, bS)) return false;
	PostPipe pp;
	{
		GraphicsPipelineStateCreateInfo ci; ci.PSODesc.Name = "Clouds apply PSO";
		auto& gp = ci.GraphicsPipeline;
		gp.NumRenderTargets = 1; gp.RTVFormats[0] = HDR_FMT; gp.DSVFormat = TEX_FORMAT_UNKNOWN;
		gp.PrimitiveTopology = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; gp.RasterizerDesc.CullMode = CULL_MODE_NONE;
		gp.DepthStencilDesc.DepthEnable = False; gp.InputLayout.NumElements = 0;
		ShaderResourceVariableDesc vars[] = {
			{SHADER_TYPE_PIXEL, "g_Source",    SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
			{SHADER_TYPE_PIXEL, "g_Depth",     SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
			{SHADER_TYPE_PIXEL, "g_Clouds",    SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
			{SHADER_TYPE_PIXEL, "g_CloudDist", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
			{SHADER_TYPE_PIXEL, "CloudCB",     SHADER_RESOURCE_VARIABLE_TYPE_STATIC},
		};
		ImmutableSamplerDesc imms[] = {{SHADER_TYPE_PIXEL, "g_Source", clampS}, {SHADER_TYPE_PIXEL, "g_Depth", pt}, {SHADER_TYPE_PIXEL, "g_Clouds", clampS}};
		ci.PSODesc.ResourceLayout.Variables = vars; ci.PSODesc.ResourceLayout.NumVariables = 5;
		ci.PSODesc.ResourceLayout.DefaultVariableType = SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC;
		ci.PSODesc.ResourceLayout.ImmutableSamplers = imms; ci.PSODesc.ResourceLayout.NumImmutableSamplers = 3;
		ci.pVS = sV; ci.pPS = sP;
		CreateGraphicsPipelineStateCached(ci, &pp.pso);
		if (!pp.pso) return false;
		if (auto* c = pp.pso->GetStaticVariableByName(SHADER_TYPE_PIXEL, "CloudCB")) c->Set(cloudCB);
		pp.pso->CreateShaderResourceBinding(&pp.srb, true);
		pp.srcVar   = pp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Source");
		pp.depthVar = pp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Depth");
		pp.histVar  = pp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Clouds");
		pp.volVar   = pp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_CloudDist");
	}
	cloudGenPSO = pG; cloudGenSRB = bG; cloudMarchPSO = pM; cloudMarchSRB = bM; cloudTemporalPSO = pT; cloudTemporalSRB = bT; cloudShadowPSO = pS; cloudShadowSRB = bS;
	cloudApplyPipe = std::move(pp);
	if (!BuildCloudProbePipe()) cout << "[NukeDiligent]	clouds probe pipeline failed to build; probes capture without clouds" << endl;
	return true;
}

// The probe-face composite: blends the face's march onto the face in place (ONE / SRC_ALPHA);
// built for the current sample count + scene format like the sky, rebuilt when they change.
bool NukeDiligent::Impl::BuildCloudProbePipe()
{
	const string vs = shaderSource("post.vs"), ps = shaderSource("clouds_probe.ps");
	if (vs.empty() || ps.empty() || !cloudCB) return false;
	auto sf = ShaderFactory();
	RefCntAutoPtr<IShader> sV, sP;
	ShaderCreateInfo sci; sci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL; sci.pShaderSourceStreamFactory = sf;
	sci.Desc = {"Clouds probe VS", SHADER_TYPE_VERTEX, true}; sci.Source = vs.c_str(); CreateShaderCached(sci, &sV);
	sci.Desc = {"Clouds probe PS", SHADER_TYPE_PIXEL, true};  sci.Source = ps.c_str(); CreateShaderCached(sci, &sP);
	if (!sV || !sP) return false;
	const Uint8 smp = samples; const TEXTURE_FORMAT fmt = SceneFmt();
	SamplerDesc lin; lin.MinFilter = FILTER_TYPE_LINEAR; lin.MagFilter = FILTER_TYPE_LINEAR; lin.MipFilter = FILTER_TYPE_LINEAR;
	lin.AddressU = TEXTURE_ADDRESS_CLAMP; lin.AddressV = TEXTURE_ADDRESS_CLAMP; lin.AddressW = TEXTURE_ADDRESS_CLAMP;
	GraphicsPipelineStateCreateInfo ci; ci.PSODesc.Name = "Clouds probe PSO";
	auto& gp = ci.GraphicsPipeline;
	gp.NumRenderTargets = 1; gp.RTVFormats[0] = fmt; gp.DSVFormat = TEX_FORMAT_D32_FLOAT;   // the face's depth is bound (test off)
	gp.PrimitiveTopology = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; gp.RasterizerDesc.CullMode = CULL_MODE_NONE;
	gp.DepthStencilDesc.DepthEnable = False; gp.DepthStencilDesc.DepthWriteEnable = False;
	gp.SmplDesc.Count = smp; gp.InputLayout.NumElements = 0;
	auto& bl = gp.BlendDesc.RenderTargets[0];
	bl.BlendEnable = True; bl.SrcBlend = BLEND_FACTOR_ONE; bl.DestBlend = BLEND_FACTOR_SRC_ALPHA; bl.BlendOp = BLEND_OPERATION_ADD;
	bl.SrcBlendAlpha = BLEND_FACTOR_ZERO; bl.DestBlendAlpha = BLEND_FACTOR_ONE; bl.BlendOpAlpha = BLEND_OPERATION_ADD;
	ShaderResourceVariableDesc vars[] = { {SHADER_TYPE_PIXEL, "g_Clouds", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC}, {SHADER_TYPE_PIXEL, "CloudCB", SHADER_RESOURCE_VARIABLE_TYPE_STATIC} };
	ImmutableSamplerDesc imms[] = {{SHADER_TYPE_PIXEL, "g_Clouds", lin}};
	ci.PSODesc.ResourceLayout.Variables = vars; ci.PSODesc.ResourceLayout.NumVariables = 2;
	ci.PSODesc.ResourceLayout.ImmutableSamplers = imms; ci.PSODesc.ResourceLayout.NumImmutableSamplers = 1;
	ci.pVS = sV; ci.pPS = sP;
	RefCntAutoPtr<IPipelineState> pso;
	CreateGraphicsPipelineStateCached(ci, &pso);
	if (!pso) return false;
	if (auto* c = pso->GetStaticVariableByName(SHADER_TYPE_PIXEL, "CloudCB")) c->Set(cloudCB);
	RefCntAutoPtr<IShaderResourceBinding> srb;
	pso->CreateShaderResourceBinding(&srb, true);
	if (!srb) return false;
	cloudProbePSO = pso; cloudProbeSRB = srb; cloudProbeVar = srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Clouds");
	cloudProbeSmp = smp; cloudProbeFmt = fmt;
	return true;
}

// A reflection-probe face, right after its sky: the layer marched at the face's resolution
// (no depth: the geometry is drawn after and covers it; no history: one clean pass) and
// blended onto the face. Reflections and the probe's IBL see the clouds.
void NukeDiligent::Impl::RunCloudsCubeFace(ITextureView* rtv, ITextureView* dsv, int res)
{
	if (!clouds.enabled || cloudsFailed || !cloudMarchPSO || !rtv || res <= 0) return;
	GenerateCloudNoise();
	if (cloudsFailed || !cloudNoiseReady) return;
	if (!cloudProbePSO || cloudProbeSmp != samples || cloudProbeFmt != SceneFmt())
	{   // stale for this sample count / format: rebuilt off the render thread, the face goes without until then
		if (!cloudsBuilding.exchange(true))
			EnqueueBuild([this] { if (!BuildCloudProbePipe()) cout << "[NukeDiligent]	clouds probe pipeline failed to build; probes capture without clouds" << endl; },
			             [this] { cloudsBuilding = false; }, kPrioGBuffer, "Clouds probe pipeline");
		return;
	}
	if (!cloudProbeMarch || cloudProbeRes != res)
	{
		Trash(cloudProbeMarch); cloudProbeMarch.Release(); Trash(cloudProbeDist); cloudProbeDist.Release();
		TextureDesc td; td.Type = RESOURCE_DIM_TEX_2D; td.Width = td.Height = (Uint32)res; td.MipLevels = 1; td.Usage = USAGE_DEFAULT;
		td.BindFlags = BIND_SHADER_RESOURCE | BIND_UNORDERED_ACCESS;
		td.Name = "Clouds probe march"; td.Format = HDR_FMT; device->CreateTexture(td, nullptr, &cloudProbeMarch);
		td.Name = "Clouds probe distance"; td.Format = TEX_FORMAT_RG16_FLOAT; device->CreateTexture(td, nullptr, &cloudProbeDist);
		cloudProbeRes = res;
		if (!cloudProbeMarch || !cloudProbeDist) return;
	}
	auto srv = [](ITexture* t) { return t->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE); };
	auto set = [&](const char* n, IDeviceObject* o) { if (auto* v = cloudMarchSRB->GetVariableByName(SHADER_TYPE_COMPUTE, n)) v->Set(o); };
	context->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE);
	// full-res screen (1,1): the depth Load lands on the 1x1 white stand-in = sky everywhere
	FillCloudCB(res, res, 1, 1, 0.0f, nullptr, 0.0f, 0.0f, 1.0f, true);
	set("g_CloudBase", srv(cloudBase)); set("g_CloudDetail", srv(cloudDetail)); set("g_CloudWeather", srv(cloudWeather));
	set("g_Depth", whiteTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
	set("g_CloudOut", cloudProbeMarch->GetDefaultView(TEXTURE_VIEW_UNORDERED_ACCESS)); set("g_CloudDist", cloudProbeDist->GetDefaultView(TEXTURE_VIEW_UNORDERED_ACCESS));
	context->SetPipelineState(cloudMarchPSO);
	context->CommitShaderResources(cloudMarchSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	context->DispatchCompute(DispatchComputeAttribs((Uint32)(res + 7) / 8, (Uint32)(res + 7) / 8, 1));
	// back onto the face: the blend
	context->SetRenderTargets(1, &rtv, dsv, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	Viewport vp; vp.TopLeftX = 0; vp.TopLeftY = 0; vp.Width = (float)res; vp.Height = (float)res; vp.MinDepth = 0; vp.MaxDepth = 1;
	context->SetViewports(1, &vp, res, res);
	if (cloudProbeVar) cloudProbeVar->Set(srv(cloudProbeMarch));
	if (!cloudProbeSaid) { cloudProbeSaid = true; cout << "[NukeDiligent]	clouds: reflection-probe faces carry the layer (res " << res << ")" << endl; }
	context->SetPipelineState(cloudProbePSO);
	context->CommitShaderResources(cloudProbeSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	DrawAttribs da{3, DRAW_FLAG_VERIFY_STATES};
	context->Draw(da);
}

// The clouds' constants for one pass; `res` = the pass's target size (march / shadow map),
// `mode` = generation mode or temporal blend (g_ClWind.w).
void NukeDiligent::Impl::FillCloudCB(int rw, int rh, int w, int h, float mode, const float4x4* prevVP, float shadowOx, float shadowOz, float shadowSize, bool probe)
{
	MapHelper<CloudCBData> cb(context, cloudCB, MAP_WRITE, MAP_FLAG_DISCARD);
	if (cb == nullptr) return;
	const NukeCloudsDesc& c = clouds;
	const float4x4 proj = probe ? curProj : curProjNoJitter;   // a cube face sets curProj only
	cb->invVP = (curView * proj).Inverse();
	cb->prevVP = prevVP ? *prevVP : (curView * proj);
	cb->cam[0] = curCamPos[0]; cb->cam[1] = curCamPos[1]; cb->cam[2] = curCamPos[2]; cb->cam[3] = probe ? 0.0f : (float)(frameId % 4096);
	// the sun: the first directional light (the sky's sun), as the sun shafts take it
	float sd[3] = { 0.0f, 1.0f, 0.0f }, sc[3] = { 1.0f, 1.0f, 1.0f }, si = 0.0f;
	for (const NukeLight& l : lights) if (l.type == 0)
	{
		sd[0] = -l.dir[0]; sd[1] = -l.dir[1]; sd[2] = -l.dir[2];
		const float len = std::sqrt(sd[0] * sd[0] + sd[1] * sd[1] + sd[2] * sd[2]); if (len > 1e-6f) for (float& v : sd) v /= len;
		sc[0] = l.color[0]; sc[1] = l.color[1]; sc[2] = l.color[2]; si = std::max(l.intensity, 0.0f);
		break;
	}
	cb->sunDir[0] = sd[0]; cb->sunDir[1] = sd[1]; cb->sunDir[2] = sd[2]; cb->sunDir[3] = si * std::max(c.sunIntensity, 0.0f);
	cb->sunCol[0] = sc[0]; cb->sunCol[1] = sc[1]; cb->sunCol[2] = sc[2]; cb->sunCol[3] = std::max(c.ambientIntensity, 0.0f);
	cb->skyTop[0] = sky.top[0]; cb->skyTop[1] = sky.top[1]; cb->skyTop[2] = sky.top[2]; cb->skyTop[3] = sky.skyIntensity;
	cb->skyHor[0] = sky.horizon[0]; cb->skyHor[1] = sky.horizon[1]; cb->skyHor[2] = sky.horizon[2]; cb->skyHor[3] = 6371000.0f;
	const int steps[3] = { 40, 64, 96 };
	cb->layer[0] = c.bottom; cb->layer[1] = c.bottom + std::max(c.thickness, 10.0f); cb->layer[2] = std::max(c.maxDistance, 1000.0f); cb->layer[3] = (float)steps[std::max(0, std::min(2, c.quality))];
	cb->shape[0] = std::max(c.shapeScale, 10.0f); cb->shape[1] = std::max(c.detailScale, 1.0f); cb->shape[2] = std::max(c.weatherScale, 100.0f); cb->shape[3] = std::max(0.0f, std::min(1.0f, c.erosion));
	cb->cover[0] = std::max(0.0f, std::min(1.0f, c.coverage)); cb->cover[1] = std::max(0.0f, std::min(1.0f, c.type)); cb->cover[2] = std::max(c.density, 0.0f); cb->cover[3] = std::max(0.0f, std::min(1.0f, c.silverLining));
	cb->phase[0] = std::max(-0.99f, std::min(0.99f, c.forwardScatter)); cb->phase[1] = std::max(-0.99f, std::min(0.99f, c.backScatter));
	cb->phase[2] = std::max(0.0f, std::min(1.0f, c.multiScatter)); cb->phase[3] = std::max(0.05f, std::min(1.0f, c.multiScatterFalloff));
	cb->wind[0] = cloudWindX; cb->wind[1] = cloudWindZ; cb->wind[2] = cloudClock; cb->wind[3] = mode;
	cb->screen[0] = (float)rw; cb->screen[1] = (float)rh; cb->screen[2] = (float)w; cb->screen[3] = (float)h;
	cb->misc[0] = curNear; cb->misc[1] = curFar; cb->misc[2] = hdr ? 0.0f : 1.0f; cb->misc[3] = sky.whitePoint;
	cb->shadow[0] = shadowOx; cb->shadow[1] = shadowOz; cb->shadow[2] = 1.0f / std::max(shadowSize, 1.0f); cb->shadow[3] = c.shadows ? std::max(0.0f, std::min(1.0f, c.shadowStrength)) : 0.0f;
}

// The tileable noises and the weather map, once (they depend on nothing the user changes;
// every setting applies at march time).
void NukeDiligent::Impl::GenerateCloudNoise()
{
	if (cloudNoiseReady || !cloudGenPSO) return;
	auto make3 = [&](RefCntAutoPtr<ITexture>& t, const char* name, int n)
	{
		TextureDesc td; td.Name = name; td.Type = RESOURCE_DIM_TEX_3D; td.Width = td.Height = (Uint32)n; td.Depth = (Uint32)n;
		td.MipLevels = 1; td.Format = TEX_FORMAT_RGBA8_UNORM; td.BindFlags = BIND_SHADER_RESOURCE | BIND_UNORDERED_ACCESS; td.Usage = USAGE_DEFAULT;
		device->CreateTexture(td, nullptr, &t);
	};
	make3(cloudBase, "Clouds base noise", 128); make3(cloudDetail, "Clouds detail noise", 32);
	{
		TextureDesc td; td.Name = "Clouds weather"; td.Type = RESOURCE_DIM_TEX_2D; td.Width = td.Height = 512;
		td.MipLevels = 1; td.Format = TEX_FORMAT_RGBA8_UNORM; td.BindFlags = BIND_SHADER_RESOURCE | BIND_UNORDERED_ACCESS; td.Usage = USAGE_DEFAULT;
		device->CreateTexture(td, nullptr, &cloudWeather);
	}
	if (!cloudBase || !cloudDetail || !cloudWeather) { cloudsFailed = true; return; }
	auto set = [&](const char* n, IDeviceObject* o) { if (auto* v = cloudGenSRB->GetVariableByName(SHADER_TYPE_COMPUTE, n)) v->Set(o); };
	auto gen = [&](float mode, ITexture* out3, ITexture* out2, int gx, int gy, int gz)
	{
		FillCloudCB(0, 0, 0, 0, mode, nullptr, 0.0f, 0.0f, 1.0f);
		set("g_Out3D", out3->GetDefaultView(TEXTURE_VIEW_UNORDERED_ACCESS));
		set("g_Out2D", out2->GetDefaultView(TEXTURE_VIEW_UNORDERED_ACCESS));
		context->SetPipelineState(cloudGenPSO);
		context->CommitShaderResources(cloudGenSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		context->DispatchCompute(DispatchComputeAttribs((Uint32)gx, (Uint32)gy, (Uint32)gz));
	};
	// a 2D stand-in for the 3D passes' unused g_Out2D and vice versa: the weather map / the detail cube
	gen(0.0f, cloudBase,   cloudWeather, 16, 16, 32);
	gen(1.0f, cloudDetail, cloudWeather, 4, 4, 8);
	gen(2.0f, cloudDetail, cloudWeather, 64, 64, 1);
	cloudNoiseReady = true;
}

// endCamera, before the fog composite: the clouds over the scene (own texture), or the input
// when the clouds are off.
ITextureView* NukeDiligent::Impl::RunClouds(ITextureView* sceneSRV, int w, int h)
{
	cloudCurSRV = nullptr;
	if (!clouds.enabled || !sceneSRV || !gbufDepthSRV || debugView != 0 || w <= 0 || h <= 0 || cloudsFailed) return sceneSRV;
	if (!cloudMarchPSO)
	{
		if (!cloudsBuilding.exchange(true))
			EnqueueBuild([this] { if (!BuildCloudPipes()) { cloudsFailed = true; cout << "[NukeDiligent]\tclouds pipeline failed to build; clouds stay off" << endl; } },
			             [this] { cloudsBuilding = false; }, kPrioGBuffer, "Clouds pipeline");
		return sceneSRV;
	}
	GenerateCloudNoise();
	if (cloudsFailed || !cloudNoiseReady) return sceneSRV;
	// the clock and the wind: the clouds drift with the global wind (scaled) plus their own drift
	{
		const double t = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
		float dt = 0.0f;
		if (cloudLastTime >= 0.0) dt = (float)std::max(0.0, std::min(0.1, t - cloudLastTime));
		cloudLastTime = t;
		if (cloudFrameStamp != frameId)   // once a frame, whatever the number of camera passes
		{
			cloudFrameStamp = frameId;
			cloudClock += dt;
			const float rad = clouds.driftDirection * 0.01745329252f;
			cloudWindX += (windDirStrength[0] * windDirStrength[3] * clouds.windInfluence + std::cos(rad) * clouds.driftSpeed) * dt;
			cloudWindZ += (windDirStrength[2] * windDirStrength[3] * clouds.windInfluence + std::sin(rad) * clouds.driftSpeed) * dt;
		}
	}
	CloudState& st = cloudStates[curCamKey];
	st.lastUsed = frameId;
	const int scale = (clouds.quality <= 0) ? 4 : 2;
	const int mw = std::max(1, (w + scale - 1) / scale), mh = std::max(1, (h + scale - 1) / scale);
	if (!st.march || st.mw != mw || st.mh != mh || st.w != w || st.h != h)
	{
		auto drop = [&](RefCntAutoPtr<ITexture>& t) { Trash(t); t.Release(); };
		drop(st.march); drop(st.dist); drop(st.hist[0]); drop(st.hist[1]); drop(st.distFull); drop(st.out);
		auto make = [&](RefCntAutoPtr<ITexture>& t, const char* name, int tw, int th, TEXTURE_FORMAT fmt, Diligent::BIND_FLAGS bind)
		{
			TextureDesc td; td.Name = name; td.Type = RESOURCE_DIM_TEX_2D; td.Width = (Uint32)tw; td.Height = (Uint32)th;
			td.MipLevels = 1; td.Format = fmt; td.BindFlags = bind; td.Usage = USAGE_DEFAULT;
			device->CreateTexture(td, nullptr, &t);
		};
		const Diligent::BIND_FLAGS cs = BIND_SHADER_RESOURCE | BIND_UNORDERED_ACCESS;
		make(st.march, "Clouds march", mw, mh, HDR_FMT, cs);
		make(st.dist, "Clouds distance", mw, mh, TEX_FORMAT_RG16_FLOAT, cs);
		make(st.hist[0], "Clouds resolved A", w, h, HDR_FMT, cs);
		make(st.hist[1], "Clouds resolved B", w, h, HDR_FMT, cs);
		make(st.distFull, "Clouds distance (full)", w, h, TEX_FORMAT_RG16_FLOAT, cs);
		make(st.out, "Clouds composite", w, h, HDR_FMT, BIND_SHADER_RESOURCE | BIND_RENDER_TARGET);
		st.mw = mw; st.mh = mh; st.w = w; st.h = h; st.cur = 0; st.valid = false;
		if (!st.march || !st.dist || !st.hist[0] || !st.hist[1] || !st.distFull || !st.out) return sceneSRV;
	}
	auto srv = [](ITexture* t) { return t->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE); };
	auto uav = [](ITexture* t) { return t->GetDefaultView(TEXTURE_VIEW_UNORDERED_ACCESS); };
	auto set = [&](IShaderResourceBinding* srb, const char* n, IDeviceObject* o) { if (auto* v = srb->GetVariableByName(SHADER_TYPE_COMPUTE, n)) v->Set(o); };
	context->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE);
	// 1) the march
	FillCloudCB(mw, mh, w, h, 0.0f, nullptr, 0.0f, 0.0f, 1.0f);
	set(cloudMarchSRB, "g_CloudBase", srv(cloudBase)); set(cloudMarchSRB, "g_CloudDetail", srv(cloudDetail)); set(cloudMarchSRB, "g_CloudWeather", srv(cloudWeather));
	set(cloudMarchSRB, "g_Depth", gbufDepthSRV);
	set(cloudMarchSRB, "g_CloudOut", uav(st.march)); set(cloudMarchSRB, "g_CloudDist", uav(st.dist));
	context->SetPipelineState(cloudMarchPSO);
	context->CommitShaderResources(cloudMarchSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	context->DispatchCompute(DispatchComputeAttribs((Uint32)(mw + 7) / 8, (Uint32)(mh + 7) / 8, 1));
	// 2) the temporal resolve (history weight 0.9 while the view holds, lower as it turns)
	float blend = 0.0f;
	if (st.valid)
	{
		float pf[16], cf[16]; memcpy(pf, &st.prevView, sizeof(pf)); memcpy(cf, &curView, sizeof(cf));
		const float dotf = pf[2] * cf[2] + pf[6] * cf[6] + pf[10] * cf[10];
		const float ang = std::acos(std::max(-1.0f, std::min(1.0f, dotf))) * 57.2958f;
		blend = 0.9f - 0.4f * std::max(0.0f, std::min(1.0f, ang / 2.0f));
	}
	FillCloudCB(mw, mh, w, h, blend, st.valid ? &st.prevVP : nullptr, 0.0f, 0.0f, 1.0f);
	set(cloudTemporalSRB, "g_CloudIn", srv(st.march)); set(cloudTemporalSRB, "g_CloudDistIn", srv(st.dist));
	set(cloudTemporalSRB, "g_CloudHist", srv(st.hist[st.cur ^ 1]));
	set(cloudTemporalSRB, "g_CloudResolved", uav(st.hist[st.cur])); set(cloudTemporalSRB, "g_CloudDistOut", uav(st.distFull));
	context->SetPipelineState(cloudTemporalPSO);
	context->CommitShaderResources(cloudTemporalSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	context->DispatchCompute(DispatchComputeAttribs((Uint32)(w + 7) / 8, (Uint32)(h + 7) / 8, 1));
	// 3) the composite
	ITextureView* resolved = srv(st.hist[st.cur]);
	if (cloudApplyPipe.srcVar)   cloudApplyPipe.srcVar->Set(sceneSRV);
	if (cloudApplyPipe.depthVar) cloudApplyPipe.depthVar->Set(gbufDepthSRV);
	if (cloudApplyPipe.histVar)  cloudApplyPipe.histVar->Set(resolved);
	if (cloudApplyPipe.volVar)   cloudApplyPipe.volVar->Set(srv(st.distFull));
	ITextureView* rtv = st.out->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET);
	context->SetRenderTargets(1, &rtv, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	Viewport vp; vp.TopLeftX = 0; vp.TopLeftY = 0; vp.Width = (float)w; vp.Height = (float)h; vp.MinDepth = 0; vp.MaxDepth = 1;
	context->SetViewports(1, &vp, w, h);
	context->SetPipelineState(cloudApplyPipe.pso);
	context->CommitShaderResources(cloudApplyPipe.srb, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	DrawAttribs da{3, DRAW_FLAG_VERIFY_STATES};
	context->Draw(da);
	context->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE);
	st.prevVP = curView * curProjNoJitter; st.prevView = curView; st.cur ^= 1; st.valid = true;
	cloudCurSRV = resolved;   // the sun shafts' source is occluded by it
	return srv(st.out);
}

// Once a frame: the sun's transmittance through the layer over a square around the camera
// (the surfaces multiply their sun shadow by it); nothing when the clouds are off.
void NukeDiligent::Impl::RunCloudShadow()
{
	cloudShadowSRV = nullptr;
	cloudShadowOrigin[0] = cloudShadowOrigin[1] = 0.0f; cloudShadowOrigin[2] = 1.0f; cloudShadowOrigin[3] = 0.0f;
	if (!clouds.enabled || !clouds.shadows || cloudsFailed || !cloudShadowPSO || !cloudNoiseReady) return;
	const int n = 512;
	if (!cloudShadowTex)
	{
		TextureDesc td; td.Name = "Clouds shadow map"; td.Type = RESOURCE_DIM_TEX_2D; td.Width = td.Height = (Uint32)n;
		td.MipLevels = 1; td.Format = TEX_FORMAT_R16_FLOAT; td.BindFlags = BIND_SHADER_RESOURCE | BIND_UNORDERED_ACCESS; td.Usage = USAGE_DEFAULT;
		device->CreateTexture(td, nullptr, &cloudShadowTex);
		if (!cloudShadowTex) { cloudsFailed = true; return; }
	}
	const float size = std::max(clouds.shadowArea, 100.0f);
	const float ox = curCamPos[0] - size * 0.5f, oz = curCamPos[2] - size * 0.5f;
	FillCloudCB(n, n, n, n, 0.0f, nullptr, ox, oz, size);
	auto set = [&](const char* nm, IDeviceObject* o) { if (auto* v = cloudShadowSRB->GetVariableByName(SHADER_TYPE_COMPUTE, nm)) v->Set(o); };
	auto srv = [](ITexture* t) { return t->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE); };
	set("g_CloudBase", srv(cloudBase)); set("g_CloudWeather", srv(cloudWeather));
	set("g_ShadowOut", cloudShadowTex->GetDefaultView(TEXTURE_VIEW_UNORDERED_ACCESS));
	context->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE);
	context->SetPipelineState(cloudShadowPSO);
	context->CommitShaderResources(cloudShadowSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	context->DispatchCompute(DispatchComputeAttribs((Uint32)n / 8, (Uint32)n / 8, 1));
	cloudShadowSRV = srv(cloudShadowTex);
	cloudShadowOrigin[0] = ox; cloudShadowOrigin[1] = oz; cloudShadowOrigin[2] = 1.0f / size; cloudShadowOrigin[3] = std::max(0.0f, std::min(1.0f, clouds.shadowStrength));
}
