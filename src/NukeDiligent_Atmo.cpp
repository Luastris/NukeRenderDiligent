// NukeDiligent_Atmo.cpp - the physical atmosphere (Environment mode Physical): Hillaire 2020
// LUTs on the Bruneton medium. Transmittance + multiple scattering rebuild when the medium
// changes; the camera's sky-view LUT and the aerial-perspective froxel volume run every camera
// pass (compute, before the sky draw); a 3-colour summary (zenith / horizon / ground) reads back
// two frames later and stands in for the procedural sky colours in the FrameCB, so water, GI
// probes, RT misses and the clouds' ambient follow the physical sky without their own LUT reads.
// The sky draw, world.ps IBL / sun transmittance and the clouds read the LUTs directly.
#include "NukeDiligentImpl.h"
#include <cmath>
#include <cstring>

using namespace Diligent;
using std::string; using std::vector; using std::cout; using std::endl;

struct AtmoCBData
{
	float radii[4], rayleigh[4], mie[4], ozone[4], ozone2[4], ground[4], sun[4], sunCol[4], cam[4], screen[4];
	float4x4 invVP;
	float misc[4];
};

static const int kAtmoSkyW = 192, kAtmoSkyH = 108, kAtmoAP = 32;

bool NukeDiligent::Impl::BuildAtmoPipes()
{
	static_assert(sizeof(AtmoCBData) == kAtmoCBSize, "AtmoCB size");
	const string cs = shaderSource("atmo_lut.cs"), vs = shaderSource("post.vs"), ps = shaderSource("atmo_apply.ps");
	if (cs.empty() || vs.empty() || ps.empty()) return false;
	auto sf = ShaderFactory();
	auto compile = [&](const string& src, const char* dbg, SHADER_TYPE type, RefCntAutoPtr<IShader>& out)
	{
		ShaderCreateInfo sci; sci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL; sci.pShaderSourceStreamFactory = sf;
		sci.Desc = {dbg, type, true}; sci.Source = src.c_str();
		CreateShaderCached(sci, &out);
		return out != nullptr;
	};
	if (!atmoCB)
	{
		BufferDesc d; d.Name = "AtmoCB"; d.Size = kAtmoCBSize; d.Usage = USAGE_DYNAMIC;
		d.BindFlags = BIND_UNIFORM_BUFFER; d.CPUAccessFlags = CPU_ACCESS_WRITE;
		device->CreateBuffer(d, nullptr, &atmoCB);
		if (!atmoCB) return false;
	}
	RefCntAutoPtr<IShader> sC, sV, sP;
	if (!compile(cs, "Atmosphere LUT CS", SHADER_TYPE_COMPUTE, sC)) return false;
	if (!compile(vs, "Atmosphere apply VS", SHADER_TYPE_VERTEX, sV)) return false;
	if (!compile(ps, "Atmosphere apply PS", SHADER_TYPE_PIXEL, sP)) return false;
	SamplerDesc lin = AtmoSampler();
	SamplerDesc pt = lin; pt.MinFilter = FILTER_TYPE_POINT; pt.MagFilter = FILTER_TYPE_POINT; pt.MipFilter = FILTER_TYPE_POINT;
	{
		ComputePipelineStateCreateInfo ci; ci.PSODesc.Name = "Atmosphere LUT PSO";
		ShaderResourceVariableDesc vars[] = { {SHADER_TYPE_COMPUTE, "AtmoCB", SHADER_RESOURCE_VARIABLE_TYPE_STATIC} };
		ImmutableSamplerDesc imms[] = {{SHADER_TYPE_COMPUTE, "g_AtTrans", lin}, {SHADER_TYPE_COMPUTE, "g_AtMulti", lin}, {SHADER_TYPE_COMPUTE, "g_AtSkyView", lin}};
		ci.PSODesc.ResourceLayout.Variables = vars; ci.PSODesc.ResourceLayout.NumVariables = 1;
		ci.PSODesc.ResourceLayout.DefaultVariableType = SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC;
		ci.PSODesc.ResourceLayout.ImmutableSamplers = imms; ci.PSODesc.ResourceLayout.NumImmutableSamplers = 3;
		ci.pCS = sC;
		RefCntAutoPtr<IPipelineState> pso;
		CreateComputePipelineStateCached(ci, &pso);
		if (!pso) return false;
		if (auto* v = pso->GetStaticVariableByName(SHADER_TYPE_COMPUTE, "AtmoCB")) v->Set(atmoCB);
		RefCntAutoPtr<IShaderResourceBinding> srb;
		pso->CreateShaderResourceBinding(&srb, true);
		if (!srb) return false;
		atmoLutPSO = pso; atmoLutSRB = srb;
	}
	{
		PostPipe pp;
		GraphicsPipelineStateCreateInfo ci; ci.PSODesc.Name = "Atmosphere apply PSO";
		auto& gp = ci.GraphicsPipeline;
		gp.NumRenderTargets = 1; gp.RTVFormats[0] = HDR_FMT; gp.DSVFormat = TEX_FORMAT_UNKNOWN;
		gp.PrimitiveTopology = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; gp.RasterizerDesc.CullMode = CULL_MODE_NONE;
		gp.DepthStencilDesc.DepthEnable = False; gp.InputLayout.NumElements = 0;
		ShaderResourceVariableDesc vars[] = {
			{SHADER_TYPE_PIXEL, "g_Source", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
			{SHADER_TYPE_PIXEL, "g_Depth",  SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
			{SHADER_TYPE_PIXEL, "g_AtAP",   SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
			{SHADER_TYPE_PIXEL, "AtmoCB",   SHADER_RESOURCE_VARIABLE_TYPE_STATIC},
		};
		ImmutableSamplerDesc imms[] = {{SHADER_TYPE_PIXEL, "g_Source", lin}, {SHADER_TYPE_PIXEL, "g_Depth", pt}, {SHADER_TYPE_PIXEL, "g_AtAP", lin}};
		ci.PSODesc.ResourceLayout.Variables = vars; ci.PSODesc.ResourceLayout.NumVariables = 4;
		ci.PSODesc.ResourceLayout.DefaultVariableType = SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC;
		ci.PSODesc.ResourceLayout.ImmutableSamplers = imms; ci.PSODesc.ResourceLayout.NumImmutableSamplers = 3;
		ci.pVS = sV; ci.pPS = sP;
		CreateGraphicsPipelineStateCached(ci, &pp.pso);
		if (!pp.pso) return false;
		if (auto* c = pp.pso->GetStaticVariableByName(SHADER_TYPE_PIXEL, "AtmoCB")) c->Set(atmoCB);
		pp.pso->CreateShaderResourceBinding(&pp.srb, true);
		pp.srcVar   = pp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Source");
		pp.depthVar = pp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Depth");
		pp.volVar   = pp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_AtAP");
		atmoApplyPipe = std::move(pp);
	}
	return true;
}

// The LUT sampler (linear, clamp): the sky-view / transmittance mappings never wrap.
SamplerDesc NukeDiligent::Impl::AtmoSampler()
{
	SamplerDesc s; s.MinFilter = FILTER_TYPE_LINEAR; s.MagFilter = FILTER_TYPE_LINEAR; s.MipFilter = FILTER_TYPE_LINEAR;
	s.AddressU = TEXTURE_ADDRESS_CLAMP; s.AddressV = TEXTURE_ADDRESS_CLAMP; s.AddressW = TEXTURE_ADDRESS_CLAMP;
	return s;
}

// Atmosphere on: the sky is physical and the medium's LUTs exist.
bool NukeDiligent::Impl::AtmoActive() const { return sky.mode == 2 && atmoLutPSO && atmoTrans && atmoMulti && !atmoFailed; }

// Fills AtmoCB for one pass (LUT mode / target size / camera).
void NukeDiligent::Impl::FillAtmoCB(int mode, int w, int h, bool probe)
{
	if (!atmoCB) return;
	MapHelper<AtmoCBData> cb(context, atmoCB, MAP_WRITE, MAP_FLAG_DISCARD);
	if (cb == nullptr) return;
	const NukeSky& s = sky;
	const float Rg = std::max(s.planetRadius, 1.0f), Rt = Rg + std::max(s.atmosphereHeight, 1.0f);
	cb->radii[0] = Rg; cb->radii[1] = Rt; cb->radii[2] = std::max(s.atmoDensity, 0.0f); cb->radii[3] = (float)s.mode;
	// scattering /km: the colour times the amount (x 1e-3: the UI amounts are in 1/Mm)
	const float ra = std::max(s.rayleighAmount, 0.0f) * 1e-3f;
	cb->rayleigh[0] = s.rayleighColor[0] * ra; cb->rayleigh[1] = s.rayleighColor[1] * ra; cb->rayleigh[2] = s.rayleighColor[2] * ra; cb->rayleigh[3] = std::max(s.rayleighHeight, 0.05f);
	cb->mie[0] = std::max(s.mieAmount, 0.0f) * 1e-3f; cb->mie[1] = std::max(s.mieAbsorption, 0.0f) * 1e-3f; cb->mie[2] = std::max(s.mieHeight, 0.05f); cb->mie[3] = std::max(-0.99f, std::min(0.99f, s.mieAnisotropy));
	const float oz = std::max(s.ozoneAmount, 0.0f) * 1e-3f;
	cb->ozone[0] = 0.650f * oz; cb->ozone[1] = 1.881f * oz; cb->ozone[2] = 0.085f * oz; cb->ozone[3] = 25.0f;
	cb->ozone2[0] = 15.0f; cb->ozone2[1] = std::max(s.aerialRange, 0.5f); cb->ozone2[2] = std::max(0.0f, std::min(1.0f, s.aerialStrength)); cb->ozone2[3] = (float)kAtmoSkyH;
	cb->ground[0] = s.groundAlbedo[0]; cb->ground[1] = s.groundAlbedo[1]; cb->ground[2] = s.groundAlbedo[2]; cb->ground[3] = (float)kAtmoSkyW;
	float sd[3] = { -s.sunDir[0], -s.sunDir[1], -s.sunDir[2] };
	{ const float l = std::sqrt(sd[0] * sd[0] + sd[1] * sd[1] + sd[2] * sd[2]); if (l > 1e-6f) for (float& v : sd) v /= l; else { sd[0] = 0; sd[1] = 1; sd[2] = 0; } }
	cb->sun[0] = sd[0]; cb->sun[1] = sd[1]; cb->sun[2] = sd[2]; cb->sun[3] = std::max(s.sunIntensity, 0.0f);
	cb->sunCol[0] = s.sunColor[0]; cb->sunCol[1] = s.sunColor[1]; cb->sunCol[2] = s.sunColor[2]; cb->sunCol[3] = std::max(s.skyIntensity, 0.0f);
	cb->cam[0] = curCamPos[0]; cb->cam[1] = curCamPos[1]; cb->cam[2] = curCamPos[2]; cb->cam[3] = -Rg * 1000.0f;
	cb->screen[0] = (float)w; cb->screen[1] = (float)h; cb->screen[2] = (float)mode; cb->screen[3] = (float)(frameId % 4096);
	cb->invVP = DirInvVP(probe);   // directions only: the short-far stand-in
	cb->misc[0] = curNear; cb->misc[1] = curFar; cb->misc[2] = hdr ? 0.0f : 1.0f; cb->misc[3] = sky.whitePoint;
}

// Binds the LUTs (or the stand-ins when the atmosphere is off) on any SRB that includes atmosphere.hlsli.
void NukeDiligent::Impl::AtmoBind(IShaderResourceBinding* srb, SHADER_TYPE stage)
{
	if (!srb) return;
	EnsureVolFallbacks();
	ITextureView* white = whiteTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
	ITextureView* clear3 = clearTex3D ? clearTex3D->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE) : nullptr;
	const bool on = AtmoActive();
	auto set = [&](const char* n, IDeviceObject* o) { if (o) if (auto* v = srb->GetVariableByName(stage, n)) v->Set(o); };
	auto srv = [](ITexture* t) -> ITextureView* { return t ? t->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE) : nullptr; };
	set("g_AtTrans",   on ? srv(atmoTrans)   : white);
	set("g_AtMulti",   on ? srv(atmoMulti)   : white);
	set("g_AtSkyView", (on && atmoSkyView) ? srv(atmoSkyView) : white);
	set("g_AtAP",      (on && atmoAP) ? srv(atmoAP) : clear3);
}
ITextureView* NukeDiligent::Impl::AtmoSkyViewSRV() { return AtmoActive() && atmoSkyView ? atmoSkyView->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE) : nullptr; }
ITextureView* NukeDiligent::Impl::AtmoTransSRV()   { return AtmoActive() && atmoTrans   ? atmoTrans->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE)   : nullptr; }

// A hash of the medium: the transmittance / multi-scattering LUTs rebuild when it changes.
static uint64_t AtmoMediumHash(const NukeSky& s)
{
	const float v[] = { s.planetRadius, s.atmosphereHeight, s.atmoDensity, s.rayleighColor[0], s.rayleighColor[1], s.rayleighColor[2], s.rayleighAmount, s.rayleighHeight,
	                    s.mieAmount, s.mieAbsorption, s.mieHeight, s.mieAnisotropy, s.ozoneAmount, s.groundAlbedo[0], s.groundAlbedo[1], s.groundAlbedo[2] };
	uint64_t h = 1469598103934665603ull;
	for (float f : v) { uint32_t u; memcpy(&u, &f, 4); h ^= u; h *= 1099511628211ull; }
	return h;
}

// beginCamera / beginCubeFace, before the FrameCB: the LUTs for this camera. Sets the effective
// sky colours (the summary when it has arrived) that the FrameCB and the clouds use.
void NukeDiligent::Impl::RunAtmosphere(bool probe)
{
	memcpy(skyTopEff, sky.top, sizeof(skyTopEff)); memcpy(skyHorEff, sky.horizon, sizeof(skyHorEff)); memcpy(skyGndEff, sky.ground, sizeof(skyGndEff));
	if (sky.mode != 2 || atmoFailed) return;
	if (!atmoLutPSO)
	{
		if (!atmoBuilding.exchange(true))
			EnqueueBuild([this] { if (!BuildAtmoPipes()) { atmoFailed = true; cout << "[NukeDiligent]\tatmosphere pipeline failed to build; the sky stays procedural" << endl; } },
			             [this] { atmoBuilding = false; }, kPrioGBuffer, "Atmosphere pipeline");
		return;
	}
	auto make2 = [&](RefCntAutoPtr<ITexture>& t, const char* name, int w, int h)
	{
		if (t) return;
		TextureDesc td; td.Name = name; td.Type = RESOURCE_DIM_TEX_2D; td.Width = (Uint32)w; td.Height = (Uint32)h;
		td.MipLevels = 1; td.Format = TEX_FORMAT_RGBA16_FLOAT; td.BindFlags = BIND_SHADER_RESOURCE | BIND_UNORDERED_ACCESS; td.Usage = USAGE_DEFAULT;
		device->CreateTexture(td, nullptr, &t);
	};
	make2(atmoTrans, "Atmosphere transmittance", 256, 64);
	make2(atmoMulti, "Atmosphere multi-scatter", 32, 32);
	make2(atmoSkyView, "Atmosphere sky-view", kAtmoSkyW, kAtmoSkyH);
	make2(atmoDummy, "Atmosphere dummy out", 8, 8);   // the summary pass's unused 2D output (its inputs must not double as outputs)
	if (!atmoAP)
	{
		TextureDesc td; td.Name = "Atmosphere aerial perspective"; td.Type = RESOURCE_DIM_TEX_3D; td.Width = td.Height = (Uint32)kAtmoAP; td.Depth = (Uint32)kAtmoAP;
		td.MipLevels = 1; td.Format = TEX_FORMAT_RGBA16_FLOAT; td.BindFlags = BIND_SHADER_RESOURCE | BIND_UNORDERED_ACCESS; td.Usage = USAGE_DEFAULT;
		device->CreateTexture(td, nullptr, &atmoAP);
	}
	if (!atmoSummaryBuf)
	{
		BufferDesc bd; bd.Name = "Atmosphere summary"; bd.Size = 3 * 16; bd.Usage = USAGE_DEFAULT; bd.BindFlags = BIND_UNORDERED_ACCESS;
		bd.Mode = BUFFER_MODE_STRUCTURED; bd.ElementByteStride = 16;
		device->CreateBuffer(bd, nullptr, &atmoSummaryBuf);
		for (auto& r : atmoRing)
		{
			BufferDesc sd; sd.Name = "Atmosphere summary readback"; sd.Size = 3 * 16; sd.Usage = USAGE_STAGING; sd.BindFlags = BIND_NONE; sd.CPUAccessFlags = CPU_ACCESS_READ;
			device->CreateBuffer(sd, nullptr, &r.staging); r.pending = -1;
		}
	}
	if (!atmoTrans || !atmoMulti || !atmoSkyView || !atmoAP || !atmoDummy || !atmoSummaryBuf) { atmoFailed = true; return; }
	auto uav = [](ITexture* t) { return t->GetDefaultView(TEXTURE_VIEW_UNORDERED_ACCESS); };
	auto set = [&](const char* n, IDeviceObject* o) { if (auto* v = atmoLutSRB->GetVariableByName(SHADER_TYPE_COMPUTE, n)) v->Set(o); };
	context->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE);
	ITextureView* white = whiteTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
	auto dispatch = [&](int mode, int w, int h, ITexture* out2, ITexture* out3, ITextureView* trans, ITextureView* multi, ITextureView* skyv)
	{
		FillAtmoCB(mode, w, h, probe);
		set("g_Out2D", uav(out2)); set("g_Out3D", uav(out3)); set("g_OutSummary", atmoSummaryBuf->GetDefaultView(BUFFER_VIEW_UNORDERED_ACCESS));
		set("g_AtTrans", trans); set("g_AtMulti", multi); set("g_AtSkyView", skyv);
		context->SetPipelineState(atmoLutPSO);
		context->CommitShaderResources(atmoLutSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		context->DispatchCompute(DispatchComputeAttribs((Uint32)(w + 7) / 8, (Uint32)(h + 7) / 8, 1));
	};
	auto srv = [](ITexture* t) { return t->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE); };
	// the medium's LUTs: when the medium changes (the multi-scatter one reads the transmittance)
	const uint64_t mh = AtmoMediumHash(sky);
	if (mh != atmoMediumHash)
	{
		atmoMediumHash = mh;
		dispatch(0, 256, 64, atmoTrans, atmoAP, white, white, white);
		dispatch(1, 32, 32, atmoMulti, atmoAP, srv(atmoTrans), white, white);
	}
	// this camera: the sky-view and the aerial perspective, then the summary off the sky-view
	dispatch(2, kAtmoSkyW, kAtmoSkyH, atmoSkyView, atmoAP, srv(atmoTrans), srv(atmoMulti), white);
	dispatch(3, kAtmoAP, kAtmoAP, atmoSkyView, atmoAP, srv(atmoTrans), srv(atmoMulti), white);
	if (!probe)
	{
		dispatch(4, 1, 1, atmoDummy, atmoAP, srv(atmoTrans), srv(atmoMulti), srv(atmoSkyView));
		// the summary readback ring: copy now, map a few frames on (DO_NOT_WAIT: a slot not ready waits)
		AtmoRing& r = atmoRing[atmoRingHead]; atmoRingHead = (atmoRingHead + 1) % 3;
		if (r.staging)
		{
			context->CopyBuffer(atmoSummaryBuf, 0, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, r.staging, 0, 3 * 16, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
			r.pending = 2;
		}
		for (AtmoRing& q : atmoRing)
		{
			if (q.pending < 0) continue;
			if (q.pending > 0) { --q.pending; continue; }
			void* p = nullptr;
			context->MapBuffer(q.staging, MAP_READ, MAP_FLAG_DO_NOT_WAIT, p);
			if (!p) { context->UnmapBuffer(q.staging, MAP_READ); continue; }   // D3D11: still drawing; release the booking
			const float* f = (const float*)p;
			bool finite = true;
			for (int i = 0; i < 12; ++i) if (!std::isfinite(f[i])) finite = false;
			if (finite) { memcpy(atmoSummary, f, sizeof(atmoSummary)); atmoSummaryValid = true; }
			context->UnmapBuffer(q.staging, MAP_READ);
			q.pending = -1;
		}
	}
	if (atmoSummaryValid)
	{
		for (int k = 0; k < 3; ++k) { skyTopEff[k] = atmoSummary[0][k]; skyHorEff[k] = atmoSummary[1][k]; skyGndEff[k] = atmoSummary[2][k]; }
	}
}

// endCamera, before the clouds / fog: aerial perspective on the geometry (own texture), or the input.
ITextureView* NukeDiligent::Impl::ApplyAtmosphere(ITextureView* sceneSRV, int w, int h)
{
	if (!AtmoActive() || !atmoApplyPipe.pso || !sceneSRV || !gbufActive || !gbufDepthSRV || !atmoAP || debugView != 0 || w <= 0 || h <= 0) return sceneSRV;   // this camera's own prepass depth only
	if (sky.aerialStrength <= 0.0f) return sceneSRV;
	if (!atmoOut || atmoW != w || atmoH != h)
	{
		Trash(atmoOut); atmoOut.Release();
		TextureDesc td; td.Name = "Atmosphere composite"; td.Type = RESOURCE_DIM_TEX_2D; td.Width = (Uint32)w; td.Height = (Uint32)h;
		td.MipLevels = 1; td.Format = HDR_FMT; td.BindFlags = BIND_SHADER_RESOURCE | BIND_RENDER_TARGET; td.Usage = USAGE_DEFAULT;
		device->CreateTexture(td, nullptr, &atmoOut);
		atmoW = w; atmoH = h;
		if (!atmoOut) return sceneSRV;
	}
	FillAtmoCB(0, w, h, false);
	if (atmoApplyPipe.srcVar)   atmoApplyPipe.srcVar->Set(sceneSRV);
	if (atmoApplyPipe.depthVar) atmoApplyPipe.depthVar->Set(gbufDepthSRV);
	if (atmoApplyPipe.volVar)   atmoApplyPipe.volVar->Set(atmoAP->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
	ITextureView* rtv = atmoOut->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET);
	context->SetRenderTargets(1, &rtv, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	Viewport vp; vp.TopLeftX = 0; vp.TopLeftY = 0; vp.Width = (float)w; vp.Height = (float)h; vp.MinDepth = 0; vp.MaxDepth = 1;
	context->SetViewports(1, &vp, w, h);
	context->SetPipelineState(atmoApplyPipe.pso);
	context->CommitShaderResources(atmoApplyPipe.srb, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	DrawAttribs da{3, DRAW_FLAG_VERIFY_STATES};
	context->Draw(da);
	context->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE);
	return atmoOut->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
}
