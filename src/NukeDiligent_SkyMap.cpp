// The sky map: the whole sky, clouds included, as an equirect panorama (skymap.cs) refreshed once
// a frame after the atmosphere's LUTs. It is THE sky for everything that looks up in a direction
// off the screen - traced rays that escape the scene, the water's reflection, world.ps
// image-based lighting - so mirrors reflect the clouds and an overcast sky lights the world
// grey, not clear-sky blue: its summary (zenith / horizon / ground, read back like the
// atmosphere's) feeds the FrameCB sky colours. The clear-sky colours (skyTopEff) stay what the
// clouds themselves are lit by (no feedback).
#include "NukeDiligentImpl.h"
#include <cmath>

using namespace Diligent;

namespace {
const int kSkyMapW = 2048, kSkyMapH = 1024;   // 0.18 degrees per texel: mirrors show the clouds' edges
const int kSkyCloudRows = 8;                   // the cloud layer marches 1/8 of its rows per frame
}

void NukeDiligent::Impl::EnsureSkyMapPipe()
{
	if (skyMapPSO || skyMapFailed || skyMapBuilding.exchange(true)) return;
	EnqueueBuild([this]
	{
		ReloadScope reloadScope("skymap");
		const std::string src = shaderSource("skymap.cs");
		if (src.empty()) { skyMapFailed = true; return; }
		// The static CBs must exist at PSO creation (the clouds' and the atmosphere's are created here
		// when their own passes have not run yet; the same descs their builders use).
		if (!cloudCB)
		{
			BufferDesc d; d.Name = "CloudCB"; d.Size = kCloudCBSize; d.Usage = USAGE_DYNAMIC;
			d.BindFlags = BIND_UNIFORM_BUFFER; d.CPUAccessFlags = CPU_ACCESS_WRITE;
			device->CreateBuffer(d, nullptr, &cloudCB);
		}
		if (!atmoCB)
		{
			BufferDesc d; d.Name = "AtmoCB"; d.Size = kAtmoCBSize; d.Usage = USAGE_DYNAMIC;
			d.BindFlags = BIND_UNIFORM_BUFFER; d.CPUAccessFlags = CPU_ACCESS_WRITE;
			device->CreateBuffer(d, nullptr, &atmoCB);
		}
		if (!skyMapCB)
		{
			BufferDesc d; d.Name = "SkyMapCB"; d.Size = 6 * 16; d.Usage = USAGE_DYNAMIC;
			d.BindFlags = BIND_UNIFORM_BUFFER; d.CPUAccessFlags = CPU_ACCESS_WRITE;
			device->CreateBuffer(d, nullptr, &skyMapCB);
		}
		if (!skySumBuf)
		{
			BufferDesc bd; bd.Name = "Sky map summary"; bd.Size = 3 * 16; bd.Usage = USAGE_DEFAULT; bd.BindFlags = BIND_UNORDERED_ACCESS;
			bd.Mode = BUFFER_MODE_STRUCTURED; bd.ElementByteStride = 16;
			device->CreateBuffer(bd, nullptr, &skySumBuf);
			for (auto& r : skyRing)
			{
				BufferDesc sd; sd.Name = "Sky map summary readback"; sd.Size = 3 * 16; sd.Usage = USAGE_STAGING; sd.BindFlags = BIND_NONE; sd.CPUAccessFlags = CPU_ACCESS_READ;
				device->CreateBuffer(sd, nullptr, &r.staging); r.pending = -1;
			}
		}
		if (!cloudCB || !atmoCB || !skyMapCB || !skySumBuf) { skyMapFailed = true; return; }
		ShaderCreateInfo sci; sci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL; sci.pShaderSourceStreamFactory = ShaderFactory();
		RefCntAutoPtr<IShader> cs;
		sci.Desc = {"Sky map CS", SHADER_TYPE_COMPUTE, true}; sci.Source = src.c_str(); CreateShaderCached(sci, &cs);
		if (!cs) { skyMapFailed = true; return; }
		ComputePipelineStateCreateInfo ci; ci.PSODesc.Name = "Sky map PSO";
		ShaderResourceVariableDesc vars[] = {
			{SHADER_TYPE_COMPUTE, "SkyMapCB", SHADER_RESOURCE_VARIABLE_TYPE_STATIC},
			{SHADER_TYPE_COMPUTE, "CloudCB",  SHADER_RESOURCE_VARIABLE_TYPE_STATIC},
			{SHADER_TYPE_COMPUTE, "AtmoCB",   SHADER_RESOURCE_VARIABLE_TYPE_STATIC},
		};
		SamplerDesc wrap; wrap.MinFilter = FILTER_TYPE_LINEAR; wrap.MagFilter = FILTER_TYPE_LINEAR; wrap.MipFilter = FILTER_TYPE_LINEAR;
		wrap.AddressU = TEXTURE_ADDRESS_WRAP; wrap.AddressV = TEXTURE_ADDRESS_WRAP; wrap.AddressW = TEXTURE_ADDRESS_WRAP;
		const SamplerDesc lutS = AtmoSampler();
		ImmutableSamplerDesc imms[] = {
			{SHADER_TYPE_COMPUTE, "g_CloudBase", wrap}, {SHADER_TYPE_COMPUTE, "g_CloudDetail", wrap}, {SHADER_TYPE_COMPUTE, "g_CloudWeather", wrap},
			{SHADER_TYPE_COMPUTE, "g_AtTrans", lutS}, {SHADER_TYPE_COMPUTE, "g_AtMulti", lutS}, {SHADER_TYPE_COMPUTE, "g_AtSkyView", lutS},
			{SHADER_TYPE_COMPUTE, "g_SkyIn", wrap},
		};
		ci.PSODesc.ResourceLayout.Variables = vars; ci.PSODesc.ResourceLayout.NumVariables = 3;
		ci.PSODesc.ResourceLayout.DefaultVariableType = SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC;
		ci.PSODesc.ResourceLayout.ImmutableSamplers = imms; ci.PSODesc.ResourceLayout.NumImmutableSamplers = 7;
		ci.pCS = cs;
		RefCntAutoPtr<IPipelineState> pso;
		CreateComputePipelineStateCached(ci, &pso);
		if (!pso) { skyMapFailed = true; std::cout << "[NukeDiligent]\tsky map pipeline FAILED" << std::endl; return; }
		if (auto* v = pso->GetStaticVariableByName(SHADER_TYPE_COMPUTE, "SkyMapCB")) v->Set(skyMapCB);
		if (auto* v = pso->GetStaticVariableByName(SHADER_TYPE_COMPUTE, "CloudCB"))  v->Set(cloudCB);
		if (auto* v = pso->GetStaticVariableByName(SHADER_TYPE_COMPUTE, "AtmoCB"))   v->Set(atmoCB);
		RefCntAutoPtr<IShaderResourceBinding> srb;
		pso->CreateShaderResourceBinding(&srb, true);
		if (!srb) { skyMapFailed = true; return; }
		skyMapPSO = pso; skyMapSRB = srb;
		std::cout << "[NukeDiligent]\tsky map ready" << std::endl;
	}, [this] { skyMapBuilding = false; }, kPrioGBuffer, "sky map");
}

// Once a frame, after RunAtmosphere (the camera's LUTs are this frame's).
void NukeDiligent::Impl::RunSkyMap()
{
	skyMapSRV = nullptr;
	auto useClear = [&] { for (int k = 0; k < 3; ++k) { skyTopIbl[k] = skyTopEff[k]; skyHorIbl[k] = skyHorEff[k]; skyGndIbl[k] = skyGndEff[k]; } };
	if (sky.mode < 1) { skySummaryValid = false; useClear(); return; }
	EnsureSkyMapPipe();
	if (skySummaryValid) { for (int k = 0; k < 3; ++k) { skyTopIbl[k] = skySummary[0][k]; skyHorIbl[k] = skySummary[1][k]; skyGndIbl[k] = skySummary[2][k]; } }
	else useClear();
	if (!skyMapPSO || !skyMapSRB || skyMapFailed) return;
	if (!skyMapTex)
	{
		TextureDesc td; td.Name = "Sky map"; td.Type = RESOURCE_DIM_TEX_2D; td.Width = (Uint32)kSkyMapW; td.Height = (Uint32)kSkyMapH;
		td.MipLevels = 0; td.Format = TEX_FORMAT_RGBA16_FLOAT; td.Usage = USAGE_DEFAULT;
		td.BindFlags = BIND_SHADER_RESOURCE | BIND_UNORDERED_ACCESS | BIND_RENDER_TARGET;
		td.MiscFlags = MISC_TEXTURE_FLAG_GENERATE_MIPS;
		device->CreateTexture(td, nullptr, &skyMapTex);
		TextureDesc cd; cd.Name = "Sky map clouds"; cd.Type = RESOURCE_DIM_TEX_2D; cd.Width = (Uint32)kSkyMapW; cd.Height = (Uint32)kSkyMapH;
		cd.MipLevels = 1; cd.Format = TEX_FORMAT_RGBA16_FLOAT; cd.Usage = USAGE_DEFAULT;
		cd.BindFlags = BIND_SHADER_RESOURCE | BIND_UNORDERED_ACCESS;
		device->CreateTexture(cd, nullptr, &skyCloudTex);
		skyCloudValid = false;
		if (!skyMapTex || !skyCloudTex) { skyMapFailed = true; return; }
	}
	GpuPass("skymap");
	const bool physical = AtmoActive() && atmoSkyView;
	const bool cloudsOn = CloudsState() == 2 && cloudBase && cloudDetail && cloudWeather;
	context->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE);
	auto srv = [](ITexture* t) { return t->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE); };
	auto set = [&](const char* n, IDeviceObject* o) { if (o) if (auto* v = skyMapSRB->GetVariableByName(SHADER_TYPE_COMPUTE, n)) v->Set(o); };
	ITextureView* white = whiteTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
	FillAtmoCB(0, 0, 0, false);
	AtmoBind(skyMapSRB, SHADER_TYPE_COMPUTE);
	// CloudCB is a dynamic buffer bound statically: it must be mapped every pass that binds it, clouds or not.
	FillCloudCB(kSkyMapW, kSkyMapH, 1, 1, 0.0f, nullptr, 0.0f, 0.0f, 1.0f, true);
	if (cloudsOn) { set("g_CloudBase", srv(cloudBase)); set("g_CloudDetail", srv(cloudDetail)); set("g_CloudWeather", srv(cloudWeather)); }
	else
	{   // the noise volumes are 3D: the stand-ins must be too (the shader never reads them with clouds off)
		EnsureVolFallbacks();
		ITextureView* clear3 = clearTex3D ? clearTex3D->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE) : nullptr;
		set("g_CloudBase", clear3); set("g_CloudDetail", clear3); set("g_CloudWeather", white);
	}
	auto fill = [&](float mode)
	{
		MapHelper<float> cb(context, skyMapCB, MAP_WRITE, MAP_FLAG_DISCARD);
		if (cb == nullptr) return false;
		for (int k = 0; k < 3; ++k) { cb[k] = skyTopEff[k]; cb[4 + k] = skyHorEff[k]; cb[8 + k] = skyGndEff[k]; }
		cb[3] = physical ? 1.0f : sky.skyIntensity;
		cb[7] = (rtWaterOcc[1] > 0.5f && rtWaterInfinite) ? 1.0f : 0.0f; cb[11] = rtWaterOcc[0];   // a boundless ocean + its level
		cb[12] = mode; cb[13] = physical ? 1.0f : 0.0f; cb[14] = cloudsOn ? 1.0f : 0.0f; cb[15] = (float)(frameId % (uint64_t)kSkyCloudRows);
		cb[16] = curCamPos[0]; cb[17] = curCamPos[1]; cb[18] = curCamPos[2]; cb[19] = skyCloudValid ? 1.0f : 0.0f;
		cb[20] = (float)kSkyMapW; cb[21] = (float)kSkyMapH; cb[22] = (float)kSkyCloudRows; cb[23] = (float)std::log2((double)kSkyMapW / 64.0);
		return true;
	};
	// the cloud layer: this frame's rows (the layer converges over kSkyCloudRows frames)
	if (!fill(0.0f)) return;
	set("g_SkyOut", skyCloudTex->GetDefaultView(TEXTURE_VIEW_UNORDERED_ACCESS));
	set("g_CloudIn", white);
	set("g_SkyIn", white);
	set("g_OutSummary", skySumBuf->GetDefaultView(BUFFER_VIEW_UNORDERED_ACCESS));
	context->SetPipelineState(skyMapPSO);
	context->CommitShaderResources(skyMapSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	context->DispatchCompute(DispatchComputeAttribs((Uint32)kSkyMapW / 8, (Uint32)(kSkyMapH / kSkyCloudRows + 7) / 8, 1));
	skyCloudValid = true;
	// the map: the sky under the layer, every texel
	if (!fill(2.0f)) return;
	set("g_SkyOut", skyMapTex->GetDefaultView(TEXTURE_VIEW_UNORDERED_ACCESS));
	set("g_CloudIn", srv(skyCloudTex));
	context->SetPipelineState(skyMapPSO);
	context->CommitShaderResources(skyMapSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	context->DispatchCompute(DispatchComputeAttribs((Uint32)kSkyMapW / 8, (Uint32)kSkyMapH / 8, 1));
	{
		StateTransitionDesc tr(skyMapTex, RESOURCE_STATE_UNKNOWN, RESOURCE_STATE_SHADER_RESOURCE, STATE_TRANSITION_FLAG_UPDATE_STATE);
		context->TransitionResourceStates(1, &tr);
	}
	context->GenerateMips(srv(skyMapTex));
	set("g_SkyOut", skyCloudTex->GetDefaultView(TEXTURE_VIEW_UNORDERED_ACCESS));   // the summary must not read and write the map
	// the summary off the finished map, read back a few frames on (the atmosphere's ring pattern)
	if (fill(1.0f))
	{
		set("g_SkyIn", srv(skyMapTex));
		context->SetPipelineState(skyMapPSO);
		context->CommitShaderResources(skyMapSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		context->DispatchCompute(DispatchComputeAttribs(1, 1, 1));
		AtmoRing& r = skyRing[skyRingHead]; skyRingHead = (skyRingHead + 1) % 3;
		if (r.staging)
		{
			context->CopyBuffer(skySumBuf, 0, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, r.staging, 0, 3 * 16, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
			r.pending = 2;
		}
		for (AtmoRing& q : skyRing)
		{
			if (q.pending < 0) continue;
			if (q.pending > 0) { --q.pending; continue; }
			void* p = nullptr;
			context->MapBuffer(q.staging, MAP_READ, MAP_FLAG_DO_NOT_WAIT, p);
			if (!p) { context->UnmapBuffer(q.staging, MAP_READ); continue; }
			const float* f = (const float*)p;
			bool finite = true;
			for (int i = 0; i < 12; ++i) if (!std::isfinite(f[i])) finite = false;
			if (finite) { memcpy(skySummary, f, sizeof(skySummary)); skySummaryValid = true; }
			context->UnmapBuffer(q.staging, MAP_READ);
			q.pending = -1;
		}
	}
	skyMapSRV = srv(skyMapTex);
}
