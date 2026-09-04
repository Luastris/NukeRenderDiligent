#include "NukeDiligentImpl.h"
#include <algorithm>
#include <cmath>

using namespace Diligent;
using namespace std;

// Dynamic diffuse GI (DDGI). Every enabled GIVolume is a grid of probes; each probe owns an
// 8x8 irradiance tile and a 16x16 visibility tile (+1 texel borders) in two shared atlases.
// Update = probe rays (compute + inline RayQuery on ray-tracing devices; elsewhere the World
// captures probe cube faces and ddgi_cube.cs turns them into the same ray set) -> ddgi_update.cs
// folds the rays into the atlases with hysteresis. The world PS reads the atlases (GICB).
namespace {
struct GIVolumeGPU { float origin[4]; float spacing[4]; int counts[4]; int atlas[4]; float bias[4]; int scroll[4]; };   // = ddgi.hlsli
struct GICBData    { GIVolumeGPU vol[8]; int count[4]; float atlasInv[4]; };
struct GIPassData  { int pass[4]; float rot[4]; float misc[4]; };
struct GIProbeData { float4x4 viewProj; float draw[4]; };
constexpr int kIrrT = 8, kVisT = 16, kIrrP = kIrrT + 2, kVisP = kVisT + 2;
constexpr int kCaptureMin = 4, kCaptureMax = 32, kCaptureRes = 32;   // probes captured per frame: a full pass in <= 128 frames

// A fresh random rotation per update so the fixed Fibonacci set covers the sphere over time.
void RandomQuat(uint64_t seed, float q[4])
{
	auto rnd = [&]() { seed = seed * 6364136223846793005ull + 1442695040888963407ull; return (float)((seed >> 33) & 0xFFFFFF) / 16777216.0f; };
	float u1 = rnd(), u2 = rnd(), u3 = rnd();
	float a = sqrtf(1.0f - u1), b = sqrtf(u1);
	q[0] = a * sinf(6.2831853f * u2); q[1] = a * cosf(6.2831853f * u2);
	q[2] = b * sinf(6.2831853f * u3); q[3] = b * cosf(6.2831853f * u3);
}
}

void NukeDiligent::setGIVolumes(const NukeGIVolumeDesc* volumes, int count)
{
	Impl* d = m_impl;
	d->giVols.clear();
	if (count > 8) count = 8;
	for (int i = 0; i < count && volumes; ++i)
	{
		Impl::GIVol v; v.desc = volumes[i];
		for (int k = 0; k < 3; ++k) v.desc.counts[k] = v.desc.counts[k] < 2 ? 2 : (v.desc.counts[k] > 64 ? 64 : v.desc.counts[k]);
		v.desc.raysPerProbe = v.desc.raysPerProbe < 16 ? 16 : (v.desc.raysPerProbe > 512 ? 512 : v.desc.raysPerProbe);
		v.probes = v.desc.counts[0] * v.desc.counts[1] * v.desc.counts[2];
		v.perRow = (int)ceil(sqrt((double)v.probes));
		v.rows   = (v.probes + v.perRow - 1) / v.perRow;
		// Scrolling: the origin lives on a lattice of whole probe steps; a move of k cells along an
		// axis keeps the tiles where they are (scroll += k) and re-seeds the k cells that entered.
		Impl::GIScroll& sc = d->giScroll[v.desc.id];
		long long cell[3];
		for (int k = 0; k < 3; ++k) cell[k] = (long long)llround(v.desc.origin[k] / std::max(v.desc.spacing[k], 1e-4f));
		if (sc.valid)
		{
			for (int k = 0; k < 3; ++k)
			{
				const int n = v.desc.counts[k];
				long long delta = cell[k] - sc.cell[k];
				if (delta == 0) continue;
				if (delta >= n || delta <= -n) { d->giResets.push_back({(int)d->giVols.size(), k, 0, n}); continue; }
				sc.scroll[k] = (int)(((sc.scroll[k] + delta) % n + n) % n);
				if (delta > 0) d->giResets.push_back({(int)d->giVols.size(), k, n - (int)delta, (int)delta});
				else           d->giResets.push_back({(int)d->giVols.size(), k, 0, (int)(-delta)});
			}
		}
		for (int k = 0; k < 3; ++k) { sc.cell[k] = cell[k]; v.scroll[k] = sc.scroll[k]; }
		sc.valid = true;
		d->giVols.push_back(v);
	}
	// Atlas layout: volumes stacked vertically, width = the widest volume. Any change of the
	// set (ids / probe counts) rebuilds both atlases and restarts their history.
	int irrW = 0, irrH = 0, visW = 0, visH = 0;
	uint64_t sig = 1469598103934665603ull;
	for (Impl::GIVol& v : d->giVols)
	{
		v.irrX = 0; v.irrY = irrH; v.visX = 0; v.visY = visH;
		irrW = std::max(irrW, v.perRow * kIrrP); irrH += v.rows * kIrrP;
		visW = std::max(visW, v.perRow * kVisP); visH += v.rows * kVisP;
		for (uint64_t x : { v.desc.id, (uint64_t)v.desc.counts[0], (uint64_t)v.desc.counts[1], (uint64_t)v.desc.counts[2] })
			sig = (sig ^ x) * 1099511628211ull;
	}
	if (d->giVols.empty())
	{
		d->giIrrSRV = d->giVisSRV = nullptr;
		return;
	}
	if (sig != d->giLayoutSig || !d->giIrrAtlas || !d->giVisAtlas)
	{
		d->Trash(d->giIrrAtlas); d->Trash(d->giVisAtlas);
		d->giIrrAtlas.Release(); d->giVisAtlas.Release();
		auto make = [&](RefCntAutoPtr<ITexture>& t, const char* name, int w, int h, TEXTURE_FORMAT fmt, int bpp)
		{
			TextureDesc td; td.Name = name; td.Type = RESOURCE_DIM_TEX_2D; td.Width = (Uint32)w; td.Height = (Uint32)h;
			td.MipLevels = 1; td.Format = fmt; td.Usage = USAGE_DEFAULT;
			td.BindFlags = BIND_SHADER_RESOURCE | BIND_UNORDERED_ACCESS;
			std::vector<uint8_t> zero((size_t)w * h * bpp, 0);   // a = 0 / dist = 0 = "never written": no hysteresis on the first update
			TextureSubResData sub; sub.pData = zero.data(); sub.Stride = (Uint64)w * bpp;
			TextureData init; init.pSubResources = &sub; init.NumSubresources = 1;
			d->device->CreateTexture(td, &init, &t);
		};
		make(d->giIrrAtlas, "DDGI irradiance", irrW, irrH, TEX_FORMAT_RGBA16_FLOAT, 8);
		make(d->giVisAtlas, "DDGI visibility", visW, visH, TEX_FORMAT_RG16_FLOAT, 4);
		d->giLayoutSig = sig; d->giIrrW = irrW; d->giIrrH = irrH; d->giVisW = visW; d->giVisH = visH;
		d->giCursor = 0; d->giResets.clear();   // fresh atlases are already zero
	}
	d->giIrrSRV = d->giIrrAtlas ? d->giIrrAtlas->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE) : nullptr;
	d->giVisSRV = d->giVisAtlas ? d->giVisAtlas->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE) : nullptr;
}

// GICB for the current volume set; dynamic buffers must be written every frame they are read.
void NukeDiligent::Impl::WriteGICB()
{
	if (!giCB || !giPassCB || !giProbeCB) return;   // created with the other world cbuffers (Pipelines)
	MapHelper<GICBData> cb(context, giCB, MAP_WRITE, MAP_FLAG_DISCARD);
	if (cb == nullptr) return;
	memset(cb, 0, sizeof(GICBData));
	int n = 0;
	for (const GIVol& v : giVols)
	{
		GIVolumeGPU& g = cb->vol[n++];
		for (int k = 0; k < 3; ++k) { g.origin[k] = v.desc.origin[k]; g.spacing[k] = v.desc.spacing[k]; g.counts[k] = v.desc.counts[k]; }
		g.origin[3] = v.desc.intensity; g.spacing[3] = v.desc.maxRayDistance; g.counts[3] = v.perRow;
		g.atlas[0] = v.irrX; g.atlas[1] = v.irrY; g.atlas[2] = v.visX; g.atlas[3] = v.visY;
		g.bias[0] = v.desc.normalBias; g.bias[1] = v.desc.viewBias; g.bias[2] = v.desc.hysteresis; g.bias[3] = (float)v.desc.raysPerProbe;
		g.scroll[0] = v.scroll[0]; g.scroll[1] = v.scroll[1]; g.scroll[2] = v.scroll[2]; g.scroll[3] = 0;
	}
	cb->count[0] = n;
	cb->atlasInv[0] = giIrrW ? 1.0f / giIrrW : 0.f; cb->atlasInv[1] = giIrrH ? 1.0f / giIrrH : 0.f;
	cb->atlasInv[2] = giVisW ? 1.0f / giVisW : 0.f; cb->atlasInv[3] = giVisH ? 1.0f / giVisH : 0.f;
}

// The three compute pipes (+ the probe debug PSO on demand). Built on the builder thread.
bool NukeDiligent::Impl::BuildGIPipes()
{
	auto sf = ShaderFactory();   // resolves ddgi.hlsli / rt_common.hlsl includes
	auto compile = [&](const char* name, const char* dbg, bool rt, RefCntAutoPtr<IShader>& out)
	{
		string src = shaderSource(name);
		if (src.empty()) return false;
		ShaderCreateInfo sci; sci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
		sci.pShaderSourceStreamFactory = sf;
		if (rt) { sci.ShaderCompiler = SHADER_COMPILER_DXC; sci.HLSLVersion = ShaderVersion{6, 5}; }
		sci.Desc = {dbg, SHADER_TYPE_COMPUTE, true}; sci.Source = src.c_str();
		CreateShaderCached(sci, &out);
		return out != nullptr;
	};
	SamplerDesc lin; lin.MinFilter = FILTER_TYPE_LINEAR; lin.MagFilter = FILTER_TYPE_LINEAR; lin.MipFilter = FILTER_TYPE_LINEAR;
	lin.AddressU = TEXTURE_ADDRESS_CLAMP; lin.AddressV = TEXTURE_ADDRESS_CLAMP; lin.AddressW = TEXTURE_ADDRESS_CLAMP;
	auto build = [&](const char* dbg, IShader* cs, const std::vector<const char*>& samplers, RefCntAutoPtr<IPipelineState>& pso, RefCntAutoPtr<IShaderResourceBinding>& srb)
	{
		ComputePipelineStateCreateInfo ci; ci.PSODesc.Name = dbg;
		ShaderResourceVariableDesc vars[] = {
			{SHADER_TYPE_COMPUTE, "GICB",     SHADER_RESOURCE_VARIABLE_TYPE_STATIC},
			{SHADER_TYPE_COMPUTE, "GIPassCB", SHADER_RESOURCE_VARIABLE_TYPE_STATIC},
			{SHADER_TYPE_COMPUTE, "FrameCB",  SHADER_RESOURCE_VARIABLE_TYPE_STATIC},
		};
		std::vector<ImmutableSamplerDesc> imms;
		for (const char* s : samplers) imms.push_back({SHADER_TYPE_COMPUTE, s, lin});
		ci.PSODesc.ResourceLayout.Variables = vars; ci.PSODesc.ResourceLayout.NumVariables = 3;
		ci.PSODesc.ResourceLayout.DefaultVariableType = SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC;
		ci.PSODesc.ResourceLayout.ImmutableSamplers = imms.data(); ci.PSODesc.ResourceLayout.NumImmutableSamplers = (Uint32)imms.size();
		ci.pCS = cs;
		CreateComputePipelineStateCached(ci, &pso);
		if (!pso) return false;
		if (auto* v = pso->GetStaticVariableByName(SHADER_TYPE_COMPUTE, "GICB"))     v->Set(giCB);
		if (auto* v = pso->GetStaticVariableByName(SHADER_TYPE_COMPUTE, "GIPassCB")) v->Set(giPassCB);
		if (auto* v = pso->GetStaticVariableByName(SHADER_TYPE_COMPUTE, "FrameCB"))  v->Set(worldFrameCB);
		pso->CreateShaderResourceBinding(&srb, true);
		return srb != nullptr;
	};
	RefCntAutoPtr<IShader> csU, csC, csT;
	if (!compile("ddgi_update.cs", "DDGI update CS", false, csU)) return false;
	if (!compile("ddgi_cube.cs",   "DDGI cube CS",   false, csC)) return false;
	RefCntAutoPtr<IPipelineState> pU, pC, pT; RefCntAutoPtr<IShaderResourceBinding> sU, sC, sT;
	if (!build("DDGI update PSO", csU, {}, pU, sU)) return false;
	if (!build("DDGI cube PSO",   csC, {"g_CubeColor"}, pC, sC)) return false;
	if (rtSupported)
	{
		if (!compile("ddgi_trace.cs", "DDGI trace CS", true, csT)) return false;
		if (!build("DDGI trace PSO", csT, {"g_Probe", "g_MatTex", "g_GIIrr"}, pT, sT)) return false;
	}
	giUpdatePSO = pU; giUpdateSRB = sU; giCubePSO = pC; giCubeSRB = sC; giTracePSO = pT; giTraceSRB = sT;
	return true;
}

bool NukeDiligent::Impl::EnsureGIPipes()
{
	WriteGICB();   // also creates the CBs the static bindings need
	if (!giCB) return false;
	if (giFailed || giBuilding) return false;
	if (!giUpdatePSO || !giCubePSO || (rtSupported && !giTracePSO))
	{
		if (!giBuilding.exchange(true))
			EnqueueBuild([this] { if (!BuildGIPipes()) { giFailed = true; cout << "[NukeDiligent]\tDDGI pipelines failed to build; dynamic GI stays off" << endl; } },
			             [this] { giBuilding = false; }, kPrioRT, "DDGI pipelines");
		return false;
	}
	return true;
}

void NukeDiligent::Impl::EnsureGIRays(uint32_t count)
{
	if (giRayBuf && giRayCap >= count) return;
	Trash(giRayBuf); giRayBuf.Release();
	giRayCap = std::max(count, 4096u);
	BufferDesc bd; bd.Name = "DDGI rays"; bd.Size = (Uint64)giRayCap * 16; bd.Usage = USAGE_DEFAULT;
	bd.BindFlags = BIND_SHADER_RESOURCE | BIND_UNORDERED_ACCESS; bd.Mode = BUFFER_MODE_STRUCTURED; bd.ElementByteStride = 16;
	device->CreateBuffer(bd, nullptr, &giRayBuf);
}

// Zero the tiles of the cells that scrolled into range (queued by setGIVolumes).
void NukeDiligent::Impl::ApplyGIResets()
{
	if (giResets.empty() || !giUpdatePSO || !giIrrAtlas) return;
	for (const GIReset& r : giResets)
	{
		if (r.vol >= (int)giVols.size()) continue;
		const GIVol& v = giVols[r.vol];
		{
			MapHelper<GIPassData> pc(context, giPassCB, MAP_WRITE, MAP_FLAG_DISCARD);
			pc->pass[0] = r.vol; pc->pass[1] = r.axis; pc->pass[2] = r.first; pc->pass[3] = 4;
			memset(pc->rot, 0, sizeof(pc->rot)); pc->rot[3] = 1.0f;
			pc->misc[0] = v.desc.maxRayDistance; pc->misc[1] = 0; pc->misc[2] = 0; pc->misc[3] = (float)r.count;
		}
		auto set = [&](const char* n, IDeviceObject* o) { if (auto* s = giUpdateSRB->GetVariableByName(SHADER_TYPE_COMPUTE, n)) s->Set(o); };
		if (giRayBuf) set("g_RayData", giRayBuf->GetDefaultView(BUFFER_VIEW_SHADER_RESOURCE));
		set("g_IrrAtlas", giIrrAtlas->GetDefaultView(TEXTURE_VIEW_UNORDERED_ACCESS));
		set("g_VisAtlas", giVisAtlas->GetDefaultView(TEXTURE_VIEW_UNORDERED_ACCESS));
		context->SetPipelineState(giUpdatePSO);
		context->CommitShaderResources(giUpdateSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		DispatchComputeAttribs da(((Uint32)v.probes + 63) / 64, 1, 1);
		context->DispatchCompute(da);
	}
	StateTransitionDesc b[2] = {
		{giIrrAtlas, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_UNORDERED_ACCESS, STATE_TRANSITION_FLAG_UPDATE_STATE},
		{giVisAtlas, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_UNORDERED_ACCESS, STATE_TRANSITION_FLAG_UPDATE_STATE} };
	context->TransitionResourceStates(2, b);
	giResets.clear();
}

// Rays of [first, first + count) probes of volume vi are in giRayBuf: fold them into the atlases.
void NukeDiligent::Impl::GIUpdateProbes(int vi, int first, int count, const float rot[4])
{
	const GIVol& v = giVols[vi];
	auto pass = [&](int mode, int threadsPerProbe)
	{
		{
			MapHelper<GIPassData> pc(context, giPassCB, MAP_WRITE, MAP_FLAG_DISCARD);
			pc->pass[0] = vi; pc->pass[1] = first; pc->pass[2] = v.desc.raysPerProbe; pc->pass[3] = mode;
			memcpy(pc->rot, rot, sizeof(float) * 4);
			pc->misc[0] = v.desc.maxRayDistance; pc->misc[1] = v.desc.hysteresis; pc->misc[2] = (float)(giFrame % 4096); pc->misc[3] = (float)count;
		}
		auto set = [&](const char* n, IDeviceObject* o) { if (auto* s = giUpdateSRB->GetVariableByName(SHADER_TYPE_COMPUTE, n)) s->Set(o); };
		set("g_RayData",  giRayBuf->GetDefaultView(BUFFER_VIEW_SHADER_RESOURCE));
		set("g_IrrAtlas", giIrrAtlas->GetDefaultView(TEXTURE_VIEW_UNORDERED_ACCESS));
		set("g_VisAtlas", giVisAtlas->GetDefaultView(TEXTURE_VIEW_UNORDERED_ACCESS));
		context->SetPipelineState(giUpdatePSO);
		context->CommitShaderResources(giUpdateSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		DispatchComputeAttribs da(((Uint32)count * threadsPerProbe + 63) / 64, 1, 1);
		context->DispatchCompute(da);
		StateTransitionDesc b[2] = {
			{giIrrAtlas, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_UNORDERED_ACCESS, STATE_TRANSITION_FLAG_UPDATE_STATE},
			{giVisAtlas, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_UNORDERED_ACCESS, STATE_TRANSITION_FLAG_UPDATE_STATE} };
		context->TransitionResourceStates(2, b);   // texels before borders, borders before the next reader
	};
	pass(0, kIrrT * kIrrT);
	pass(1, kVisT * kVisT);
	pass(2, kIrrP * kIrrP);
	pass(3, kVisP * kVisP);
}

// Ray-tracing devices: every probe of every volume, every frame.
void NukeDiligent::updateGIVolumes()
{
	Impl* d = m_impl;
	if (d->giVols.empty() || !d->EnsureGIPipes()) return;
	++d->giFrame;
	if (!d->rtSupported || !d->rtSceneReady || !d->tlas || !d->giTracePSO) return;
	d->GpuPass("gi");
	d->context->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE);
	d->WriteGICB();
	d->WriteFrameCB(float3(d->curCamPos[0], d->curCamPos[1], d->curCamPos[2]));   // lights + sky for the hit shading (dynamic CB: this frame)
	d->ApplyGIResets();
	for (int vi = 0; vi < (int)d->giVols.size(); ++vi)
	{
		const Impl::GIVol& v = d->giVols[vi];
		const uint32_t rays = (uint32_t)v.desc.raysPerProbe, total = (uint32_t)v.probes * rays;
		d->EnsureGIRays(total);
		if (!d->giRayBuf) continue;
		float rot[4]; RandomQuat((uint64_t)d->giFrame * 7919ull + v.desc.id * 104729ull, rot);
		{
			MapHelper<GIPassData> pc(d->context, d->giPassCB, MAP_WRITE, MAP_FLAG_DISCARD);
			pc->pass[0] = vi; pc->pass[1] = 0; pc->pass[2] = (int)rays; pc->pass[3] = v.probes;
			memcpy(pc->rot, rot, sizeof(float) * 4);
			pc->misc[0] = v.desc.maxRayDistance; pc->misc[1] = v.desc.hysteresis; pc->misc[2] = (float)(d->giFrame % 4096); pc->misc[3] = 0;
		}
		auto set = [&](const char* n, IDeviceObject* o) { if (auto* s = d->giTraceSRB->GetVariableByName(SHADER_TYPE_COMPUTE, n)) s->Set(o); };
		ITextureView* white = d->whiteTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
		set("g_TLAS",      (IDeviceObject*)d->tlas.RawPtr());
		set("g_Probe",     (d->probeActive && d->probeCubeSRV) ? d->probeCubeSRV : d->fallbackCubeSRV);
		set("g_AllNrm",    d->rtNrmSRV);
		set("g_AllUV",     d->rtUVSRV ? d->rtUVSRV : d->rtNrmSRV);
		set("g_AllPos",    d->rtPosSRV ? d->rtPosSRV : d->rtNrmSRV);
		set("g_Instances", d->rtInstSRV);
		set("g_MatBytes",  d->rtMatSRV ? d->rtMatSRV : d->rtInstSRV);
		set("g_DynCol",    d->rtDynColSRV ? d->rtDynColSRV : d->rtNrmSRV);
		set("g_GIIrr",     d->giIrrSRV);
		set("g_GIVis",     d->giVisSRV);
		set("g_RayData",   d->giRayBuf->GetDefaultView(BUFFER_VIEW_UNORDERED_ACCESS));
		{
			IDeviceObject* arr[Impl::kMaxMatTex];
			for (uint32_t k = 0; k < Impl::kMaxMatTex; ++k)
			{
				if (k < d->matTexSRVs.size()) { if (ITextureView* s = d->GetTexSRV(d->matTexPtr[k])) d->matTexSRVs[k] = s; arr[k] = d->matTexSRVs[k] ? d->matTexSRVs[k] : white; }
				else arr[k] = white;
			}
			if (auto* s = d->giTraceSRB->GetVariableByName(SHADER_TYPE_COMPUTE, "g_MatTex")) s->SetArray(arr, 0, Impl::kMaxMatTex);
		}
		d->context->SetPipelineState(d->giTracePSO);
		d->context->CommitShaderResources(d->giTraceSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		DispatchComputeAttribs da((total + 63) / 64, 1, 1);
		d->context->DispatchCompute(da);
		StateTransitionDesc b(d->giRayBuf, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_UNORDERED_ACCESS, STATE_TRANSITION_FLAG_UPDATE_STATE);
		d->context->TransitionResourceStates(1, &b);
		d->GIUpdateProbes(vi, 0, v.probes, rot);
	}
	d->GpuPass("scene");
}

// ---- raster fallback: the World captures kCaptureSlots probes per frame, round-robin ------------
int NukeDiligent::giCaptureBudget()
{
	Impl* d = m_impl;
	if (d->rtSupported || d->giVols.empty() || !d->EnsureGIPipes()) return 0;
	int total = 0; for (const Impl::GIVol& v : d->giVols) total += v.probes;
	const int budget = std::min(total, std::max(kCaptureMin, std::min(kCaptureMax, total / 128)));
	if ((int)d->giCaptures.size() < budget) d->giCaptures.resize(budget);
	return budget;
}

bool NukeDiligent::giCaptureBegin(int slot, int face, float pos[3], float* nearZ, float* farZ)
{
	Impl* d = m_impl;
	if (slot < 0 || slot >= (int)d->giCaptures.size() || d->giVols.empty()) return false;
	// slot -> (volume, probe) by the global cursor
	int total = 0; for (const Impl::GIVol& v : d->giVols) total += v.probes;
	if (total == 0) return false;
	int idx = (int)((d->giCursor + slot) % (uint64_t)total), vi = 0;
	while (idx >= d->giVols[vi].probes) { idx -= d->giVols[vi].probes; ++vi; }
	const Impl::GIVol& v = d->giVols[vi];
	const int nx = v.desc.counts[0], ny = v.desc.counts[1];
	const int cx = idx % nx, cy = (idx / nx) % ny, cz = idx / (nx * ny);
	pos[0] = v.desc.origin[0] + v.desc.spacing[0] * cx;
	pos[1] = v.desc.origin[1] + v.desc.spacing[1] * cy;
	pos[2] = v.desc.origin[2] + v.desc.spacing[2] * cz;
	*nearZ = 0.05f; *farZ = v.desc.maxRayDistance;
	Impl::GICapture& c = d->giCaptures[slot];
	if (!c.cube) c.cube = createReflectionCube(kCaptureRes);
	if (!c.cube) return false;
	c.vol = vi; c.probe = idx; c.valid = true;
	d->giCaptureMaxD = v.desc.maxRayDistance;   // world.ps writes distance / maxD into alpha while set
	beginCubeFace(c.cube, face, pos, *nearZ, *farZ);
	return true;
}

void NukeDiligent::giCaptureEnd(int slot, int face)
{
	Impl* d = m_impl;
	if (slot < 0 || slot >= (int)d->giCaptures.size() || !d->giCaptures[slot].cube) return;
	endCubeFace(d->giCaptures[slot].cube, face);
	d->giCaptureMaxD = 0.0f;
}

void NukeDiligent::giCaptureCommit()
{
	Impl* d = m_impl;
	if (d->giVols.empty() || !d->giCubePSO) return;
	d->GpuPass("gi");
	d->context->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE);
	d->WriteGICB();
	d->ApplyGIResets();
	int total = 0; for (const Impl::GIVol& v : d->giVols) total += v.probes;
	int used = 0;
	for (int slot = 0; slot < (int)d->giCaptures.size(); ++slot)
	{
		Impl::GICapture& c = d->giCaptures[slot];
		if (c.valid) ++used;
		if (!c.valid) continue;
		c.valid = false;
		if (c.vol >= (int)d->giVols.size()) continue;
		const Impl::GIVol& v = d->giVols[c.vol];
		auto cit = d->cubes.find(c.cube);
		if (cit == d->cubes.end() || !cit->second.srv) continue;
		const uint32_t rays = (uint32_t)v.desc.raysPerProbe;
		d->EnsureGIRays((uint32_t)v.probes * rays);
		if (!d->giRayBuf) continue;
		float rot[4]; RandomQuat((uint64_t)d->giFrame * 7919ull + (uint64_t)c.probe * 104729ull + v.desc.id, rot);
		{
			MapHelper<GIPassData> pc(d->context, d->giPassCB, MAP_WRITE, MAP_FLAG_DISCARD);
			pc->pass[0] = c.vol; pc->pass[1] = c.probe; pc->pass[2] = (int)rays; pc->pass[3] = 0;
			memcpy(pc->rot, rot, sizeof(float) * 4);
			// A probe is captured once per (total / budget) frames: apply the per-frame hysteresis
			// to that interval, or a full pass would still show only half of the light.
			const float interval = std::max(1.0f, (float)total / (float)std::max(1, (int)d->giCaptures.size()));
			pc->misc[0] = v.desc.maxRayDistance; pc->misc[1] = powf(v.desc.hysteresis, interval); pc->misc[2] = (float)(d->giFrame % 4096); pc->misc[3] = 1;
		}
		auto set = [&](const char* n, IDeviceObject* o) { if (auto* s = d->giCubeSRB->GetVariableByName(SHADER_TYPE_COMPUTE, n)) s->Set(o); };
		set("g_CubeColor", cit->second.srv);
		set("g_RayData",   d->giRayBuf->GetDefaultView(BUFFER_VIEW_UNORDERED_ACCESS));
		d->context->SetPipelineState(d->giCubePSO);
		d->context->CommitShaderResources(d->giCubeSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		DispatchComputeAttribs da((rays + 63) / 64, 1, 1);
		d->context->DispatchCompute(da);
		StateTransitionDesc b(d->giRayBuf, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_UNORDERED_ACCESS, STATE_TRANSITION_FLAG_UPDATE_STATE);
		d->context->TransitionResourceStates(1, &b);
		d->GIUpdateProbes(c.vol, c.probe, 1, rot);
	}
	d->giCursor += used;
	++d->giFrame;
	d->GpuPass("scene");
}

// Debug Probes: the probe grid as small spheres shaded by their own irradiance, drawn into the
// camera targets while its (MS) depth is still bound.
void NukeDiligent::Impl::DrawGIProbes()
{
	if (giVols.empty() || !giIrrSRV || !giCB) return;
	bool any = false; for (const GIVol& v : giVols) any = any || v.desc.debugProbes;
	if (!any) return;
	if (!giProbePSO || giProbeSamples != (int)samples || giProbeFmt != SceneFmt())
	{
		if (giProbePSO) Trash(giProbePSO);
		giProbePSO.Release(); giProbeSRB.Release();
		string vsSrc = shaderSource("giprobe.vs"), psSrc = shaderSource("giprobe.ps");
		if (vsSrc.empty() || psSrc.empty()) return;
		auto sf = ShaderFactory();
		ShaderCreateInfo sci; sci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL; sci.pShaderSourceStreamFactory = sf;
		RefCntAutoPtr<IShader> v, p;
		sci.Desc = {"GI Probe VS", SHADER_TYPE_VERTEX, true}; sci.Source = vsSrc.c_str(); CreateShaderCached(sci, &v);
		sci.Desc = {"GI Probe PS", SHADER_TYPE_PIXEL, true};  sci.Source = psSrc.c_str(); CreateShaderCached(sci, &p);
		if (!v || !p) return;
		GraphicsPipelineStateCreateInfo ci; ci.PSODesc.Name = "GI Probe PSO";
		auto& gp = ci.GraphicsPipeline;
		gp.NumRenderTargets = 1; gp.RTVFormats[0] = SceneFmt(); gp.DSVFormat = TEX_FORMAT_D32_FLOAT;
		gp.PrimitiveTopology = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		gp.RasterizerDesc.CullMode = CULL_MODE_BACK;
		gp.DepthStencilDesc.DepthEnable = True; gp.DepthStencilDesc.DepthWriteEnable = True;
		gp.SmplDesc.Count = samples;
		gp.InputLayout.NumElements = 0;
		SamplerDesc lin; lin.MinFilter = FILTER_TYPE_LINEAR; lin.MagFilter = FILTER_TYPE_LINEAR; lin.MipFilter = FILTER_TYPE_LINEAR;
		lin.AddressU = TEXTURE_ADDRESS_CLAMP; lin.AddressV = TEXTURE_ADDRESS_CLAMP;
		ShaderResourceVariableDesc vars[] = {{SHADER_TYPE_PIXEL, "g_GIIrr", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC}};
		ImmutableSamplerDesc imms[] = {{SHADER_TYPE_PIXEL, "g_GIIrr", lin}};
		ci.PSODesc.ResourceLayout.Variables = vars; ci.PSODesc.ResourceLayout.NumVariables = 1;
		ci.PSODesc.ResourceLayout.ImmutableSamplers = imms; ci.PSODesc.ResourceLayout.NumImmutableSamplers = 1;
		ci.pVS = v; ci.pPS = p;
		CreateGraphicsPipelineStateCached(ci, &giProbePSO);
		if (!giProbePSO) return;
		for (SHADER_TYPE st : { SHADER_TYPE_VERTEX, SHADER_TYPE_PIXEL })
		{
			if (auto* c = giProbePSO->GetStaticVariableByName(st, "GICB"))     c->Set(giCB);
			if (auto* c = giProbePSO->GetStaticVariableByName(st, "GIProbeCB")) c->Set(giProbeCB);
		}
		giProbePSO->CreateShaderResourceBinding(&giProbeSRB, true);
		giProbeSamples = (int)samples; giProbeFmt = SceneFmt();
	}
	for (int vi = 0; vi < (int)giVols.size(); ++vi)
	{
		const GIVol& v = giVols[vi];
		if (!v.desc.debugProbes) continue;
		{
			MapHelper<GIProbeData> pc(context, giProbeCB, MAP_WRITE, MAP_FLAG_DISCARD);
			pc->viewProj = curView * curProj;
			pc->draw[0] = (float)vi;
			pc->draw[1] = std::min({ v.desc.spacing[0], v.desc.spacing[1], v.desc.spacing[2] }) * 0.08f;
			pc->draw[2] = pc->draw[3] = 0.0f;
		}
		if (auto* s = giProbeSRB->GetVariableByName(SHADER_TYPE_PIXEL, "g_GIIrr")) s->Set(giIrrSRV);
		context->SetPipelineState(giProbePSO);
		context->CommitShaderResources(giProbeSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		DrawAttribs da; da.NumVertices = 12 * 8 * 6; da.NumInstances = (Uint32)v.probes; da.Flags = DRAW_FLAG_VERIFY_STATES;
		context->Draw(da);
	}
}
