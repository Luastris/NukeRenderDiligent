#include "NukeDiligentImpl.h"
#include <algorithm>
#include <cmath>
#include <chrono>

using namespace Diligent;
using namespace std;

// Froxel volumetric lighting / fog. Per camera a view-aligned grid (screen tiles x exponential
// depth slices) holds the medium and its in-scatter: vol_inject.cs lights every froxel from the
// frame's lights (sun through the shadow map, or an inline shadow ray on ray-tracing devices)
// and the DDGI/sky ambient, blended with last frame's grid; vol_integrate.cs walks each column
// front to back; vol_apply.ps composites the column at every pixel's depth before the post chain.
namespace {
struct VolCBData { float4x4 view, proj, invViewProj, prevViewProj; float grid[4], range[4], cam[4], medium[4], albedo[4], misc[4], jitter[4], screen[4]; };
struct FogVolGPU  { float posShape[4], extDensity[4], rot[4], albedoFall[4], emisNoise[4], noiseMisc[4], fluidInfo[4]; };   // = vol_inject.cs FogVol
struct FluidCBData { float res[4], box[4], up[4], pos[4], wind[4], params[4], params2[4], counts[4], dispPos[8][4], dispVel[8][4], forcePos[8][4], forceDir[8][4], forceMisc[8][4], forceDent[8][4], forceDent2[8][4], misc[4], splat[4]; };
struct FogVolCBData { float wind[4]; int count[4]; FogVolGPU vols[32]; };
static_assert(sizeof(VolCBData) == NukeDiligent::Impl::kVolCBSize, "VolCB size");
static_assert(sizeof(FogVolCBData) == NukeDiligent::Impl::kFogVolCBSize, "FogVolCB size");
static_assert(sizeof(FluidCBData) == NukeDiligent::Impl::kFluidCBSize, "FluidCB size");
constexpr int kTile[4]   = { 16, 16, 8, 6 };   // Off, Low, Medium, High: screen pixels per froxel
constexpr int kSlices[4] = { 32, 32, 64, 96 };
float Halton(int i, int b) { float f = 1.0f, r = 0.0f; while (i > 0) { f /= (float)b; r += f * (float)(i % b); i /= b; } return r; }
}

void NukeDiligent::setVolumetrics(const NukeVolumetricsDesc& desc) { m_impl->vol = desc; }

void NukeDiligent::setFogDisplacers(const NukeFogDisplacerDesc* displacers, int count)
{
	m_impl->fogDisplacers.clear();
	if (displacers && count > 0) m_impl->fogDisplacers.assign(displacers, displacers + count);
}

void NukeDiligent::setFogVolumes(const NukeFogVolumeDesc* volumes, int count)
{
	m_impl->fogVols.clear();
	if (!volumes || count <= 0) return;
	if (count > Impl::kFogVolMax) count = Impl::kFogVolMax;
	m_impl->fogVols.assign(volumes, volumes + count);
}

bool NukeDiligent::Impl::BuildVolPipes()
{
	const string csI = shaderSource("vol_inject.cs"), csT = shaderSource("vol_temporal.cs"), csN = shaderSource("vol_integrate.cs");
	const string vs = shaderSource("post.vs"), psA = shaderSource("vol_apply.ps");
	if (csI.empty() || csN.empty() || vs.empty() || psA.empty()) return false;
	auto sf = ShaderFactory();   // vol.hlsli / ddgi.hlsli / rt_common.hlsl includes
	ShaderMacro rtMacro[] = {{"RT_ENABLED", "1"}};
	auto compile = [&](const string& src, const char* dbg, SHADER_TYPE type, bool rt, RefCntAutoPtr<IShader>& out)
	{
		ShaderCreateInfo sci; sci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL; sci.pShaderSourceStreamFactory = sf;
		if (rt) { sci.ShaderCompiler = SHADER_COMPILER_DXC; sci.HLSLVersion = ShaderVersion{6, 5}; sci.Macros = ShaderMacroArray{rtMacro, 1}; }
		sci.Desc = {dbg, type, true}; sci.Source = src.c_str();
		CreateShaderCached(sci, &out);
		return out != nullptr;
	};
	SamplerDesc lin; lin.MinFilter = FILTER_TYPE_LINEAR; lin.MagFilter = FILTER_TYPE_LINEAR; lin.MipFilter = FILTER_TYPE_LINEAR;
	lin.AddressU = TEXTURE_ADDRESS_CLAMP; lin.AddressV = TEXTURE_ADDRESS_CLAMP; lin.AddressW = TEXTURE_ADDRESS_CLAMP;
	SamplerDesc pt = lin; pt.MinFilter = FILTER_TYPE_POINT; pt.MagFilter = FILTER_TYPE_POINT; pt.MipFilter = FILTER_TYPE_POINT;
	auto buildCS = [&](const char* dbg, IShader* cs, const vector<const char*>& samplers, RefCntAutoPtr<IPipelineState>& pso, RefCntAutoPtr<IShaderResourceBinding>& srb)
	{
		ComputePipelineStateCreateInfo ci; ci.PSODesc.Name = dbg;
		ShaderResourceVariableDesc vars[] = {
			{SHADER_TYPE_COMPUTE, "VolCB",    SHADER_RESOURCE_VARIABLE_TYPE_STATIC},
			{SHADER_TYPE_COMPUTE, "FrameCB",  SHADER_RESOURCE_VARIABLE_TYPE_STATIC},
			{SHADER_TYPE_COMPUTE, "GICB",     SHADER_RESOURCE_VARIABLE_TYPE_STATIC},
			{SHADER_TYPE_COMPUTE, "FogVolCB", SHADER_RESOURCE_VARIABLE_TYPE_STATIC},
		};
		vector<ImmutableSamplerDesc> imms;
		for (const char* s : samplers) imms.push_back({SHADER_TYPE_COMPUTE, s, lin});
		ci.PSODesc.ResourceLayout.Variables = vars; ci.PSODesc.ResourceLayout.NumVariables = 4;
		ci.PSODesc.ResourceLayout.DefaultVariableType = SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC;
		ci.PSODesc.ResourceLayout.ImmutableSamplers = imms.data(); ci.PSODesc.ResourceLayout.NumImmutableSamplers = (Uint32)imms.size();
		ci.pCS = cs;
		CreateComputePipelineStateCached(ci, &pso);
		if (!pso) return false;
		if (auto* v = pso->GetStaticVariableByName(SHADER_TYPE_COMPUTE, "VolCB"))   v->Set(volCB);
		if (auto* v = pso->GetStaticVariableByName(SHADER_TYPE_COMPUTE, "FrameCB")) v->Set(worldFrameCB);
		if (auto* v = pso->GetStaticVariableByName(SHADER_TYPE_COMPUTE, "GICB"))    v->Set(giCB);
		if (auto* v = pso->GetStaticVariableByName(SHADER_TYPE_COMPUTE, "FogVolCB")) v->Set(fogVolCB);
		pso->CreateShaderResourceBinding(&srb, true);
		return srb != nullptr;
	};
	RefCntAutoPtr<IShader> sI, sT, sN, sV, sP;
	if (!compile(csI, "Vol inject CS", SHADER_TYPE_COMPUTE, rtSupported, sI)) return false;
	if (!compile(csT, "Vol temporal CS", SHADER_TYPE_COMPUTE, false, sT)) return false;
	if (!compile(csN, "Vol integrate CS", SHADER_TYPE_COMPUTE, false, sN)) return false;
	if (!compile(vs, "Vol apply VS", SHADER_TYPE_VERTEX, false, sV)) return false;
	if (!compile(psA, "Vol apply PS", SHADER_TYPE_PIXEL, false, sP)) return false;
	RefCntAutoPtr<IPipelineState> pI, pT, pN; RefCntAutoPtr<IShaderResourceBinding> bI, bT, bN;
	if (!buildCS("Vol inject PSO",    sI, {"g_GIIrr", "g_Fluid0", "g_Fluid1", "g_Fluid2", "g_Fluid3"}, pI, bI)) return false;
	if (!buildCS("Vol temporal PSO",  sT, {"g_ScatPrev"}, pT, bT)) return false;
	if (!buildCS("Vol integrate PSO", sN, {}, pN, bN)) return false;
	// Composite: a fullscreen pass over the scene colour (own output texture, not the chain scratch).
	PostPipe pp;
	{
		GraphicsPipelineStateCreateInfo ci; ci.PSODesc.Name = "Vol apply PSO";
		auto& gp = ci.GraphicsPipeline;
		gp.NumRenderTargets = 1; gp.RTVFormats[0] = HDR_FMT; gp.DSVFormat = TEX_FORMAT_UNKNOWN;
		gp.PrimitiveTopology = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; gp.RasterizerDesc.CullMode = CULL_MODE_NONE;
		gp.DepthStencilDesc.DepthEnable = False; gp.InputLayout.NumElements = 0;
		ShaderResourceVariableDesc vars[] = {
			{SHADER_TYPE_PIXEL, "g_Source", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
			{SHADER_TYPE_PIXEL, "g_Depth",  SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
			{SHADER_TYPE_PIXEL, "g_Volume", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
		};
		ImmutableSamplerDesc imms[] = {{SHADER_TYPE_PIXEL, "g_Source", lin}, {SHADER_TYPE_PIXEL, "g_Depth", pt}, {SHADER_TYPE_PIXEL, "g_Volume", lin}};
		ci.PSODesc.ResourceLayout.Variables = vars; ci.PSODesc.ResourceLayout.NumVariables = 3;
		ci.PSODesc.ResourceLayout.ImmutableSamplers = imms; ci.PSODesc.ResourceLayout.NumImmutableSamplers = 3;
		ci.pVS = sV; ci.pPS = sP;
		CreateGraphicsPipelineStateCached(ci, &pp.pso);
		if (!pp.pso) return false;
		if (auto* c = pp.pso->GetStaticVariableByName(SHADER_TYPE_PIXEL, "VolCB")) c->Set(volCB);
		pp.pso->CreateShaderResourceBinding(&pp.srb, true);
		pp.srcVar   = pp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Source");
		pp.depthVar = pp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Depth");
		pp.histVar  = pp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Volume");   // the integrated grid
	}
	volInjectPSO = pI; volInjectSRB = bI; volTemporalPSO = pT; volTemporalSRB = bT; volIntegratePSO = pN; volIntegrateSRB = bN; volApplyPipe = std::move(pp);
	return true;
}

// VolCB is a dynamic buffer bound statically by the sprite PSOs: a pass without a grid still
// needs it mapped (Vulkan rejects a stale dynamic allocation), with "grid off" contents.
void NukeDiligent::Impl::TouchVolCB()
{
	if (!volCB) return;
	MapHelper<VolCBData> cb(context, volCB, MAP_WRITE, MAP_FLAG_DISCARD);
	if (cb == nullptr) return;
	memset(cb, 0, sizeof(VolCBData));
	cb->grid[2] = 1.0f; cb->range[0] = 1.0f; cb->range[1] = 2.0f; cb->range[2] = 0.693f; cb->range[3] = 1.44f;
	cb->cam[0] = curNear; cb->cam[1] = curFar; cb->screen[0] = (float)std::max(curRTW, 1); cb->screen[1] = (float)std::max(curRTH, 1);
	// FogVolCB too: the water PSOs bind it statically, and a pass without the fog grid (quality 0,
	// no prepass) never mapped it -> Vulkan "dynamic buffer not mapped" on the first water draw.
	// RunVolumetrics remaps it with the real volumes when the grid runs.
	if (fogVolCB)
	{
		MapHelper<FogVolCBData> fc(context, fogVolCB, MAP_WRITE, MAP_FLAG_DISCARD);
		if (fc != nullptr) memset(fc, 0, sizeof(FogVolCBData));
	}
}

// beginCamera (after AO / SSGI): this camera's grid off its prepass camera.
void NukeDiligent::Impl::RunVolumetrics(int w, int h)
{
	volCur = nullptr;
	if (vol.quality <= 0 || w <= 0 || h <= 0 || !gbufDepthSRV) return;
	if (vol.density <= 0.0f && vol.shaftDensity <= 0.0f && fogVols.empty()) return;   // nothing scatters: no grid, no cost
	if (!volCB)
	{
		BufferDesc d; d.Name = "VolCB"; d.Size = sizeof(VolCBData); d.Usage = USAGE_DYNAMIC;
		d.BindFlags = BIND_UNIFORM_BUFFER; d.CPUAccessFlags = CPU_ACCESS_WRITE;
		device->CreateBuffer(d, nullptr, &volCB);
		if (!volCB) return;
	}
	if (volBuilding || volFailed) return;
	if (!volInjectPSO || !volIntegratePSO || !volApplyPipe.pso)
	{
		if (!volBuilding.exchange(true))
			EnqueueBuild([this] { if (!BuildVolPipes()) { volFailed = true; cout << "[NukeDiligent]\tvolumetrics pipelines failed to build; volumetric fog stays off" << endl; } },
			             [this] { volBuilding = false; }, kPrioGBuffer, "Volumetrics pipelines");
		return;
	}
	if (rtSupported && (!rtSceneReady || !tlas)) return;   // the shadow rays need the scene TLAS
	if (!rtSupported && !shadowSRV) return;                // the sun visibility needs the shadow array

	VolState& st = volStates[curCamKey];
	st.lastUsed = frameId;
	const int q = vol.quality < 1 ? 1 : (vol.quality > 3 ? 3 : vol.quality);
	const int tile = kTile[q], gd = kSlices[q], gw = (w + tile - 1) / tile, gh = (h + tile - 1) / tile;
	auto make3 = [&](RefCntAutoPtr<ITexture>& t, const char* name)
	{
		Trash(t); t.Release();
		TextureDesc td; td.Name = name; td.Type = RESOURCE_DIM_TEX_3D; td.Width = (Uint32)gw; td.Height = (Uint32)gh; td.Depth = (Uint32)gd;
		td.MipLevels = 1; td.Format = TEX_FORMAT_RGBA16_FLOAT; td.BindFlags = BIND_SHADER_RESOURCE | BIND_UNORDERED_ACCESS; td.Usage = USAGE_DEFAULT;
		device->CreateTexture(td, nullptr, &t);
	};
	if (!st.raw || !st.scat[0] || !st.scat[1] || !st.integ || !st.light || st.sw != gw || st.sh != gh || st.d != gd)
	{
		make3(st.raw, "Vol scatter raw"); make3(st.scat[0], "Vol scatter A"); make3(st.scat[1], "Vol scatter B"); make3(st.integ, "Vol integrated"); make3(st.light, "Vol light");
		st.sw = gw; st.sh = gh; st.d = gd; st.valid = false; st.cur = 0;
	}
	if (!st.out || st.w != w || st.h != h)
	{
		Trash(st.out); st.out.Release();
		TextureDesc td; td.Name = "Vol composite"; td.Type = RESOURCE_DIM_TEX_2D; td.Width = (Uint32)w; td.Height = (Uint32)h;
		td.MipLevels = 1; td.Format = HDR_FMT; td.BindFlags = BIND_SHADER_RESOURCE | BIND_RENDER_TARGET; td.Usage = USAGE_DEFAULT;
		device->CreateTexture(td, nullptr, &st.out);
		st.w = w; st.h = h;
	}
	if (!st.raw || !st.scat[0] || !st.scat[1] || !st.integ || !st.light || !st.out) return;
	// Fluid volumes: step every visible one this frame; the density field replaces its analytic shape.
	float fluidDt = 0.0f;
	{   // wall-clock step, capped (a hitch must not fling the fluid): the wind clock runs in its own units
		const double t = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
		if (volLastTime >= 0.0) fluidDt = (float)std::max(0.0, std::min(0.05, t - volLastTime));
		volLastTime = t;
		volWindClock += fluidDt;   // the medium drifts at the wind's m/s over REAL seconds
	}
	ITextureView* fluidSRV[kFluidSlots] = { nullptr, nullptr, nullptr, nullptr };
	std::map<uint64_t, int> fluidSlotOf;
	if (fogVolCB)   // local fog volumes + the wind that advects their noise
	{
		MapHelper<FogVolCBData> fc(context, fogVolCB, MAP_WRITE, MAP_FLAG_DISCARD);
		if (fc != nullptr)
		{
			memset(fc, 0, sizeof(FogVolCBData));
			fc->wind[0] = windDirStrength[0] * windDirStrength[3]; fc->wind[1] = windDirStrength[1] * windDirStrength[3];
			fc->wind[2] = windDirStrength[2] * windDirStrength[3]; fc->wind[3] = volWindClock;
			// Culling: a volume outside the frustum, or behind the scene by last frame's Hi-Z
			// verdict (the volumes ride the occlusion pass with the draws), costs nothing.
			float vpm[16]; { const float4x4 VP = gbufView * gbufProj; memcpy(vpm, &VP, sizeof(vpm)); }
			volOcclTags.clear();
			int n = 0;
			std::vector<const NukeFogVolumeDesc*> reflOnly;   // culled from the grid, kept for reflections
			auto fillStatic = [&](FogVolGPU& g, const NukeFogVolumeDesc& d)
			{
				g.posShape[0] = d.pos[0]; g.posShape[1] = d.pos[1]; g.posShape[2] = d.pos[2]; g.posShape[3] = (float)d.shape;
				g.extDensity[0] = d.halfExt[0]; g.extDensity[1] = d.halfExt[1]; g.extDensity[2] = d.halfExt[2]; g.extDensity[3] = d.density;
				g.rot[0] = -d.rot[0]; g.rot[1] = -d.rot[1]; g.rot[2] = -d.rot[2]; g.rot[3] = d.rot[3];   // conjugate: world -> local
				g.albedoFall[0] = d.albedo[0]; g.albedoFall[1] = d.albedo[1]; g.albedoFall[2] = d.albedo[2]; g.albedoFall[3] = std::max(0.0f, std::min(1.0f, d.falloff));
				g.emisNoise[0] = d.emission[0]; g.emisNoise[1] = d.emission[1]; g.emisNoise[2] = d.emission[2]; g.emisNoise[3] = std::max(0.0f, std::min(1.0f, d.noise));
				g.noiseMisc[0] = std::max(d.noiseScale, 0.1f); g.noiseMisc[1] = d.windAdvect; g.noiseMisc[2] = std::max(d.shaftDensity, 0.0f); g.noiseMisc[3] = std::max(d.heightFalloff, 0.0f);
			};
			for (const NukeFogVolumeDesc& d : fogVols)
			{
				if (n >= kFogVolMax) break;
				// world AABB of the rotated shape: |R| * half extents
				const float qx = d.rot[0], qy = d.rot[1], qz = d.rot[2], qw = d.rot[3];
				const float r[9] = { 1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw),
				                     2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw),
				                     2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy) };
				float ext[3], mn[3], mx[3];
				for (int a = 0; a < 3; ++a)
				{
					ext[a] = std::fabs(r[a * 3 + 0]) * d.halfExt[0] + std::fabs(r[a * 3 + 1]) * d.halfExt[1] + std::fabs(r[a * 3 + 2]) * d.halfExt[2];
					mn[a] = d.pos[a] - ext[a]; mx[a] = d.pos[a] + ext[a];
				}
				int oL = 0, oR = 0, oB = 0, oT = 0, oN = 0, oF = 0;
				for (int c = 0; c < 8; ++c)
				{
					const float px = (c & 1) ? mx[0] : mn[0], py = (c & 2) ? mx[1] : mn[1], pz = (c & 4) ? mx[2] : mn[2];
					const float x = px * vpm[0] + py * vpm[4] + pz * vpm[8]  + vpm[12];
					const float y = px * vpm[1] + py * vpm[5] + pz * vpm[9]  + vpm[13];
					const float z = px * vpm[2] + py * vpm[6] + pz * vpm[10] + vpm[14];
					const float ww = px * vpm[3] + py * vpm[7] + pz * vpm[11] + vpm[15];
					if (x < -ww) ++oL; if (x > ww) ++oR; if (y < -ww) ++oB; if (y > ww) ++oT; if (z < 0.0f) ++oN; if (z > ww) ++oF;
				}
				// Culled from the GRID only: a volume outside the frustum or hidden behind the scene
				// still shows in reflections, so it joins the list after the visible ones (count.y).
				bool culled = (oL == 8 || oR == 8 || oB == 8 || oT == 8 || oN == 8 || oF == 8);   // frustum
				if (!culled)
				{
					OcclTag tag; tag.id = FogVolOcclId(d); memcpy(tag.mn, mn, sizeof(mn)); memcpy(tag.mx, mx, sizeof(mx));
					volOcclTags.push_back(tag);
					if (occlEnabled && !OcclDecide(tag.id)) culled = true;                          // Hi-Z: hidden last frame
				}
				if (culled) { reflOnly.push_back(&d); continue; }
				FogVolGPU& g = fc->vols[n++];
				g.fluidInfo[0] = g.fluidInfo[1] = g.fluidInfo[2] = g.fluidInfo[3] = 0.0f;
				if (d.fluid && d.shape == 0)   // box shapes carry the sim grid; up to kFluidSlots per frame
				{
					int slot = -1;
					for (int k = 0; k < kFluidSlots; ++k) if (!fluidSRV[k]) { slot = k; break; }
					if (slot >= 0)
						if (ITextureView* fv = StepFluid(d, fluidDt)) { fluidSRV[slot] = fv; g.fluidInfo[0] = (float)(slot + 1); }
				}
				fillStatic(g, d);
			}
			fc->count[0] = n;   // the grid's volumes
			for (const NukeFogVolumeDesc* d : reflOnly)
			{
				if (n >= kFogVolMax) break;
				FogVolGPU& g = fc->vols[n++];
				g.fluidInfo[0] = g.fluidInfo[1] = g.fluidInfo[2] = g.fluidInfo[3] = 0.0f;   // analytic rest shape in reflections
				fillStatic(g, *d);
			}
			fc->count[1] = n;   // ... plus the culled ones: what the reflections see
		}
	}

	const float gn = std::max(curNear, 0.25f);
	const float gf = std::max(std::min(vol.maxDistance, curFar), gn * 2.0f);
	{
		MapHelper<VolCBData> cb(context, volCB, MAP_WRITE, MAP_FLAG_DISCARD);
		if (cb == nullptr) return;
		cb->view = gbufView; cb->proj = gbufProj;
		cb->invViewProj  = (gbufView * gbufProj).Inverse();
		cb->prevViewProj = st.valid ? (st.prevView * st.prevProj) : (gbufView * gbufProj);
		cb->grid[0] = (float)gw; cb->grid[1] = (float)gh; cb->grid[2] = (float)gd; cb->grid[3] = (float)volFrame;
		cb->range[0] = gn; cb->range[1] = gf; cb->range[2] = logf(gf / gn); cb->range[3] = 1.0f / logf(gf / gn);
		cb->cam[0] = curNear; cb->cam[1] = curFar; cb->cam[2] = curCamPos[0]; cb->cam[3] = curCamPos[1];
		cb->medium[0] = std::max(vol.density, 0.0f); cb->medium[1] = vol.heightBase; cb->medium[2] = std::max(vol.heightFalloff, 0.0f);
		cb->medium[3] = std::max(-0.95f, std::min(0.95f, vol.anisotropy));
		cb->albedo[0] = vol.albedo[0]; cb->albedo[1] = vol.albedo[1]; cb->albedo[2] = vol.albedo[2]; cb->albedo[3] = std::max(vol.lightIntensity, 0.0f);
		// History weight: 0.9 while the view holds, down to 0.5 as it turns - the in-scatter is
		// view-dependent (the phase toward the sun), and reprojected history dragged the sun's
		// glow behind a turning camera. The lights are sub-sampled, so less history is not noise.
		float blend = 0.0f;
		if (st.valid)
		{
			float pf[16], cf[16]; memcpy(pf, &st.prevView, sizeof(pf)); memcpy(cf, &gbufView, sizeof(cf));
			const float dotf = pf[2] * cf[2] + pf[6] * cf[6] + pf[10] * cf[10];   // the view matrices' forward columns
			const float ang = std::acos(std::max(-1.0f, std::min(1.0f, dotf))) * 57.2958f;
			const float t = std::max(0.0f, std::min(1.0f, ang / 1.5f));
			blend = 0.9f - 0.4f * t;
		}
		cb->misc[0] = std::max(vol.ambientIntensity, 0.0f); cb->misc[1] = blend; cb->misc[2] = hdr ? 0.0f : 1.0f; cb->misc[3] = sky.whitePoint;
		const int j = volFrame % 64 + 1;
		cb->jitter[0] = Halton(j, 2); cb->jitter[1] = Halton(j, 3); cb->jitter[2] = Halton(j, 5); cb->jitter[3] = curCamPos[2];
		cb->screen[0] = (float)w; cb->screen[1] = (float)h; cb->screen[2] = (float)tile; cb->screen[3] = std::max(vol.shaftDensity, 0.0f);
	}
	ITextureView* white = whiteTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
	auto set = [&](IShaderResourceBinding* srb, const char* n, IDeviceObject* o) { if (auto* s = srb->GetVariableByName(SHADER_TYPE_COMPUTE, n)) s->Set(o); };
	context->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE);
	// 1) inject
	set(volInjectSRB, "g_Scat",     st.raw->GetDefaultView(TEXTURE_VIEW_UNORDERED_ACCESS));
	set(volInjectSRB, "g_VolLightOut", st.light->GetDefaultView(TEXTURE_VIEW_UNORDERED_ACCESS));
	set(volInjectSRB, "g_GIIrr",    giIrrSRV ? giIrrSRV : white);
	EnsureVolFallbacks();
	{
		ITextureView* clear3 = clearTex3D ? clearTex3D->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE) : nullptr;
		static const char* kFluidNames[kFluidSlots] = { "g_Fluid0", "g_Fluid1", "g_Fluid2", "g_Fluid3" };
		for (int k = 0; k < kFluidSlots; ++k) set(volInjectSRB, kFluidNames[k], fluidSRV[k] ? fluidSRV[k] : clear3);
	}
	set(volInjectSRB, "g_GIVis",    giVisSRV ? giVisSRV : white);
	if (rtSupported)
	{
		set(volInjectSRB, "g_TLAS",      (IDeviceObject*)tlas.RawPtr());
		set(volInjectSRB, "g_Instances", rtInstSRV);
		set(volInjectSRB, "g_DynCol",    rtDynColSRV ? rtDynColSRV : rtNrmSRV);
		set(volInjectSRB, "g_AllNrm",    rtNrmSRV);
		set(volInjectSRB, "g_AllUV",     rtUVSRV ? rtUVSRV : rtNrmSRV);
		set(volInjectSRB, "g_AllPos",    rtPosSRV ? rtPosSRV : rtNrmSRV);
		set(volInjectSRB, "g_MatBytes",  rtMatSRV ? rtMatSRV : rtInstSRV);
	}
	else { set(volInjectSRB, "g_Shadow", shadowSRV); set(volInjectSRB, "g_ShadowCube", shadowCubeSRV); }
	context->SetPipelineState(volInjectPSO);
	context->CommitShaderResources(volInjectSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	context->DispatchCompute(DispatchComputeAttribs((Uint32)(gw + 7) / 8, (Uint32)(gh + 7) / 8, (Uint32)(gd + 3) / 4));
	// 1b) temporal: the raw grid against the clamped, reprojected history
	set(volTemporalSRB, "g_ScatRaw",  st.raw->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
	set(volTemporalSRB, "g_ScatPrev", st.scat[st.cur ^ 1]->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
	set(volTemporalSRB, "g_Scat",     st.scat[st.cur]->GetDefaultView(TEXTURE_VIEW_UNORDERED_ACCESS));
	context->SetPipelineState(volTemporalPSO);
	context->CommitShaderResources(volTemporalSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	context->DispatchCompute(DispatchComputeAttribs((Uint32)(gw + 7) / 8, (Uint32)(gh + 7) / 8, (Uint32)(gd + 3) / 4));
	// 2) integrate the columns
	set(volIntegrateSRB, "g_Scat",  st.scat[st.cur]->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
	set(volIntegrateSRB, "g_Integ", st.integ->GetDefaultView(TEXTURE_VIEW_UNORDERED_ACCESS));
	context->SetPipelineState(volIntegratePSO);
	context->CommitShaderResources(volIntegrateSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	context->DispatchCompute(DispatchComputeAttribs((Uint32)(gw + 7) / 8, (Uint32)(gh + 7) / 8, 1));
	st.prevView = gbufView; st.prevProj = gbufProj; st.cur ^= 1; st.valid = true;
	++volFrame;
	volCur = &st;
}

// ---- Fluid volumes ---------------------------------------------------------------------------------
bool NukeDiligent::Impl::BuildFluidPipes()
{
	const string cs = shaderSource("vol_fluid.cs");
	if (cs.empty()) return false;
	auto sf = ShaderFactory();
	RefCntAutoPtr<IShader> sh;
	{
		ShaderCreateInfo sci; sci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL; sci.pShaderSourceStreamFactory = sf;
		sci.Desc = {"Vol fluid CS", SHADER_TYPE_COMPUTE, true}; sci.Source = cs.c_str();
		CreateShaderCached(sci, &sh);
		if (!sh) return false;
	}
	if (!fluidCB)
	{
		BufferDesc d; d.Name = "FluidCB"; d.Size = kFluidCBSize; d.Usage = USAGE_DYNAMIC;
		d.BindFlags = BIND_UNIFORM_BUFFER; d.CPUAccessFlags = CPU_ACCESS_WRITE;
		device->CreateBuffer(d, nullptr, &fluidCB);
		if (!fluidCB) return false;
	}
	SamplerDesc lin; lin.MinFilter = FILTER_TYPE_LINEAR; lin.MagFilter = FILTER_TYPE_LINEAR; lin.MipFilter = FILTER_TYPE_LINEAR;
	lin.AddressU = TEXTURE_ADDRESS_CLAMP; lin.AddressV = TEXTURE_ADDRESS_CLAMP; lin.AddressW = TEXTURE_ADDRESS_CLAMP;
	ComputePipelineStateCreateInfo ci; ci.PSODesc.Name = "Vol fluid PSO";
	ShaderResourceVariableDesc vars[] = { {SHADER_TYPE_COMPUTE, "FluidCB", SHADER_RESOURCE_VARIABLE_TYPE_STATIC} };
	ImmutableSamplerDesc imms[] = { {SHADER_TYPE_COMPUTE, "g_VelIn", lin}, {SHADER_TYPE_COMPUTE, "g_DensIn", lin} };
	ci.PSODesc.ResourceLayout.Variables = vars; ci.PSODesc.ResourceLayout.NumVariables = 1;
	ci.PSODesc.ResourceLayout.DefaultVariableType = SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC;
	ci.PSODesc.ResourceLayout.ImmutableSamplers = imms; ci.PSODesc.ResourceLayout.NumImmutableSamplers = 2;
	ci.pCS = sh;
	RefCntAutoPtr<IPipelineState> pso;
	CreateComputePipelineStateCached(ci, &pso);
	if (!pso) return false;
	if (auto* v = pso->GetStaticVariableByName(SHADER_TYPE_COMPUTE, "FluidCB")) v->Set(fluidCB);
	RefCntAutoPtr<IShaderResourceBinding> srb;
	pso->CreateShaderResourceBinding(&srb, true);
	if (!srb) return false;
	fluidPSO = pso; fluidSRB = srb;
	return true;
}

// One simulation step of a fluid volume: velocity advection + forces (wind, displacers, curl
// noise), pressure projection (Jacobi), density advection + refill toward the shape. Returns
// the density field for this frame's inject.
ITextureView* NukeDiligent::Impl::StepFluid(const NukeFogVolumeDesc& d, float dt)
{
	if (fluidFailed) return nullptr;
	if (!fluidPSO)
	{
		if (!fluidBuilding.exchange(true))
			EnqueueBuild([this] { if (!BuildFluidPipes()) { fluidFailed = true; cout << "[NukeDiligent]\tfluid fog pipeline failed to build; fluid volumes stay static" << endl; } },
			             [this] { fluidBuilding = false; }, kPrioGBuffer, "Fluid fog pipeline");
		return nullptr;
	}
	FluidState& st = fluidStates[d.id];
	st.lastUsed = frameId;
	// The air steps at 30 Hz (a 250 Hz step moves a thirtieth of a cell: all interpolation blur);
	// the parcels move and splat every frame with the frame's dt - a 30 Hz hop printed stepped trails.
	st.accum += dt;
	const bool airDue = st.accum >= 1.0f / 30.0f || !st.valid;
	const float airDt = std::min(st.accum, 0.05f);
	if (airDue) st.accum = 0.0f;
	dt = std::min(dt, 0.05f);
	// resolution: fluidRes along the longest half extent, the others in proportion (multiples of 4)
	const float he[3] = { std::max(d.halfExt[0], 0.01f), std::max(d.halfExt[1], 0.01f), std::max(d.halfExt[2], 0.01f) };
	const float hmax = std::max(he[0], std::max(he[1], he[2]));
	const int res = std::max(8, std::min(192, d.fluidRes));
	int r[3];
	for (int a = 0; a < 3; ++a) r[a] = std::max(8, ((int)std::lround(res * he[a] / hmax) + 3) / 4 * 4);
	if (!st.map[0] || st.rx != r[0] || st.ry != r[1] || st.rz != r[2])
	{
		for (auto& t : st.map) { Trash(t); t.Release(); } for (auto& t : st.vel) { Trash(t); t.Release(); } for (auto& t : st.prs) { Trash(t); t.Release(); } Trash(st.div); st.div.Release(); Trash(st.rho); st.rho.Release();
		auto make = [&](RefCntAutoPtr<ITexture>& t, const char* name, TEXTURE_FORMAT fmt)
		{
			TextureDesc td; td.Name = name; td.Type = RESOURCE_DIM_TEX_3D; td.Width = (Uint32)r[0]; td.Height = (Uint32)r[1]; td.Depth = (Uint32)r[2];
			td.MipLevels = 1; td.Format = fmt; td.BindFlags = BIND_SHADER_RESOURCE | BIND_UNORDERED_ACCESS; td.Usage = USAGE_DEFAULT;
			device->CreateTexture(td, nullptr, &t);
		};
		make(st.map[0], "Fluid density A", TEX_FORMAT_R16_FLOAT); make(st.map[1], "Fluid density B", TEX_FORMAT_R16_FLOAT); make(st.map[2], "Fluid density C", TEX_FORMAT_R16_FLOAT);
		make(st.vel[0], "Fluid velocity A", TEX_FORMAT_RGBA16_FLOAT); make(st.vel[1], "Fluid velocity B", TEX_FORMAT_RGBA16_FLOAT);
		make(st.prs[0], "Fluid pressure A", TEX_FORMAT_R16_FLOAT); make(st.prs[1], "Fluid pressure B", TEX_FORMAT_R16_FLOAT);
		make(st.div, "Fluid divergence", TEX_FORMAT_R16_FLOAT); make(st.rho, "Fluid fog", TEX_FORMAT_R16_FLOAT);
		if (!st.ledger)
		{
			BufferDesc bd; bd.Name = "Fluid mass ledger"; bd.Size = 32; bd.BindFlags = BIND_UNORDERED_ACCESS; bd.Usage = USAGE_DEFAULT;
			bd.Mode = BUFFER_MODE_STRUCTURED; bd.ElementByteStride = 4;
			const unsigned zeros[8] = { 0, 0, 0, 0, 0, 0, 0, 0 }; BufferData init; init.pData = zeros; init.DataSize = 32;
			device->CreateBuffer(bd, &init, &st.ledger);
		}
		st.rx = r[0]; st.ry = r[1]; st.rz = r[2]; st.cur = 0; st.valid = false; st.ledgerCur = 0;
	}
	// Clumps: a parcel's radius is the Clump Size (or a tenth of the shape's smaller side), the
	// parcels stand 0.55 radii apart (overlapping into one fog), the splat grid has three cells
	// per radius - none of it tied to the air grid, so the look does not change with the
	// resolution or the shape's size.
	const bool clumps = d.fluidMode == 1;
	if (clumps)
	{
		const float hmin = std::min(he[0], std::min(he[1], he[2]));
		const float rad = d.clumpSize > 0.01f ? d.clumpSize : std::max(0.1f, 0.2f * hmin);
		const float vol = 8.0f * he[0] * he[1] * he[2], spacing = rad * 0.55f;
		const int n = std::max(256, std::min(65536, (int)(vol / (spacing * spacing * spacing)))) / 256 * 256;
		int sr[3]; for (int a = 0; a < 3; ++a) sr[a] = std::max(8, std::min(96, ((int)std::ceil(2.0f * he[a] / (rad / 3.0f)) + 3) / 4 * 4));
		if (!st.parcels || !st.acc || !st.rhoS || st.parcelCount != n || st.sx != sr[0] || st.sy != sr[1] || st.sz != sr[2] || std::fabs(st.parcelRadius - rad) > 1e-4f)
		{
			Trash(st.parcels); st.parcels.Release(); Trash(st.acc); st.acc.Release(); Trash(st.rhoS); st.rhoS.Release();
			auto make3 = [&](RefCntAutoPtr<ITexture>& t, const char* name, TEXTURE_FORMAT fmt)
			{
				TextureDesc td; td.Name = name; td.Type = RESOURCE_DIM_TEX_3D; td.Width = (Uint32)sr[0]; td.Height = (Uint32)sr[1]; td.Depth = (Uint32)sr[2];
				td.MipLevels = 1; td.Format = fmt; td.BindFlags = BIND_SHADER_RESOURCE | BIND_UNORDERED_ACCESS; td.Usage = USAGE_DEFAULT;
				device->CreateTexture(td, nullptr, &t);
			};
			make3(st.acc, "Fluid parcel splat", TEX_FORMAT_R32_UINT); make3(st.rhoS, "Fluid clump fog", TEX_FORMAT_R16_FLOAT);
			BufferDesc bd; bd.Name = "Fluid parcels"; bd.Size = (Uint64)n * 32; bd.BindFlags = BIND_UNORDERED_ACCESS; bd.Usage = USAGE_DEFAULT;
			bd.Mode = BUFFER_MODE_STRUCTURED; bd.ElementByteStride = 16;
			device->CreateBuffer(bd, nullptr, &st.parcels);
			st.parcelCount = n; st.parcelRadius = rad; st.sx = sr[0]; st.sy = sr[1]; st.sz = sr[2]; st.valid = false;
			if (!st.acc || !st.rhoS || !st.parcels) return nullptr;
		}
	}
	if (st.mode != d.fluidMode) { st.mode = d.fluidMode; st.valid = false; }
	{
		if (!st.map[0] || !st.map[1] || !st.map[2] || !st.vel[0] || !st.vel[1] || !st.prs[0] || !st.prs[1] || !st.div || !st.rho) return nullptr;
	}
	// displacers in the box's local frame
	const float qx = -d.rot[0], qy = -d.rot[1], qz = -d.rot[2], qw = d.rot[3];   // world -> local
	auto toLocal = [&](const float* v, bool point, float* o)
	{
		float x = v[0], y = v[1], z = v[2];
		if (point) { x -= d.pos[0]; y -= d.pos[1]; z -= d.pos[2]; }
		// q * v * q^-1
		const float tx = 2.0f * (qy * z - qz * y), ty = 2.0f * (qz * x - qx * z), tz = 2.0f * (qx * y - qy * x);
		o[0] = x + qw * tx + (qy * tz - qz * ty); o[1] = y + qw * ty + (qz * tx - qx * tz); o[2] = z + qw * tz + (qx * ty - qy * tx);
	};
	float passDt = dt;
	auto fill = [&](int mode)
	{
		MapHelper<FluidCBData> cb(context, fluidCB, MAP_WRITE, MAP_FLAG_DISCARD);
		if (cb == nullptr) return false;
		memset(cb, 0, sizeof(FluidCBData));
		cb->res[0] = (float)r[0]; cb->res[1] = (float)r[1]; cb->res[2] = (float)r[2]; cb->res[3] = (float)mode;
		cb->box[0] = he[0]; cb->box[1] = he[1]; cb->box[2] = he[2]; cb->box[3] = st.valid ? passDt : 0.0f;
		{ const float wu[3] = { 0.0f, 1.0f, 0.0f }; float lu[3]; toLocal(wu, false, lu); cb->up[0] = lu[0]; cb->up[1] = lu[1]; cb->up[2] = lu[2]; cb->up[3] = 0.0f; }   // world up in the box frame (vortex axis)
		cb->pos[0] = d.pos[0]; cb->pos[1] = d.pos[1]; cb->pos[2] = d.pos[2]; cb->pos[3] = (float)d.shape;
		float wl[3]; const float ww[3] = { windDirStrength[0] * windDirStrength[3], windDirStrength[1] * windDirStrength[3], windDirStrength[2] * windDirStrength[3] };
		toLocal(ww, false, wl);
		cb->wind[0] = wl[0]; cb->wind[1] = wl[1]; cb->wind[2] = wl[2]; cb->wind[3] = d.fluidRefill;
		cb->params[0] = d.fluidTurbulence; cb->params[1] = d.fluidDissipation; cb->params[2] = std::max(0.0f, std::min(1.0f, d.noise)); cb->params[3] = std::max(d.noiseScale, 0.1f);
		cb->params2[0] = std::max(0.0f, std::min(1.0f, d.falloff)); cb->params2[1] = volWindClock; cb->params2[2] = d.windAdvect; cb->params2[3] = st.valid ? 1.0f : 0.0f;
		int n = 0;
		for (const NukeFogDisplacerDesc& dp : fogDisplacers)
		{
			if (n >= 8) break;
			float lp[3], lv[3]; toLocal(dp.pos, true, lp); toLocal(dp.vel, false, lv);
			if (std::fabs(lp[0]) > he[0] + dp.radius || std::fabs(lp[1]) > he[1] + dp.radius || std::fabs(lp[2]) > he[2] + dp.radius) continue;
			cb->dispPos[n][0] = lp[0]; cb->dispPos[n][1] = lp[1]; cb->dispPos[n][2] = lp[2]; cb->dispPos[n][3] = dp.radius;
			cb->dispVel[n][0] = lv[0]; cb->dispVel[n][1] = lv[1]; cb->dispVel[n][2] = lv[2]; cb->dispVel[n][3] = dp.strength;
			++n;
		}
		// Force Fields / wind zones: the frame's bend volumes, in the box frame (strength = m/s^2)
		int nf = 0;
		for (int i = 0; i < bendVolumeCount && nf < 8; ++i)
		{
			const float* bv = bendVolumes[i];
			float lp[3], ld[3]; toLocal(bv, true, lp); toLocal(bv + 4, false, ld);
			const float rad = std::max(bv[3], 0.01f);
			if (std::fabs(lp[0]) > he[0] + rad || std::fabs(lp[1]) > he[1] + rad || std::fabs(lp[2]) > he[2] + rad) continue;
			cb->forcePos[nf][0] = lp[0]; cb->forcePos[nf][1] = lp[1]; cb->forcePos[nf][2] = lp[2]; cb->forcePos[nf][3] = rad;
			cb->forceDir[nf][0] = ld[0]; cb->forceDir[nf][1] = ld[1]; cb->forceDir[nf][2] = ld[2]; cb->forceDir[nf][3] = bv[7];
			cb->forceMisc[nf][0] = bv[8]; cb->forceMisc[nf][1] = bv[9]; cb->forceMisc[nf][2] = bv[10]; cb->forceMisc[nf][3] = bv[11];   // vortex pull
			cb->forceDent[nf][0] = bv[13]; cb->forceDent[nf][1] = bv[14]; cb->forceDent[nf][2] = bv[15]; cb->forceDent[nf][3] = bv[16];   // depth, sharpness, size, density
			cb->forceDent2[nf][0] = bv[12]; cb->forceDent2[nf][1] = cb->forceDent2[nf][2] = cb->forceDent2[nf][3] = 0.0f;               // inner radius
			++nf;
		}
		cb->counts[0] = (float)n; cb->counts[1] = (float)nf; cb->counts[2] = std::max(d.heightFalloff, 0.0f); cb->counts[3] = (float)st.ledgerCur;
		cb->misc[0] = (float)n; cb->misc[1] = 2.0f * he[0] / r[0]; cb->misc[2] = 2.0f * he[1] / r[1]; cb->misc[3] = 2.0f * he[2] / r[2];
		cb->pos[0] = st.parcelRadius;
		cb->splat[0] = (float)st.sx; cb->splat[1] = (float)st.sy; cb->splat[2] = (float)st.sz; cb->splat[3] = (float)st.parcelCount;
		return true;
	};
	auto set = [&](const char* n, IDeviceObject* o) { if (auto* v = fluidSRB->GetVariableByName(SHADER_TYPE_COMPUTE, n)) v->Set(o); };
	auto srv = [](ITexture* t) { return t->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE); };
	auto uav = [](ITexture* t) { return t->GetDefaultView(TEXTURE_VIEW_UNORDERED_ACCESS); };
	const DispatchComputeAttribs da((Uint32)(r[0] + 7) / 8, (Uint32)(r[1] + 7) / 8, (Uint32)(r[2] + 3) / 4);
	auto run = [&](int mode, ITexture* velIn, ITexture* densIn, ITexture* prsIn, ITexture* divIn, ITexture* velOut, ITexture* densOut, ITexture* prsOut, ITexture* divOut)
	{
		if (!fill(mode)) return;
		set("g_VelIn", srv(velIn)); set("g_DensIn", srv(densIn)); set("g_PrsIn", srv(prsIn)); set("g_DivIn", srv(divIn));
		set("g_VelOut", uav(velOut)); set("g_DensOut", uav(densOut)); set("g_PrsOut", uav(prsOut)); set("g_DivOut", uav(divOut));
		context->SetPipelineState(fluidPSO);
		context->CommitShaderResources(fluidSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		context->DispatchCompute(da);
	};
	const int c = st.cur % 2, o = c ^ 1, mi = st.cur % 3, ms = (mi + 1) % 3, mo = (mi + 2) % 3;
	ITexture *vA = st.vel[c], *vB = st.vel[o], *dA = st.map[mi], *dS = st.map[ms], *dB = st.map[mo], *pA = st.prs[0], *pB = st.prs[1], *dv = st.div;
	// 0: velocity advection + forces (vA -> vB); 1: divergence of vB + clear pressure (-> dv, pA);
	// 2: Jacobi x12 (pA <-> pB); 3: project vB - grad p (-> vA); 4/5: the reference map advected by vA (dA -> dS -> dB), the fog -> rho
	set("g_RhoOut", uav(clumps ? st.rhoS : st.rho)); set("g_Ledger", st.ledger->GetDefaultView(BUFFER_VIEW_UNORDERED_ACCESS));   // every pass commits the whole SRB
	set("g_Scratch", srv(dB));
	if (!st.parcels || !st.acc)
	{   // grid mode: the clump resources still have to be bound (the SRB is committed whole) - tiny stand-ins of the right type
		TextureDesc td; td.Name = "Fluid parcel splat (stand-in)"; td.Type = RESOURCE_DIM_TEX_3D; td.Width = td.Height = td.Depth = 8;
		td.MipLevels = 1; td.Format = TEX_FORMAT_R32_UINT; td.BindFlags = BIND_SHADER_RESOURCE | BIND_UNORDERED_ACCESS; td.Usage = USAGE_DEFAULT;
		if (!st.acc) device->CreateTexture(td, nullptr, &st.acc);
		BufferDesc bd; bd.Name = "Fluid parcels (stand-in)"; bd.Size = 256 * 32; bd.BindFlags = BIND_UNORDERED_ACCESS; bd.Usage = USAGE_DEFAULT;
		bd.Mode = BUFFER_MODE_STRUCTURED; bd.ElementByteStride = 16;
		if (!st.parcels) device->CreateBuffer(bd, nullptr, &st.parcels);
		if (!st.acc || !st.parcels) return nullptr;
	}
	set("g_Parcels", st.parcels->GetDefaultView(BUFFER_VIEW_UNORDERED_ACCESS)); set("g_Acc", uav(st.acc));
	auto runP = [&](int mode, ITexture* velIn)   // per-parcel passes: 256 threads a group
	{
		if (!fill(mode)) return;
		set("g_VelIn", srv(velIn));
		context->SetPipelineState(fluidPSO);
		context->CommitShaderResources(fluidSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		context->DispatchCompute(DispatchComputeAttribs((Uint32)(st.parcelCount + 255) / 256, 1, 1));
	};
	ITexture* pin = pA; ITexture* pout = pB;
	if (airDue)
	{
		passDt = airDt;
		run(0, vA, dA, pA, dv, vB, dB, pB, dv);
		run(1, vB, dA, pA, dv, vA, dB, pA, dv);
		for (int i = 0; i < 12; ++i) { run(2, vB, dA, pin, dv, vA, dB, pout, dv); std::swap(pin, pout); }
		run(3, vB, dA, pin, dv, vA, dB, pout, dv);
		passDt = dt;
	}
	if (clumps)
	{
		const DispatchComputeAttribs ds((Uint32)(st.sx + 7) / 8, (Uint32)(st.sy + 7) / 8, (Uint32)(st.sz + 3) / 4);
		auto runS = [&](int mode)   // splat-grid passes
		{
			if (!fill(mode)) return;
			set("g_VelIn", srv(vA));
			context->SetPipelineState(fluidPSO);
			context->CommitShaderResources(fluidSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
			context->DispatchCompute(ds);
		};
		runS(5);        // clear the splat and next frame's ledger slot, sum the resting amount
		runP(4, vA);    // the parcels ride the air + the fields (the air of the last 30 Hz step)
		runP(6, vA);    // splat
		runS(7);        // resolve -> rhoS
		if (airDue) st.cur = (st.cur + 5) % 6;   // the velocity flips only when the air stepped
		st.valid = true; st.ledgerCur = (st.ledgerCur + 1) % 3;
		return srv(st.rhoS);
	}
	if (!airDue) { return srv(st.rho); }   // grid mode: everything at the air's rate
	passDt = airDt;
	run(8, vA, dA, pin, dv, vB, dS, pout, dv);   // density: forward advection -> the scratch
	set("g_Scratch", srv(dS));
	run(9, vA, dA, pin, dv, vB, dB, pout, dv);   // MacCormack correction, continuity, relaxation -> dB, rho
	st.cur = (st.cur + 5) % 6; st.valid = true; st.ledgerCur = (st.ledgerCur + 1) % 3;
	return srv(st.rho);
}

// A fog volume's occlusion tag: stable while the volume stands still (the hash of its placement;
// a moving one is "never tested" = visible until its verdict lands), top bit = not a draw.
uint64_t NukeDiligent::Impl::FogVolOcclId(const NukeFogVolumeDesc& d)
{
	uint64_t h = 1469598103934665603ull;
	auto mix = [&](long long v) { h ^= (uint64_t)v; h *= 1099511628211ull; };
	for (int i = 0; i < 3; ++i) { mix((long long)std::lround(d.pos[i] * 8.0f)); mix((long long)std::lround(d.halfExt[i] * 8.0f)); }
	mix(d.shape);
	return h | (1ull << 63);
}

void NukeDiligent::Impl::EnsureVolFallbacks()
{
	if (clearTex3D) return;
	TextureDesc td; td.Name = "Clear 1x1x1"; td.Type = RESOURCE_DIM_TEX_3D; td.Width = td.Height = td.Depth = 1;
	td.MipLevels = 1; td.Format = TEX_FORMAT_RGBA8_UNORM; td.BindFlags = BIND_SHADER_RESOURCE; td.Usage = USAGE_IMMUTABLE;
	const uint8_t px[4] = {0, 0, 0, 255};   // no in-scatter, no extinction
	TextureSubResData sub; sub.pData = px; sub.Stride = 4; sub.DepthStride = 4;
	TextureData init; init.pSubResources = &sub; init.NumSubresources = 1;
	device->CreateTexture(td, &init, &clearTex3D);
}

ITextureView* NukeDiligent::Impl::VolScatSRV()
{
	EnsureVolFallbacks();
	if (volCur && volCur->scat[volCur->cur ^ 1]) return volCur->scat[volCur->cur ^ 1]->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);   // RunVolumetrics flipped cur after writing
	return clearTex3D ? clearTex3D->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE) : nullptr;
}

// ---- Screen-space sun shafts ----------------------------------------------------------------------
bool NukeDiligent::Impl::BuildSunShaftPipes()
{
	const string vs = shaderSource("post.vs"), ps = shaderSource("sunshafts.ps");
	if (vs.empty() || ps.empty()) return false;
	auto sf = ShaderFactory();
	RefCntAutoPtr<IShader> sV, sP;
	{
		ShaderCreateInfo sci; sci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL; sci.pShaderSourceStreamFactory = sf;
		sci.Desc = {"Sun shafts VS", SHADER_TYPE_VERTEX, true}; sci.Source = vs.c_str(); CreateShaderCached(sci, &sV);
		sci.Desc = {"Sun shafts PS", SHADER_TYPE_PIXEL, true};  sci.Source = ps.c_str(); CreateShaderCached(sci, &sP);
		if (!sV || !sP) return false;
	}
	if (!sunShaftCB)
	{
		BufferDesc d; d.Name = "SunShaftCB"; d.Size = 5 * 16 + 64; d.Usage = USAGE_DYNAMIC;
		d.BindFlags = BIND_UNIFORM_BUFFER; d.CPUAccessFlags = CPU_ACCESS_WRITE;
		device->CreateBuffer(d, nullptr, &sunShaftCB);
		if (!sunShaftCB) return false;
	}
	SamplerDesc lin; lin.MinFilter = FILTER_TYPE_LINEAR; lin.MagFilter = FILTER_TYPE_LINEAR; lin.MipFilter = FILTER_TYPE_LINEAR;
	lin.AddressU = TEXTURE_ADDRESS_CLAMP; lin.AddressV = TEXTURE_ADDRESS_CLAMP; lin.AddressW = TEXTURE_ADDRESS_CLAMP;
	SamplerDesc pt = lin; pt.MinFilter = FILTER_TYPE_POINT; pt.MagFilter = FILTER_TYPE_POINT; pt.MipFilter = FILTER_TYPE_POINT;
	GraphicsPipelineStateCreateInfo ci; ci.PSODesc.Name = "Sun shafts PSO";
	auto& gp = ci.GraphicsPipeline;
	gp.NumRenderTargets = 1; gp.RTVFormats[0] = HDR_FMT; gp.DSVFormat = TEX_FORMAT_UNKNOWN;
	gp.PrimitiveTopology = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; gp.RasterizerDesc.CullMode = CULL_MODE_NONE;
	gp.DepthStencilDesc.DepthEnable = False; gp.InputLayout.NumElements = 0;
	ShaderResourceVariableDesc vars[] = {
		{SHADER_TYPE_PIXEL, "g_Source", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
		{SHADER_TYPE_PIXEL, "g_Depth",  SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
		{SHADER_TYPE_PIXEL, "g_Mask",   SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
		{SHADER_TYPE_PIXEL, "g_Clouds", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
		{SHADER_TYPE_PIXEL, "g_AtTrans", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},   // the physical atmosphere: the source through it
		{SHADER_TYPE_PIXEL, "AtmoCB",   SHADER_RESOURCE_VARIABLE_TYPE_STATIC},
	};
	ImmutableSamplerDesc imms[] = {{SHADER_TYPE_PIXEL, "g_Source", lin}, {SHADER_TYPE_PIXEL, "g_Depth", pt}, {SHADER_TYPE_PIXEL, "g_Mask", lin}, {SHADER_TYPE_PIXEL, "g_Clouds", lin}, {SHADER_TYPE_PIXEL, "g_AtTrans", AtmoSampler()}};
	ci.PSODesc.ResourceLayout.Variables = vars; ci.PSODesc.ResourceLayout.NumVariables = 6;
	ci.PSODesc.ResourceLayout.ImmutableSamplers = imms; ci.PSODesc.ResourceLayout.NumImmutableSamplers = 5;
	ci.pVS = sV; ci.pPS = sP;
	if (!atmoCB)
	{
		BufferDesc d; d.Name = "AtmoCB"; d.Size = kAtmoCBSize; d.Usage = USAGE_DYNAMIC;
		d.BindFlags = BIND_UNIFORM_BUFFER; d.CPUAccessFlags = CPU_ACCESS_WRITE;
		device->CreateBuffer(d, nullptr, &atmoCB);
	}
	RefCntAutoPtr<IPipelineState> pso;
	CreateGraphicsPipelineStateCached(ci, &pso);
	if (!pso) return false;
	if (auto* c = pso->GetStaticVariableByName(SHADER_TYPE_PIXEL, "SunShaftCB")) c->Set(sunShaftCB);
	if (auto* a = pso->GetStaticVariableByName(SHADER_TYPE_PIXEL, "AtmoCB")) a->Set(atmoCB);
	RefCntAutoPtr<IShaderResourceBinding> srb;
	pso->CreateShaderResourceBinding(&srb, true);
	if (!srb) return false;
	ssSrcVar = srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Source");
	ssDepthVar = srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Depth");
	ssMaskVar = srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Mask");
	ssCloudVar = srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Clouds");
	sunShaftSRB = srb; sunShaftPSO = pso;
	return true;
}

// After the fog composite: the sun's sky, masked by the depth, smeared radially toward the sun
// past the occluders. Own output texture; the input passes through when the sun is behind the
// camera or the effect is off.
ITextureView* NukeDiligent::Impl::RunSunShafts(ITextureView* sceneSRV, int w, int h)
{
	if (!sceneSRV || vol.sunShaftIntensity <= 0.0f || !gbufActive || !gbufDepthSRV || debugView != 0 || w <= 0 || h <= 0 || ssFailed) return sceneSRV;
	if (!sunShaftPSO)
	{
		if (!ssBuilding.exchange(true))
			EnqueueBuild([this] { if (!BuildSunShaftPipes()) { ssFailed = true; cout << "[NukeDiligent]\tsun shafts pipeline failed to build; sun shafts stay off" << endl; } },
			             [this] { ssBuilding = false; }, kPrioGBuffer, "Sun shafts pipeline");
		return sceneSRV;
	}
	const NukeLight* sun = nullptr;
	for (const NukeLight& l : lights) if (l.type == 0) { sun = &l; break; }
	if (!sun || sun->intensity <= 0.0f) return sceneSRV;
	float dir[3] = { -sun->dir[0], -sun->dir[1], -sun->dir[2] };   // toward the sun
	{ const float len = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]); if (len < 1e-6f) return sceneSRV; for (float& v : dir) v /= len; }
	float vpm[16]; { const float4x4 VP = curView * curProjNoJitter; memcpy(vpm, &VP, sizeof(vpm)); }
	const float cx = dir[0] * vpm[0] + dir[1] * vpm[4] + dir[2] * vpm[8];
	const float cy = dir[0] * vpm[1] + dir[1] * vpm[5] + dir[2] * vpm[9];
	const float cw = dir[0] * vpm[3] + dir[1] * vpm[7] + dir[2] * vpm[11];
	if (cw <= 1e-4f) return sceneSRV;   // the sun is behind the camera
	const float ux = cx / cw * 0.5f + 0.5f, uy = 0.5f - cy / cw * 0.5f;
	const float off = std::max(std::fabs(ux - 0.5f), std::fabs(uy - 0.5f)) * 2.0f;   // 1 = the screen edge
	const float t = std::max(0.0f, std::min(1.0f, (off - 1.0f) / 0.8f));
	const float onScreen = 1.0f - t * t * (3.0f - 2.0f * t);   // fades out 0.4 screens past the edge
	if (onScreen <= 0.001f) return sceneSRV;

	const int hw = std::max(1, (w + 1) / 2), hh = std::max(1, (h + 1) / 2);
	if (!ssMask[0] || !ssMask[1] || !ssOut || ssW != w || ssH != h)
	{
		for (auto& tx : ssMask) { Trash(tx); tx.Release(); }
		Trash(ssOut); ssOut.Release();
		TextureDesc td; td.Type = RESOURCE_DIM_TEX_2D; td.MipLevels = 1; td.Format = HDR_FMT;
		td.BindFlags = BIND_SHADER_RESOURCE | BIND_RENDER_TARGET; td.Usage = USAGE_DEFAULT;
		td.Name = "Sun shaft mask"; td.Width = (Uint32)hw; td.Height = (Uint32)hh;
		device->CreateTexture(td, nullptr, &ssMask[0]); device->CreateTexture(td, nullptr, &ssMask[1]);
		td.Name = "Sun shafts out"; td.Width = (Uint32)w; td.Height = (Uint32)h;
		device->CreateTexture(td, nullptr, &ssOut);
		ssW = w; ssH = h;
		if (!ssMask[0] || !ssMask[1] || !ssOut) return sceneSRV;
	}
	struct CB { float sun[4], prm[4], col[4], dir[4]; float4x4 invVP; float ecl[4]; };
	const float4x4 invVP = DirInvVP(false);   // directions only
	// Shaft radiance at the sun: the light's colour, normalised, scaled by its strength (soft):
	// the user's intensity is the artistic gain on top.
	float col[3] = { sun->color[0], sun->color[1], sun->color[2] };
	{ const float m = std::max(col[0], std::max(col[1], col[2])); if (m > 1e-4f) for (float& c : col) c /= m; }
	const float strength = 0.35f * std::max(0.5f, std::min(2.0f, std::sqrt(std::max(sun->intensity, 0.0f) * 0.5f)));
	auto pass = [&](int mode, float reach, ITextureView* src, ITextureView* mask, ITexture* dst, int dw, int dh)
	{
		{
			MapHelper<CB> cb(context, sunShaftCB, MAP_WRITE, MAP_FLAG_DISCARD);
			if (cb == nullptr) return false;
			cb->sun[0] = ux; cb->sun[1] = uy; cb->sun[2] = onScreen; cb->sun[3] = vol.sunShaftIntensity;
			cb->prm[0] = reach; cb->prm[1] = 0.94f; cb->prm[2] = (float)mode; cb->prm[3] = hdr ? 0.0f : 1.0f;
			cb->col[0] = col[0] * strength; cb->col[1] = col[1] * strength; cb->col[2] = col[2] * strength; cb->col[3] = sky.whitePoint;
			cb->dir[0] = dir[0]; cb->dir[1] = dir[1]; cb->dir[2] = dir[2]; cb->dir[3] = std::max(sky.sunSize, 0.0005f);   // w = the disc's angular radius
			cb->invVP = invVP;
			cb->ecl[0] = sky.eclipse; cb->ecl[1] = cb->ecl[2] = cb->ecl[3] = 0.0f;   // the eclipsing moon's offset (sun radii)
		}
		if (ssSrcVar)   ssSrcVar->Set(src);
		if (ssDepthVar) ssDepthVar->Set(gbufDepthSRV);
		if (ssMaskVar)  ssMaskVar->Set(mask);
		if (ssCloudVar) ssCloudVar->Set(cloudCurSRV ? (IDeviceObject*)cloudCurSRV : (IDeviceObject*)whiteTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
		ITextureView* rtv = dst->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET);
		context->SetRenderTargets(1, &rtv, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		Viewport vp; vp.TopLeftX = 0; vp.TopLeftY = 0; vp.Width = (float)dw; vp.Height = (float)dh; vp.MinDepth = 0; vp.MaxDepth = 1;
		context->SetViewports(1, &vp, dw, dh);
		context->SetPipelineState(sunShaftPSO);
		context->CommitShaderResources(sunShaftSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		DrawAttribs da{3, DRAW_FLAG_VERIFY_STATES};
		context->Draw(da);
		return true;
	};
	const float reach = std::max(0.05f, std::min(1.0f, vol.sunShaftLength));
	ITextureView* m0 = ssMask[0]->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
	ITextureView* m1 = ssMask[1]->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
	AtmoBind(sunShaftSRB, SHADER_TYPE_PIXEL);
	FillAtmoCB(0, 0, 0, false);   // the PSO binds AtmoCB statically: mapped every pass
	if (!pass(0, reach, sceneSRV, m1, ssMask[0], hw, hh)) return sceneSRV;   // mask
	// Radial blur, SHORT reach first: 32 taps over one long-tap's length box-filter the source, so
	// the long pass then samples a continuous smear (the other order left a comb: a 1-degree disc
	// is narrower than a long tap, and the rays came out dashed).
	pass(1, reach / 32.0f, sceneSRV, m0, ssMask[1], hw, hh);
	pass(1, reach, sceneSRV, m1, ssMask[0], hw, hh);
	pass(2, reach, sceneSRV, m0, ssOut, w, h);                                // composite
	context->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE);
	return ssOut->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
}

// endCamera: the fogged scene (own texture), or the input when this camera has no grid.
ITextureView* NukeDiligent::Impl::ApplyVolumetrics(ITextureView* sceneSRV, int w, int h)
{
	VolState* st = volCur;
	if (!st || !sceneSRV || !volApplyPipe.pso || !st->out || !st->integ || !gbufDepthSRV || debugView != 0) return sceneSRV;
	if (volApplyPipe.srcVar)   volApplyPipe.srcVar->Set(sceneSRV);
	if (volApplyPipe.depthVar) volApplyPipe.depthVar->Set(gbufDepthSRV);
	if (volApplyPipe.histVar)  volApplyPipe.histVar->Set(st->integ->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
	ITextureView* rtv = st->out->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET);
	context->SetRenderTargets(1, &rtv, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	Viewport vp; vp.TopLeftX = 0; vp.TopLeftY = 0; vp.Width = (float)w; vp.Height = (float)h; vp.MinDepth = 0; vp.MaxDepth = 1;
	context->SetViewports(1, &vp, w, h);
	context->SetPipelineState(volApplyPipe.pso);
	context->CommitShaderResources(volApplyPipe.srb, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	DrawAttribs da{3, DRAW_FLAG_VERIFY_STATES};
	context->Draw(da);
	context->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE);
	volCur = nullptr;   // one composite per camera pass
	return st->out->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
}
