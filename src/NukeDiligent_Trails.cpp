// W5 ground trails: the carve map. A camera-following window (kTrailExtent m, kTrailN texels
// per side, R8 ping-pong) holding how much of the accumulated layer (snow/sand) movers have
// pressed out per texel. Each live frame (the sky-occlusion hook): the previous map is carried
// over by a texel shift as the window follows the camera and fresh fall fills it (fill/s), then
// this frame's footprints (iRender::setGroundTrails, from the engine's grounded movers) are
// MAX-stamped as soft discs. world.ps / gbuffer.ps / the domain shaders read it (trails.hlsli).
#include "NukeDiligentImpl.h"
#include <API/Model/Surface.h>
#include <API/Model/Time.h>

using namespace Diligent;

namespace {
const int   kTrailN      = 1024;
const float kTrailExtent = 64.0f;   // m per side: 6.25 cm texels
const int   kTrailMaxImp = 256;     // the stamp buffer (nearest first)
}

void NukeDiligent::setGroundTrails(const float* xzrw, int count, float fillPerSec)
{
	Impl* d = m_impl;
	d->trailImprints.clear();
	if (xzrw && count > 0)
	{
		const int n = std::min(count, kTrailMaxImp);
		d->trailImprints.assign(xzrw, xzrw + (size_t)n * 4);
	}
	d->trailFill = fillPerSec < 0.f ? 0.f : fillPerSec;
}

void NukeDiligent::Impl::EnsureTrailPipes()
{
	if ((trailShiftPSO && trailStampPSO) || trailFailed || trailBuilding.exchange(true)) return;
	EnqueueBuild([this]
	{
		std::string pvs = shaderSource("post.vs"), sps = shaderSource("trails_shift.ps");
		std::string svs = shaderSource("trails_stamp.vs"), tps = shaderSource("trails_stamp.ps");
		if (pvs.empty() || sps.empty() || svs.empty() || tps.empty()) { trailFailed = true; return; }
		if (!trailSimCB)
		{
			BufferDesc cbd; cbd.Name = "TrailSimCB"; cbd.Size = 32 + 16 * kTrailMaxImp;
			cbd.Usage = USAGE_DYNAMIC; cbd.BindFlags = BIND_UNIFORM_BUFFER; cbd.CPUAccessFlags = CPU_ACCESS_WRITE;
			device->CreateBuffer(cbd, nullptr, &trailSimCB);
			if (!trailSimCB) { trailFailed = true; return; }
		}
		auto make = [&](const char* dbg, const std::string& vsrc, const std::string& psrc, bool stamp,
		                RefCntAutoPtr<IPipelineState>& pso, RefCntAutoPtr<IShaderResourceBinding>& srb) -> bool
		{
			ShaderCreateInfo sci; sci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
			sci.pShaderSourceStreamFactory = ShaderFactory();
			RefCntAutoPtr<IShader> v, p;
			sci.Desc = {"Trails VS", SHADER_TYPE_VERTEX, true}; sci.Source = vsrc.c_str(); CreateShaderCached(sci, &v);
			sci.Desc = {dbg, SHADER_TYPE_PIXEL, true};         sci.Source = psrc.c_str(); CreateShaderCached(sci, &p);
			if (!v || !p) return false;
			GraphicsPipelineStateCreateInfo ci; ci.PSODesc.Name = dbg;
			auto& gp = ci.GraphicsPipeline;
			gp.NumRenderTargets = 1; gp.RTVFormats[0] = TEX_FORMAT_R8_UNORM; gp.DSVFormat = TEX_FORMAT_UNKNOWN;
			gp.PrimitiveTopology = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; gp.RasterizerDesc.CullMode = CULL_MODE_NONE;
			gp.DepthStencilDesc.DepthEnable = False; gp.SmplDesc.Count = 1; gp.InputLayout.NumElements = 0;
			if (stamp)   // footprints MAX into the carried map
			{
				auto& rt = gp.BlendDesc.RenderTargets[0];
				rt.BlendEnable = True; rt.SrcBlend = BLEND_FACTOR_ONE; rt.DestBlend = BLEND_FACTOR_ONE; rt.BlendOp = BLEND_OPERATION_MAX;
				rt.SrcBlendAlpha = BLEND_FACTOR_ONE; rt.DestBlendAlpha = BLEND_FACTOR_ONE; rt.BlendOpAlpha = BLEND_OPERATION_MAX;
			}
			ci.pVS = v; ci.pPS = p;
			std::vector<ShaderResourceVariableDesc> vars;
			if (!stamp) vars.push_back({SHADER_TYPE_PIXEL, "g_Prev", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC});
			ci.PSODesc.ResourceLayout.Variables = vars.empty() ? nullptr : vars.data(); ci.PSODesc.ResourceLayout.NumVariables = (Uint32)vars.size();
			ci.PSODesc.ResourceLayout.DefaultVariableType = SHADER_RESOURCE_VARIABLE_TYPE_STATIC;
			CreateGraphicsPipelineStateCached(ci, &pso);
			if (!pso) return false;
			if (auto* c = pso->GetStaticVariableByName(SHADER_TYPE_PIXEL,  "TrailSimCB")) c->Set(trailSimCB);
			if (auto* c = pso->GetStaticVariableByName(SHADER_TYPE_VERTEX, "TrailSimCB")) c->Set(trailSimCB);
			pso->CreateShaderResourceBinding(&srb, true);
			return srb != nullptr;
		};
		if (!make("Trails Shift PS", pvs, sps, false, trailShiftPSO, trailShiftSRB) ||
		    !make("Trails Stamp PS", svs, tps, true, trailStampPSO, trailStampSRB))
		{ trailFailed = true; std::cout << "[NukeDiligent]\ttrail pipelines FAILED" << std::endl; }
		else std::cout << "[NukeDiligent]\tground trails ready" << std::endl;
	}, [this] { trailBuilding = false; }, kPrioExtra, "ground trails");
}

void NukeDiligent::Impl::RunTrails()
{
	trailSRV = nullptr;
	auto off = [&]
	{
		trailValid = false; trailImprints.clear();
		if (!trailsCBZero && trailsCB) { const float z[4] = {}; context->UpdateBuffer(trailsCB, 0, 16, z, RESOURCE_STATE_TRANSITION_MODE_TRANSITION); trailsCBZero = true; }
	};
	if (!nuke::Surface::AnyConditionFromSky() || !trailsCB) { off(); return; }
	EnsureTrailPipes();
	if (!trailShiftPSO || !trailStampPSO || !trailShiftSRB || !trailStampSRB) { off(); return; }
	for (int k = 0; k < 2; ++k)
		if (!trailTex[k])
		{
			TextureDesc td; td.Name = k ? "Ground Trails B" : "Ground Trails A"; td.Type = RESOURCE_DIM_TEX_2D;
			td.Width = td.Height = (Uint32)kTrailN; td.Format = TEX_FORMAT_R8_UNORM; td.MipLevels = 1;
			td.BindFlags = BIND_RENDER_TARGET | BIND_SHADER_RESOURCE;
			device->CreateTexture(td, nullptr, &trailTex[k]);
			if (!trailTex[k]) { off(); return; }
			trailValid = false;
		}
	GpuPass("trails");
	// Window: ahead of the camera along its view, snapped to texels (tracks must not swim).
	const float size = kTrailExtent, texel = size / (float)kTrailN;
	const float fl = std::sqrt(curCamFwd[0] * curCamFwd[0] + curCamFwd[2] * curCamFwd[2]);
	const float aim = size * 0.3f;
	const float cx = curCamPos[0] + (fl > 1e-3f ? curCamFwd[0] / fl * aim : 0.0f);
	const float cz = curCamPos[2] + (fl > 1e-3f ? curCamFwd[2] / fl * aim : 0.0f);
	const float nx = std::floor((cx - size * 0.5f) / texel + 0.5f) * texel;
	const float nz = std::floor((cz - size * 0.5f) / texel + 0.5f) * texel;
	float shiftX = 0.0f, shiftZ = 0.0f;
	if (trailValid)
	{
		shiftX = (nx - trailOrigin[0]) / texel; shiftZ = (nz - trailOrigin[1]) / texel;
		if (std::fabs(shiftX) >= kTrailN || std::fabs(shiftZ) >= kTrailN) { shiftX = shiftZ = 0.0f; trailValid = false; }
	}
	trailOrigin[0] = nx; trailOrigin[1] = nz;
	// Game clock: fill by game time, once per frame.
	float dt = 0.0f;
	if (trailClockFrame != frameId) { trailClockFrame = frameId; dt = (float)std::min(std::max(Time::getSingleton()->gameDelta, 0.0), 0.25); }
	const int count = (int)(trailImprints.size() / 4);
	{
		MapHelper<float> cb(context, trailSimCB, MAP_WRITE, MAP_FLAG_DISCARD);
		cb[0] = shiftX; cb[1] = shiftZ; cb[2] = trailFill * dt; cb[3] = trailValid ? 1.0f : 0.0f;
		cb[4] = nx; cb[5] = nz; cb[6] = 1.0f / size; cb[7] = (float)kTrailN;
		if (count > 0) memcpy(&cb[8], trailImprints.data(), sizeof(float) * 4 * (size_t)count);
	}
	const int next = trailCur ^ 1;
	ITextureView* rtv = trailTex[next]->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET);
	context->SetRenderTargets(1, &rtv, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	Viewport vp; vp.TopLeftX = 0; vp.TopLeftY = 0; vp.Width = (float)kTrailN; vp.Height = (float)kTrailN; vp.MinDepth = 0; vp.MaxDepth = 1;
	context->SetViewports(1, &vp, kTrailN, kTrailN);
	if (auto* v = trailShiftSRB->GetVariableByName(SHADER_TYPE_PIXEL, "g_Prev"))
		v->Set(trailTex[trailCur]->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
	context->SetPipelineState(trailShiftPSO);
	context->CommitShaderResources(trailShiftSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	{ DrawAttribs da{ 3, DRAW_FLAG_VERIFY_STATES }; context->Draw(da); }
	if (count > 0)
	{
		context->SetPipelineState(trailStampPSO);
		context->CommitShaderResources(trailStampSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		DrawAttribs da{ 6, DRAW_FLAG_VERIFY_STATES }; da.NumInstances = (Uint32)count; context->Draw(da);
	}
	context->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE);
	trailCur = next; trailValid = true; trailImprints.clear();
	trailSRV = trailTex[trailCur]->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
	const float p[4] = { nx, nz, 1.0f / size, 1.0f };
	context->UpdateBuffer(trailsCB, 0, 16, p, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	trailsCBZero = false;
}
