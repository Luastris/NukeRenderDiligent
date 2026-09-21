#include "NukeDiligentImpl.h"
#include <API/Model/Time.h>

// Lens film: the wet camera lens as a renderer camera post (lensfilm.ps + lensdrops.ps). The
// film is PER CAMERA (keyed like everything else by camKey) and has two sources - a soak map
// a module injects for this camera pass (the water's waterline band, LensFilmInject) and
// rain by rate (iRender::setLensRain, the weather) - so neither module depends on the other
// and the lens looks the same whoever wets it. Runs after the module camera-post hooks (the
// water's underwater composite feeds it), before the user post chain.

using namespace Diligent;

void NukeDiligent::setLensRain(float rate, float amount, float drainSeconds)
{
	m_impl->lensRainRate = rate < 0.f ? 0.f : rate;
	m_impl->lensRainAmount = amount < 0.f ? 0.f : amount;
	m_impl->lensRainDrain = drainSeconds < 0.2f ? 0.2f : drainSeconds;
	m_impl->lensRainStamp = m_impl->frameId;
}

// The two pipes compile off the draw path (EnqueueBuild); the pass passes through until then.
void NukeDiligent::Impl::EnsureLensPipes()
{
	if ((lensFilmPSO && lensDropsPSO) || lensFailed || lensBuilding.exchange(true)) return;
	EnqueueBuild([this]
	{
		std::string vs = shaderSource("post.vs"), fps = shaderSource("lensfilm.ps"), dps = shaderSource("lensdrops.ps");
		if (vs.empty() || fps.empty() || dps.empty()) { lensFailed = true; return; }
		if (!lensCB)
		{
			BufferDesc cbd; cbd.Name = "LensCB"; cbd.Size = sizeof(float) * 12;
			cbd.Usage = USAGE_DYNAMIC; cbd.BindFlags = BIND_UNIFORM_BUFFER; cbd.CPUAccessFlags = CPU_ACCESS_WRITE;
			device->CreateBuffer(cbd, nullptr, &lensCB);
			if (!lensCB) { lensFailed = true; return; }
		}
		auto make = [&](const char* dbg, const std::string& ps, TEXTURE_FORMAT fmt, std::initializer_list<const char*> texVars,
		                RefCntAutoPtr<IPipelineState>& pso, RefCntAutoPtr<IShaderResourceBinding>& srb) -> bool
		{
			ShaderCreateInfo sci; sci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
			sci.pShaderSourceStreamFactory = ShaderFactory();
			RefCntAutoPtr<IShader> v, p;
			sci.Desc = {"Lens VS", SHADER_TYPE_VERTEX, true}; sci.Source = vs.c_str(); CreateShaderCached(sci, &v);
			sci.Desc = {dbg, SHADER_TYPE_PIXEL, true};       sci.Source = ps.c_str(); CreateShaderCached(sci, &p);
			if (!v || !p) return false;
			GraphicsPipelineStateCreateInfo ci; ci.PSODesc.Name = dbg;
			auto& gp = ci.GraphicsPipeline;
			gp.NumRenderTargets = 1; gp.RTVFormats[0] = fmt; gp.DSVFormat = TEX_FORMAT_UNKNOWN;
			gp.PrimitiveTopology = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; gp.RasterizerDesc.CullMode = CULL_MODE_NONE;
			gp.DepthStencilDesc.DepthEnable = False; gp.SmplDesc.Count = 1; gp.InputLayout.NumElements = 0;
			ci.pVS = v; ci.pPS = p;
			SamplerDesc clampS; clampS.MinFilter = FILTER_TYPE_LINEAR; clampS.MagFilter = FILTER_TYPE_LINEAR; clampS.MipFilter = FILTER_TYPE_LINEAR;
			clampS.AddressU = TEXTURE_ADDRESS_CLAMP; clampS.AddressV = TEXTURE_ADDRESS_CLAMP; clampS.AddressW = TEXTURE_ADDRESS_CLAMP;
			std::vector<ShaderResourceVariableDesc> vars; std::vector<ImmutableSamplerDesc> imms;
			for (const char* n : texVars) { vars.push_back({SHADER_TYPE_PIXEL, n, SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC}); imms.push_back({SHADER_TYPE_PIXEL, n, clampS}); }
			ci.PSODesc.ResourceLayout.Variables = vars.data(); ci.PSODesc.ResourceLayout.NumVariables = (Uint32)vars.size();
			ci.PSODesc.ResourceLayout.ImmutableSamplers = imms.data(); ci.PSODesc.ResourceLayout.NumImmutableSamplers = (Uint32)imms.size();
			ci.PSODesc.ResourceLayout.DefaultVariableType = SHADER_RESOURCE_VARIABLE_TYPE_STATIC;
			CreateGraphicsPipelineStateCached(ci, &pso);
			if (!pso) return false;
			if (auto* c = pso->GetStaticVariableByName(SHADER_TYPE_PIXEL, "LensCB")) c->Set(lensCB);
			pso->CreateShaderResourceBinding(&srb, true);
			return srb != nullptr;
		};
		if (!make("Lens Film PS", fps, TEX_FORMAT_R16_FLOAT, {"g_Prev", "g_Inject"}, lensFilmPSO, lensFilmSRB) ||
		    !make("Lens Drops PS", dps, HDR_FMT, {"g_Scene", "g_Wet"}, lensDropsPSO, lensDropsSRB))
		{ lensFailed = true; std::cout << "[NukeDiligent]\tlens film pipelines FAILED" << std::endl; }
		else std::cout << "[NukeDiligent]\tlens film ready" << std::endl;
	}, [this] { lensBuilding = false; }, kPrioExtra, "lens film");
}

// A module wets this camera pass's lens: `soakSRV` = per-pixel soak per second (R16F), amount =
// the composite strength, drain = seconds the film takes to run off.
void NukeDiligent::Impl::LensInject(ITextureView* soakSRV, float amount, float drainSeconds)
{
	lensInjectSRV = soakSRV; lensInjectAmount = amount; lensInjectDrain = drainSeconds;
	lensInjectOn = soakSRV != nullptr && amount > 0.001f;
}

ITextureView* NukeDiligent::Impl::RunLensFilm(ITextureView* sceneSRV)
{
	const bool injOn  = lensInjectOn;
	const bool rainOn = (frameId - lensRainStamp) <= 1 && lensRainRate > 0.001f && lensRainAmount > 0.001f;
	lensInjectOn = false;   // consumed by this pass only
	// Cameras drop out of the map once their film has run off and they stopped rendering.
	for (auto it = lensCams.begin(); it != lensCams.end();)
	{
		if (it->second.alive <= 0.0f && it->second.lastUsed + 300 < frameId)
		{ for (auto& t : it->second.tex) if (t) Trash(t); if (it->second.out) Trash(it->second.out); it = lensCams.erase(it); }
		else ++it;
	}
	auto found = lensCams.find(curCamKey);
	if (!injOn && !rainOn && (found == lensCams.end() || found->second.alive <= 0.0f)) return nullptr;
	if (!sceneSRV || curRTW <= 0 || curRTH <= 0) return nullptr;
	EnsureLensPipes();
	if (!lensFilmPSO || !lensDropsPSO || !lensFilmSRB || !lensDropsSRB || !lensCB) return nullptr;

	LensCam& c = (found != lensCams.end()) ? found->second : lensCams[curCamKey];
	const float drain  = injOn ? std::max(lensInjectDrain, 0.2f) : std::max(lensRainDrain, 0.2f);
	const float amount = std::max(injOn ? lensInjectAmount : 0.0f, rainOn ? lensRainAmount : 0.0f);
	if (injOn || rainOn) { c.alive = drain * 6.5f + 0.5f; c.amount = amount; c.drain = drain; }   // long enough for the slowest bead
	c.lastUsed = frameId;
	// The film runs on the GAME clock (pauses freeze it), accumulated once per frame.
	if (lensClockFrame != frameId) { lensClockFrame = frameId; lensClock += std::max(Time::getSingleton()->gameDelta, 0.0); }
	const double now = lensClock;
	float dt = (c.lastTime >= 0.0) ? (float)std::max(now - c.lastTime, 0.0) : 0.0f;
	c.lastTime = now;
	dt = std::min(dt, 0.1f);
	if (!injOn && !rainOn) c.alive -= dt;

	bool fresh = false;
	auto ensure = [&](RefCntAutoPtr<ITexture>& t, const char* name, TEXTURE_FORMAT f)
	{
		if (t) { const auto& d = t->GetDesc(); if ((int)d.Width != curRTW || (int)d.Height != curRTH) { Trash(t); t.Release(); } }
		if (!t)
		{
			TextureDesc td; td.Name = name; td.Type = RESOURCE_DIM_TEX_2D; td.Width = (Uint32)curRTW; td.Height = (Uint32)curRTH;
			td.Format = f; td.MipLevels = 1; td.BindFlags = BIND_RENDER_TARGET | BIND_SHADER_RESOURCE;
			device->CreateTexture(td, nullptr, &t);
			if (t && f == TEX_FORMAT_R16_FLOAT) fresh = true;
		}
		return t != nullptr;
	};
	if (!ensure(c.tex[0], "Lens film A", TEX_FORMAT_R16_FLOAT) || !ensure(c.tex[1], "Lens film B", TEX_FORMAT_R16_FLOAT)
	    || !ensure(c.out, "Lens film scene", HDR_FMT)) return nullptr;
	if (fresh)
	{
		const float zero[4] = {0, 0, 0, 0};
		for (int k = 0; k < 2; ++k)
		{
			ITextureView* rt = c.tex[k]->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET);
			context->SetRenderTargets(1, &rt, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
			context->ClearRenderTarget(rt, zero, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		}
	}
	{
		struct CB { float l0[4], l1[4], l2[4]; };
		MapHelper<CB> cb(context, lensCB, MAP_WRITE, MAP_FLAG_DISCARD);
		if (cb == nullptr) return nullptr;
		cb->l0[0] = dt; cb->l0[1] = 1.0f / c.drain; cb->l0[2] = (float)std::fmod(now, 3600.0); cb->l0[3] = injOn ? 1.f : 0.f;
		cb->l1[0] = rainOn ? lensRainRate : 0.f; cb->l1[1] = rainOn ? lensRainAmount : 0.f; cb->l1[2] = (float)curRTW; cb->l1[3] = (float)curRTH;
		cb->l2[0] = 1.0f / (float)curRTW; cb->l2[1] = 1.0f / (float)curRTH; cb->l2[2] = c.amount; cb->l2[3] = 0.f;
	}
	Viewport vp; vp.TopLeftX = 0; vp.TopLeftY = 0; vp.Width = (float)curRTW; vp.Height = (float)curRTH; vp.MinDepth = 0; vp.MaxDepth = 1;
	ITextureView* white = whiteTex ? whiteTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE) : nullptr;
	// 1) the mask: prev -> next
	const int nxt = c.cur ^ 1;
	{
		if (auto* v = lensFilmSRB->GetVariableByName(SHADER_TYPE_PIXEL, "g_Prev"))   v->Set(c.tex[c.cur]->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
		if (auto* v = lensFilmSRB->GetVariableByName(SHADER_TYPE_PIXEL, "g_Inject")) v->Set(injOn ? lensInjectSRV : white);
		ITextureView* rtv = c.tex[nxt]->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET);
		context->SetRenderTargets(1, &rtv, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		context->SetViewports(1, &vp, curRTW, curRTH);
		context->SetPipelineState(lensFilmPSO);
		context->CommitShaderResources(lensFilmSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		DrawAttribs da{3, DRAW_FLAG_VERIFY_STATES};
		context->Draw(da);
	}
	c.cur = nxt;
	// 2) the composite: the scene through the film
	{
		if (auto* v = lensDropsSRB->GetVariableByName(SHADER_TYPE_PIXEL, "g_Scene")) v->Set(sceneSRV);
		if (auto* v = lensDropsSRB->GetVariableByName(SHADER_TYPE_PIXEL, "g_Wet"))   v->Set(c.tex[c.cur]->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
		ITextureView* rtv = c.out->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET);
		context->SetRenderTargets(1, &rtv, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		context->SetViewports(1, &vp, curRTW, curRTH);
		context->SetPipelineState(lensDropsPSO);
		context->CommitShaderResources(lensDropsSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		DrawAttribs da{3, DRAW_FLAG_VERIFY_STATES};
		context->Draw(da);
	}
	return c.out->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
}
