#include "NukeDiligentImpl.h"


void NukeDiligent::Impl::CreateSkyResources()
{
	skyPSO.Release(); skySRB.Release(); skyCB.Release();   // rebuild-safe (MSAA change re-calls this)
	std::string vs = shaderSource("sky.vs"), ps = shaderSource("sky.ps");
	if (vs.empty() || ps.empty()) { cout << "[NukeDiligent]\tsky shaders missing" << endl; return; }
	ShaderCreateInfo sci; sci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
	RefCntAutoPtr<IShader> v, p;
	sci.Desc = {"Sky VS", SHADER_TYPE_VERTEX, true}; sci.Source = vs.c_str(); CreateShaderCached(sci, &v);
	sci.Desc = {"Sky PS", SHADER_TYPE_PIXEL, true};  sci.Source = ps.c_str(); CreateShaderCached(sci, &p);
	if (!v || !p) return;

	BufferDesc cbd; cbd.Name = "SkyCB"; cbd.Size = sizeof(float4x4) + sizeof(float) * 4 * 9;   // InvVP + 9 float4
	cbd.Usage = USAGE_DYNAMIC; cbd.BindFlags = BIND_UNIFORM_BUFFER; cbd.CPUAccessFlags = CPU_ACCESS_WRITE;
	device->CreateBuffer(cbd, nullptr, &skyCB);

	GraphicsPipelineStateCreateInfo ci; ci.PSODesc.Name = "Sky PSO";
	auto& gp = ci.GraphicsPipeline;
	gp.NumRenderTargets = 1; gp.RTVFormats[0] = SceneFmt();   // sky draws into the scene target
	gp.DSVFormat = TEX_FORMAT_D32_FLOAT;   // a depth buffer is bound in the camera pass; match it (test off)
	gp.PrimitiveTopology = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	gp.RasterizerDesc.CullMode = CULL_MODE_NONE;
	gp.DepthStencilDesc.DepthEnable = False;
	gp.DepthStencilDesc.DepthWriteEnable = False;
	gp.SmplDesc.Count = samples;   // MSAA: sky draws into the MS camera target
	gp.InputLayout.NumElements = 0;   // fullscreen triangle from SV_VertexID
	ShaderResourceVariableDesc svars[] = {
		{SHADER_TYPE_PIXEL, "g_StarTex", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
		{SHADER_TYPE_PIXEL, "g_MoonTex", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
	};
	ci.PSODesc.ResourceLayout.Variables = svars; ci.PSODesc.ResourceLayout.NumVariables = 2;
	SamplerDesc ssamp; ssamp.MinFilter = FILTER_TYPE_LINEAR; ssamp.MagFilter = FILTER_TYPE_LINEAR; ssamp.MipFilter = FILTER_TYPE_LINEAR;
	ssamp.AddressU = TEXTURE_ADDRESS_WRAP; ssamp.AddressV = TEXTURE_ADDRESS_CLAMP;
	SamplerDesc msamp; msamp.MinFilter = FILTER_TYPE_LINEAR; msamp.MagFilter = FILTER_TYPE_LINEAR; msamp.MipFilter = FILTER_TYPE_LINEAR;
	msamp.AddressU = TEXTURE_ADDRESS_CLAMP; msamp.AddressV = TEXTURE_ADDRESS_CLAMP;
	ImmutableSamplerDesc simm[] = {
		{SHADER_TYPE_PIXEL, "g_StarTex", ssamp},
		{SHADER_TYPE_PIXEL, "g_MoonTex", msamp},
	};
	ci.PSODesc.ResourceLayout.ImmutableSamplers = simm; ci.PSODesc.ResourceLayout.NumImmutableSamplers = 2;
	ci.pVS = v; ci.pPS = p;
	CreateGraphicsPipelineStateCached(ci, &skyPSO);
	if (skyPSO)
	{
		if (auto* sv = skyPSO->GetStaticVariableByName(SHADER_TYPE_PIXEL, "SkyCB")) sv->Set(skyCB);
		skyPSO->CreateShaderResourceBinding(&skySRB, true);
		skyStarVar = skySRB->GetVariableByName(SHADER_TYPE_PIXEL, "g_StarTex");
		skyMoonVar = skySRB->GetVariableByName(SHADER_TYPE_PIXEL, "g_MoonTex");
	}
	cout << "[NukeDiligent]\tsky pipeline" << (skyPSO ? " ready" : " FAILED") << endl;
}

void NukeDiligent::Impl::DrawSky()
{
	if (!skyPSO || sky.mode != 1) return;
	float4x4 invVP = (curView * curProj).Inverse();
	ITextureView* starSRV = sky.starsTex ? GetTexSRV(sky.starsTex) : nullptr;
	ITextureView* moonSRV = (sky.moonTex && sky.moonAmount > 0.0f) ? GetTexSRV(sky.moonTex) : nullptr;
	struct SkyData { float4x4 invVP; float camPos[4]; float top[4]; float horizon[4]; float ground[4]; float params[4]; float sunDir[4]; float sunCol[4]; float moonDir[4]; float moonParams[4]; };
	{
		MapHelper<SkyData> cb(context, skyCB, MAP_WRITE, MAP_FLAG_DISCARD);
		cb->invVP = invVP;
		cb->camPos[0] = curCamPos[0]; cb->camPos[1] = curCamPos[1]; cb->camPos[2] = curCamPos[2]; cb->camPos[3] = 1;
		for (int k = 0; k < 3; ++k) { cb->top[k] = sky.top[k]; cb->horizon[k] = sky.horizon[k]; cb->ground[k] = sky.ground[k]; cb->sunDir[k] = sky.sunDir[k]; cb->sunCol[k] = sky.sunColor[k]; cb->moonDir[k] = sky.moonDir[k]; }
		cb->top[3] = cb->horizon[3] = cb->ground[3] = cb->sunDir[3] = cb->sunCol[3] = cb->moonDir[3] = 1;
		cb->params[0] = sky.skyIntensity; cb->params[1] = sky.sunIntensity; cb->params[2] = sky.stars;
		cb->params[3] = starSRV ? 1.0f : 0.0f;   // has a star texture (else procedural)
		cb->moonParams[0] = moonSRV ? sky.moonAmount : 0.0f; cb->moonParams[1] = sky.moonSize; cb->moonParams[2] = sky.moonPhase;
		cb->moonParams[3] = hdr ? 0.0f : 1.0f;   // HDR off: sky tonemaps itself (RGBA8 scene, post is passthrough)
	}
	if (skyStarVar) skyStarVar->Set(starSRV ? starSRV : whiteTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
	if (skyMoonVar) skyMoonVar->Set(moonSRV ? moonSRV : whiteTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
	context->SetPipelineState(skyPSO);
	context->CommitShaderResources(skySRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	DrawAttribs da{3, DRAW_FLAG_VERIFY_STATES};
	context->Draw(da);
}
