#include "NukeDiligentImpl.h"
#include <cstring>

// Sprite pipeline: alpha-blended unlit textured quads, one texture per draw, drawn in the
// camera pass (SceneFmt + MSAA) with depth test but no depth write.

// Build the unlit, screen and lit sprite PSOs and their SRBs/constant buffers.
void NukeDiligent::Impl::CreateSpriteResources()
{
	ReloadScope reloadScope("sprite");
	spritePSO.Release(); spriteSRB.Release(); spriteCB.Release(); spriteTexVar = nullptr;
	spriteScreenPSO.Release(); spriteScreenSRB.Release(); spriteScreenTexVar = nullptr;
	spriteScreenPSOBB.Release(); spriteScreenSRBBB.Release(); spriteScreenTexVarBB = nullptr;
	std::string vs = shaderSource("sprite.vs"), ps = shaderSource("sprite.ps");
	if (vs.empty() || ps.empty()) { std::cout << "[NukeDiligent]\tsprite shaders missing" << std::endl; return; }

	auto sf = ShaderFactory();   // sprite.ps / sprite_six.ps include vol.hlsli
	ShaderCreateInfo sci; sci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL; sci.pShaderSourceStreamFactory = sf;
	RefCntAutoPtr<IShader> v, p;
	sci.Desc = {"Sprite VS", SHADER_TYPE_VERTEX, true}; sci.Source = vs.c_str(); CreateShaderCached(sci, &v);
	sci.Desc = {"Sprite PS", SHADER_TYPE_PIXEL, true};  sci.Source = ps.c_str(); CreateShaderCached(sci, &p);
	if (!v || !p) return;

	BufferDesc cbd; cbd.Name = "SpriteCB"; cbd.Size = sizeof(SpriteCBData);   // VP + g_Soft + g_Soft2 + g_Sdf + g_Outline + g_Clip
	cbd.Usage = USAGE_DYNAMIC; cbd.BindFlags = BIND_UNIFORM_BUFFER; cbd.CPUAccessFlags = CPU_ACCESS_WRITE;
	device->CreateBuffer(cbd, nullptr, &spriteCB);
	// The vertex buffer is created/grown on demand in FlushSprites (it survives PSO rebuilds).

	GraphicsPipelineStateCreateInfo ci; ci.PSODesc.Name = "Sprite PSO";
	auto& gp = ci.GraphicsPipeline;
	gp.NumRenderTargets = 1; gp.RTVFormats[0] = SceneFmt();   // composites into the (MS) HDR camera target
	gp.DSVFormat = TEX_FORMAT_D32_FLOAT;
	gp.PrimitiveTopology = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	gp.RasterizerDesc.CullMode = CULL_MODE_NONE;             // double-sided
	gp.DepthStencilDesc.DepthEnable = True;                  // occluded by opaque geometry...
	gp.DepthStencilDesc.DepthWriteEnable = False;            // ...but transparent: no depth write
	gp.SmplDesc.Count = samples;                            // MSAA: match the camera target
	auto& rt = gp.BlendDesc.RenderTargets[0];
	rt.BlendEnable   = True;
	rt.SrcBlend      = BLEND_FACTOR_SRC_ALPHA; rt.DestBlend      = BLEND_FACTOR_INV_SRC_ALPHA; rt.BlendOp      = BLEND_OPERATION_ADD;
	rt.SrcBlendAlpha = BLEND_FACTOR_ONE;       rt.DestBlendAlpha = BLEND_FACTOR_INV_SRC_ALPHA; rt.BlendOpAlpha = BLEND_OPERATION_ADD;

	LayoutElement layout[] = {
		{0, 0, 3, VT_FLOAT32, False},   // pos
		{1, 0, 2, VT_FLOAT32, False},   // uv
		{2, 0, 4, VT_FLOAT32, False},   // tint
	};
	gp.InputLayout.LayoutElements = layout; gp.InputLayout.NumElements = 3;
	ci.pVS = v; ci.pPS = p;

	// g_Sprite: dynamic PS texture + a linear-clamp immutable sampler (combined-sampler convention).
	// g_VolInteg / g_VolLight: this camera's froxel grid (own-column fog + grid light), white3D when off.
	ShaderResourceVariableDesc vars[] = { {SHADER_TYPE_PIXEL, "g_Sprite",     SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
	                                      {SHADER_TYPE_PIXEL, "g_SceneDepth", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},    // soft particles / fog depth (Load — no sampler)
	                                      {SHADER_TYPE_PIXEL, "g_VolInteg",   SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
	                                      {SHADER_TYPE_PIXEL, "g_VolLight",   SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
	                                      {SHADER_TYPE_PIXEL, "g_Mask",       SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC} };   // alpha mask (setSpriteMask)
	SamplerDesc samp;
	samp.MinFilter = FILTER_TYPE_LINEAR; samp.MagFilter = FILTER_TYPE_LINEAR; samp.MipFilter = FILTER_TYPE_LINEAR;
	samp.AddressU = TEXTURE_ADDRESS_CLAMP; samp.AddressV = TEXTURE_ADDRESS_CLAMP; samp.AddressW = TEXTURE_ADDRESS_CLAMP;
	ImmutableSamplerDesc imms[] = { {SHADER_TYPE_PIXEL, "g_Sprite", samp}, {SHADER_TYPE_PIXEL, "g_VolInteg", samp}, {SHADER_TYPE_PIXEL, "g_VolLight", samp},
	                                {SHADER_TYPE_PIXEL, "g_Mask", samp} };
	ci.PSODesc.ResourceLayout.Variables            = vars; ci.PSODesc.ResourceLayout.NumVariables         = 5;
	ci.PSODesc.ResourceLayout.ImmutableSamplers    = imms; ci.PSODesc.ResourceLayout.NumImmutableSamplers = 4;
	ci.PSODesc.ResourceLayout.DefaultVariableType  = SHADER_RESOURCE_VARIABLE_TYPE_STATIC;

	if (!whiteTex3D)
	{
		TextureDesc td; td.Name = "White 1x1x1"; td.Type = RESOURCE_DIM_TEX_3D; td.Width = td.Height = td.Depth = 1;
		td.MipLevels = 1; td.Format = TEX_FORMAT_RGBA8_UNORM; td.BindFlags = BIND_SHADER_RESOURCE; td.Usage = USAGE_IMMUTABLE;
		const uint8_t px[4] = {255, 255, 255, 255};
		TextureSubResData sub; sub.pData = px; sub.Stride = 4; sub.DepthStride = 4;
		TextureData init; init.pSubResources = &sub; init.NumSubresources = 1;
		device->CreateTexture(td, &init, &whiteTex3D);
	}
	ITextureView* white3 = whiteTex3D ? whiteTex3D->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE) : nullptr;

	// Coverage mask PSO: the sprite VS with sprite_cover.ps into an R8 target, depth-tested
	// against the single-sample G-buffer depth, union blend (ONE / INV_SRC_ALPHA).
	{
		coverPSO.Release(); coverSRB.Release(); coverTexVar = nullptr;
		std::string cps = shaderSource("sprite_cover.ps");
		RefCntAutoPtr<IShader> cp;
		if (!cps.empty()) { sci.Desc = {"Sprite Cover PS", SHADER_TYPE_PIXEL, true}; sci.Source = cps.c_str(); CreateShaderCached(sci, &cp); }
		if (cp)
		{
			GraphicsPipelineStateCreateInfo c2 = ci; c2.PSODesc.Name = "Sprite Cover PSO";
			auto& g2 = c2.GraphicsPipeline;
			g2.NumRenderTargets = 1; g2.RTVFormats[0] = TEX_FORMAT_R8_UNORM; g2.DSVFormat = TEX_FORMAT_D32_FLOAT;
			g2.SmplDesc.Count = 1;
			g2.DepthStencilDesc.DepthEnable = True; g2.DepthStencilDesc.DepthWriteEnable = False;
			auto& r2 = g2.BlendDesc.RenderTargets[0];
			r2.BlendEnable = True;
			r2.SrcBlend = BLEND_FACTOR_ONE; r2.DestBlend = BLEND_FACTOR_INV_SRC_ALPHA; r2.BlendOp = BLEND_OPERATION_ADD;
			r2.SrcBlendAlpha = BLEND_FACTOR_ONE; r2.DestBlendAlpha = BLEND_FACTOR_INV_SRC_ALPHA; r2.BlendOpAlpha = BLEND_OPERATION_ADD;
			c2.pPS = cp;
			ShaderResourceVariableDesc cvars[] = { {SHADER_TYPE_PIXEL, "g_Sprite", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
			                                       {SHADER_TYPE_PIXEL, "g_Mask",   SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC} };
			ImmutableSamplerDesc cimms[] = { {SHADER_TYPE_PIXEL, "g_Sprite", samp}, {SHADER_TYPE_PIXEL, "g_Mask", samp} };
			c2.PSODesc.ResourceLayout.Variables = cvars; c2.PSODesc.ResourceLayout.NumVariables = 2;
			c2.PSODesc.ResourceLayout.ImmutableSamplers = cimms; c2.PSODesc.ResourceLayout.NumImmutableSamplers = 2;
			CreateGraphicsPipelineStateCached(c2, &coverPSO);
			if (coverPSO)
			{
				if (auto* sv = coverPSO->GetStaticVariableByName(SHADER_TYPE_VERTEX, "SpriteCB")) sv->Set(spriteCB);
				coverPSO->CreateShaderResourceBinding(&coverSRB, true);
				coverTexVar = coverSRB ? coverSRB->GetVariableByName(SHADER_TYPE_PIXEL, "g_Sprite") : nullptr;
				coverMaskVar = coverSRB ? coverSRB->GetVariableByName(SHADER_TYPE_PIXEL, "g_Mask") : nullptr;
			}
		}
		if (!coverZeroTex)
		{
			TextureDesc td; td.Name = "Cover zero 1x1"; td.Type = RESOURCE_DIM_TEX_2D; td.Width = td.Height = 1;
			td.MipLevels = 1; td.Format = TEX_FORMAT_R8_UNORM; td.BindFlags = BIND_SHADER_RESOURCE; td.Usage = USAGE_IMMUTABLE;
			const uint8_t px[1] = {0};
			TextureSubResData sub; sub.pData = px; sub.Stride = 1;
			TextureData init; init.pSubResources = &sub; init.NumSubresources = 1;
			device->CreateTexture(td, &init, &coverZeroTex);
		}
	}

	CreateGraphicsPipelineStateCached(ci, &spritePSO);
	if (spritePSO)
	{
		if (auto* sv = spritePSO->GetStaticVariableByName(SHADER_TYPE_VERTEX, "SpriteCB")) sv->Set(spriteCB);
		if (auto* sp = spritePSO->GetStaticVariableByName(SHADER_TYPE_PIXEL,  "SpriteCB")) sp->Set(spriteCB);
		if (auto* sp = spritePSO->GetStaticVariableByName(SHADER_TYPE_PIXEL,  "VolCB"))    sp->Set(volCB);
		spritePSO->CreateShaderResourceBinding(&spriteSRB, true);
		if (spriteSRB) spriteTexVar   = spriteSRB->GetVariableByName(SHADER_TYPE_PIXEL, "g_Sprite");
		if (spriteSRB) spriteMaskVar  = spriteSRB->GetVariableByName(SHADER_TYPE_PIXEL, "g_Mask");
		if (spriteSRB) spriteDepthVar = spriteSRB->GetVariableByName(SHADER_TYPE_PIXEL, "g_SceneDepth");
		if (spriteSRB) spriteVolIntegVar = spriteSRB->GetVariableByName(SHADER_TYPE_PIXEL, "g_VolInteg");
		if (spriteSRB) spriteVolLightVar = spriteSRB->GetVariableByName(SHADER_TYPE_PIXEL, "g_VolLight");
	}

	// After-post screen variants: same shaders, RTV in the output format, single-sample, no depth.
	auto buildScreen = [&](TEXTURE_FORMAT fmt, const char* nm, RefCntAutoPtr<IPipelineState>& pso,
	                       RefCntAutoPtr<IShaderResourceBinding>& srb, IShaderResourceVariable*& tvar)
	{
		GraphicsPipelineStateCreateInfo si; si.PSODesc.Name = nm;
		auto& g = si.GraphicsPipeline;
		g.NumRenderTargets = 1; g.RTVFormats[0] = fmt;
		g.DSVFormat = TEX_FORMAT_UNKNOWN;
		g.PrimitiveTopology = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		g.RasterizerDesc.CullMode = CULL_MODE_NONE;
		g.DepthStencilDesc.DepthEnable = False;
		g.SmplDesc.Count = 1;
		auto& b = g.BlendDesc.RenderTargets[0];
		b.BlendEnable = True; b.SrcBlend = BLEND_FACTOR_SRC_ALPHA; b.DestBlend = BLEND_FACTOR_INV_SRC_ALPHA; b.BlendOp = BLEND_OPERATION_ADD;
		b.SrcBlendAlpha = BLEND_FACTOR_ONE; b.DestBlendAlpha = BLEND_FACTOR_INV_SRC_ALPHA; b.BlendOpAlpha = BLEND_OPERATION_ADD;
		g.InputLayout.LayoutElements = layout; g.InputLayout.NumElements = 3;
		si.pVS = v; si.pPS = p;
		si.PSODesc.ResourceLayout.Variables            = vars; si.PSODesc.ResourceLayout.NumVariables         = (Uint32)(sizeof(vars) / sizeof(vars[0]));
		si.PSODesc.ResourceLayout.ImmutableSamplers    = imms; si.PSODesc.ResourceLayout.NumImmutableSamplers = (Uint32)(sizeof(imms) / sizeof(imms[0]));
		si.PSODesc.ResourceLayout.DefaultVariableType  = SHADER_RESOURCE_VARIABLE_TYPE_STATIC;
		CreateGraphicsPipelineStateCached(si, &pso);
		if (pso)
		{
			if (auto* sv = pso->GetStaticVariableByName(SHADER_TYPE_VERTEX, "SpriteCB")) sv->Set(spriteCB);
			if (auto* sp = pso->GetStaticVariableByName(SHADER_TYPE_PIXEL,  "SpriteCB")) sp->Set(spriteCB);
			if (auto* sp = pso->GetStaticVariableByName(SHADER_TYPE_PIXEL,  "VolCB"))    sp->Set(volCB);
			pso->CreateShaderResourceBinding(&srb, true);
			if (srb) tvar = srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Sprite");
			if (srb) if (auto* dv = srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_SceneDepth"))
				dv->Set(whiteTex ? whiteTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE) : nullptr);   // screen sprites: soft/fog always off
			if (srb) if (auto* v = srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_VolInteg")) v->Set(white3);
			if (srb) if (auto* v = srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_VolLight")) v->Set(white3);
			if (srb) if (auto* v = srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Mask"))     // screen sprites: never masked
				v->Set(whiteTex ? whiteTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE) : nullptr);
		}
	};
	buildScreen(TEX_FORMAT_RGBA8_UNORM, "Sprite Screen PSO", spriteScreenPSO, spriteScreenSRB, spriteScreenTexVar);
	TEXTURE_FORMAT bbFmt = swapChain ? swapChain->GetDesc().ColorBufferFormat : TEX_FORMAT_RGBA8_UNORM;
	buildScreen(bbFmt, "Sprite Screen PSO BB", spriteScreenPSOBB, spriteScreenSRBBB, spriteScreenTexVarBB);

	// Lit variant (drawSpriteRunLit): diffuse+normal Lambert off the shared worldFrameCB.
	spriteLitPSO.Release(); spriteLitCB.Release(); spriteLitSRBs.clear();
	std::string lvs = shaderSource("sprite_lit.vs"), lps = shaderSource("sprite_lit.ps");
	if (!lvs.empty() && !lps.empty() && worldFrameCB)
	{
		RefCntAutoPtr<IShader> lv, lp;
		sci.Desc = {"SpriteLit VS", SHADER_TYPE_VERTEX, true}; sci.Source = lvs.c_str(); CreateShaderCached(sci, &lv);
		sci.Desc = {"SpriteLit PS", SHADER_TYPE_PIXEL, true};  sci.Source = lps.c_str(); CreateShaderCached(sci, &lp);
		if (lv && lp)
		{
			BufferDesc lcb; lcb.Name = "SpriteLitCB"; lcb.Size = sizeof(float) * 12;
			lcb.Usage = USAGE_DYNAMIC; lcb.BindFlags = BIND_UNIFORM_BUFFER; lcb.CPUAccessFlags = CPU_ACCESS_WRITE;
			device->CreateBuffer(lcb, nullptr, &spriteLitCB);

			GraphicsPipelineStateCreateInfo li; li.PSODesc.Name = "SpriteLit PSO";
			auto& lg = li.GraphicsPipeline;
			lg = gp;   // same targets/blend/depth/MSAA/topology/cull as the unlit sprite PSO
			lg.InputLayout.LayoutElements = layout; lg.InputLayout.NumElements = 3;
			li.pVS = lv; li.pPS = lp;
			ShaderResourceVariableDesc lvars[] = {
				{SHADER_TYPE_PIXEL, "g_Sprite", SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
				{SHADER_TYPE_PIXEL, "g_Normal", SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
				{SHADER_TYPE_PIXEL, "g_Mask",   SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC} };   // alpha mask, per flush
			ImmutableSamplerDesc limms[] = { {SHADER_TYPE_PIXEL, "g_Sprite", samp},
			                                 {SHADER_TYPE_PIXEL, "g_Normal", samp},
			                                 {SHADER_TYPE_PIXEL, "g_Mask",   samp} };
			li.PSODesc.ResourceLayout.Variables            = lvars; li.PSODesc.ResourceLayout.NumVariables         = 3;
			li.PSODesc.ResourceLayout.ImmutableSamplers    = limms; li.PSODesc.ResourceLayout.NumImmutableSamplers = 3;
			li.PSODesc.ResourceLayout.DefaultVariableType  = SHADER_RESOURCE_VARIABLE_TYPE_STATIC;
			CreateGraphicsPipelineStateCached(li, &spriteLitPSO);
			if (spriteLitPSO)
			{
				if (auto* sv = spriteLitPSO->GetStaticVariableByName(SHADER_TYPE_VERTEX, "SpriteCB"))     sv->Set(spriteCB);
				if (auto* sv = spriteLitPSO->GetStaticVariableByName(SHADER_TYPE_PIXEL,  "SpriteLitCB"))  sv->Set(spriteLitCB);
				if (auto* sv = spriteLitPSO->GetStaticVariableByName(SHADER_TYPE_PIXEL,  "FrameCB"))      sv->Set(worldFrameCB);
			}
		}
	}

	// Six-way lit smoke (drawSpriteRunSixWay): lit VS (world position through) + sprite_six.ps.
	spriteSixPSO.Release(); spriteSixSRBs.clear();
	std::string sps = shaderSource("sprite_six.ps");
	if (!lvs.empty() && !sps.empty() && worldFrameCB && spriteLitCB)
	{
		RefCntAutoPtr<IShader> lv, sp;
		sci.Desc = {"SpriteLit VS", SHADER_TYPE_VERTEX, true}; sci.Source = lvs.c_str(); CreateShaderCached(sci, &lv);
		sci.Desc = {"SpriteSix PS", SHADER_TYPE_PIXEL, true};  sci.Source = sps.c_str(); CreateShaderCached(sci, &sp);
		if (lv && sp)
		{
			GraphicsPipelineStateCreateInfo si; si.PSODesc.Name = "SpriteSix PSO";
			auto& sg = si.GraphicsPipeline;
			sg = gp;
			sg.InputLayout.LayoutElements = layout; sg.InputLayout.NumElements = 3;
			si.pVS = lv; si.pPS = sp;
			ShaderResourceVariableDesc svars[] = {
				{SHADER_TYPE_PIXEL, "g_Sprite",     SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
				{SHADER_TYPE_PIXEL, "g_SpriteB",    SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
				{SHADER_TYPE_PIXEL, "g_SceneDepth", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
				{SHADER_TYPE_PIXEL, "g_VolInteg",   SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
				{SHADER_TYPE_PIXEL, "g_VolLight",   SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
				{SHADER_TYPE_PIXEL, "g_Mask",       SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC} };   // alpha mask, per flush
			ImmutableSamplerDesc simms[] = { {SHADER_TYPE_PIXEL, "g_Sprite", samp}, {SHADER_TYPE_PIXEL, "g_SpriteB", samp},
			                                 {SHADER_TYPE_PIXEL, "g_VolInteg", samp}, {SHADER_TYPE_PIXEL, "g_VolLight", samp},
			                                 {SHADER_TYPE_PIXEL, "g_Mask", samp} };
			si.PSODesc.ResourceLayout.Variables            = svars; si.PSODesc.ResourceLayout.NumVariables         = 6;
			si.PSODesc.ResourceLayout.ImmutableSamplers    = simms; si.PSODesc.ResourceLayout.NumImmutableSamplers = 5;
			si.PSODesc.ResourceLayout.DefaultVariableType  = SHADER_RESOURCE_VARIABLE_TYPE_STATIC;
			CreateGraphicsPipelineStateCached(si, &spriteSixPSO);
			if (spriteSixPSO)
			{
				if (auto* sv = spriteSixPSO->GetStaticVariableByName(SHADER_TYPE_VERTEX, "SpriteCB"))    sv->Set(spriteCB);
				if (auto* sv = spriteSixPSO->GetStaticVariableByName(SHADER_TYPE_PIXEL,  "SpriteCB"))    sv->Set(spriteCB);
				if (auto* sv = spriteSixPSO->GetStaticVariableByName(SHADER_TYPE_PIXEL,  "SpriteLitCB")) sv->Set(spriteLitCB);
				if (auto* sv = spriteSixPSO->GetStaticVariableByName(SHADER_TYPE_PIXEL,  "FrameCB"))     sv->Set(worldFrameCB);
				if (auto* sv = spriteSixPSO->GetStaticVariableByName(SHADER_TYPE_PIXEL,  "VolCB"))       sv->Set(volCB);
			}
		}
	}

	std::cout << "[NukeDiligent]\tsprite pipeline" << (spritePSO ? " ready" : " FAILED")
	          << (spriteLitPSO ? " (+lit)" : "") << (spriteSixPSO ? " (+six-way)" : "") << std::endl;
}

// The froxel grid for a sprite run: this camera's integrated columns + incident light (white3D
// when volumetrics are off), the prepass depth for the fog behind the sprite, and g_Soft2 =
// (grid active, light amount). softOut says whether the soft fade may run (needs the prepass).
void NukeDiligent::Impl::BindSpriteVolume(IShaderResourceBinding* srb, IShaderResourceVariable* integVar, IShaderResourceVariable* lightVar,
                                          IShaderResourceVariable* depthVar, float soft2[4], bool& softOut)
{
	(void)srb;
	const bool grid = volCur && volCur->integ && volCur->light && gbufActive && gbufDepthSRV;
	ITextureView* white3 = whiteTex3D ? whiteTex3D->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE) : nullptr;
	if (integVar) integVar->Set(grid ? volCur->integ->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE) : white3);
	if (lightVar) lightVar->Set(grid ? volCur->light->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE) : white3);
	softOut = spriteSoftDist > 0.f && gbufActive && gbufDepthSRV;
	if (depthVar) depthVar->Set((softOut || grid) ? gbufDepthSRV : whiteTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
	soft2[0] = grid ? 1.f : 0.f; soft2[1] = grid ? spriteVolLight : 0.f; soft2[2] = soft2[3] = 0.f;
}

// Accumulate one quad; the batch flushes when the texture changes or at endCamera.
void NukeDiligent::drawSprite(Texture* tex, const float center[3], const float right[3], const float up[3],
                              const float uv[4], const float tint[4])
{
	m_impl->lastInstBind.pso = nullptr;   // sprite pipeline replaces the instanced VB/PSO state
	if (!m_impl->spritePSO || !tex) return;
	if (!m_impl->cameraPassActive) return;   // no camera targets bound -> nowhere valid to draw
	if (m_impl->spriteParams.overlay) { m_impl->AppendOverlayQuad(tex, center, right, up, uv, tint); return; }
	if (m_impl->spriteLitTex) m_impl->FlushSpritesLit();   // kind switch: keep paint order
	if (m_impl->spriteSixA) m_impl->FlushSpritesSix();
	if (m_impl->spriteBatchOpen && (tex != m_impl->spriteBatchTex || !Impl::SameSpriteParams(m_impl->spriteParams, m_impl->spriteBatchParams)))
		m_impl->FlushSprites();   // texture or params changed -> new batch
	m_impl->spriteBatchTex = tex; m_impl->spriteBatchMask = m_impl->spriteMask; m_impl->spriteBatchParams = m_impl->spriteParams;
	m_impl->spriteBatchOpen = true;

	auto push = [&](float sx, float sy, float u, float vv)
	{
		std::vector<float>& b = m_impl->spriteBatchVerts;
		b.push_back(center[0] + sx * right[0] + sy * up[0]);
		b.push_back(center[1] + sx * right[1] + sy * up[1]);
		b.push_back(center[2] + sx * right[2] + sy * up[2]);
		b.push_back(u); b.push_back(vv);
		b.push_back(tint[0]); b.push_back(tint[1]); b.push_back(tint[2]); b.push_back(tint[3]);
	};
	const float u0 = uv[0], v0 = uv[1], u1 = uv[2], v1 = uv[3];
	push(-1.f,  1.f, u0, v0); push( 1.f,  1.f, u1, v0); push( 1.f, -1.f, u1, v1);   // TL, TR, BR
	push(-1.f,  1.f, u0, v0); push( 1.f, -1.f, u1, v1); push(-1.f, -1.f, u0, v1);   // TL, BR, BL
}

// An overlay world quad: world-space verts into the after-post list, one run per texture / params.
void NukeDiligent::Impl::AppendOverlayQuad(Texture* tex, const float center[3], const float right[3], const float up[3],
                                           const float uv[4], const float tint[4])
{
	spriteOvlVP = curView * curProj;
	auto push = [&](float sx, float sy, float u, float vv)
	{
		std::vector<float>& b = spriteOvlVerts;
		b.push_back(center[0] + sx * right[0] + sy * up[0]);
		b.push_back(center[1] + sx * right[1] + sy * up[1]);
		b.push_back(center[2] + sx * right[2] + sy * up[2]);
		b.push_back(u); b.push_back(vv);
		b.push_back(tint[0]); b.push_back(tint[1]); b.push_back(tint[2]); b.push_back(tint[3]);
	};
	const float u0 = uv[0], v0 = uv[1], u1 = uv[2], v1 = uv[3];
	push(-1.f,  1.f, u0, v0); push( 1.f,  1.f, u1, v0); push( 1.f, -1.f, u1, v1);
	push(-1.f,  1.f, u0, v0); push( 1.f, -1.f, u1, v1); push(-1.f, -1.f, u0, v1);
	if (spriteOvlRuns.empty() || spriteOvlRuns.back().tex != tex || !SameSpriteParams(spriteOvlRuns.back().prm, spriteParams))
	{
		SprRun nr; nr.tex = tex; nr.count = 0; nr.prm = spriteParams;
		nr.clipPx[0] = 1.f; nr.clipPx[1] = 0.f; nr.clipPx[2] = 0.f; nr.clipPx[3] = 0.f;   // never clipped
		spriteOvlRuns.push_back(nr);
	}
	spriteOvlRuns.back().count += 6;
}

// Sticky SDF / clip parameters for the sprite draws that follow (canvas widgets, ABI 61).
void NukeDiligent::setSpriteParams(const NukeSpriteParams* p)
{
	m_impl->spriteParams = p ? *p : NukeSpriteParams();
}

// The params part of SpriteCB: g_Sdf = (sdf on, soft, outline width, 0), g_Outline = colour,
// g_Clip = pixel rect {x0, y0, x1, y1} (x0 >= x1 = no clip).
void NukeDiligent::Impl::FillSpriteCB(SpriteCBData& cb, const NukeSpriteParams& p, const float clipPx[4])
{
	cb.sdf[0] = p.sdf ? 1.f : 0.f; cb.sdf[1] = p.sdfSoft; cb.sdf[2] = p.outlineWidth; cb.sdf[3] = 0.f;
	memcpy(cb.outline, p.outline, sizeof(cb.outline));
	if (clipPx) memcpy(cb.clip, clipPx, sizeof(cb.clip));
	else { cb.clip[0] = 1.f; cb.clip[1] = 0.f; cb.clip[2] = 0.f; cb.clip[3] = 0.f; }
}

// Set the soft-particle fade distance for subsequent sprite runs (0 = off); flushes the open batch.
void NukeDiligent::setSpriteSoftDepth(float dist)
{
	if (dist < 0.f) dist = 0.f;
	if (m_impl->spriteSoftDist == dist) return;
	if (m_impl->spriteBatchOpen) m_impl->FlushSprites();
	if (m_impl->spriteSixA) m_impl->FlushSpritesSix();
	m_impl->spriteSoftDist = dist;
}

// Froxel-grid lighting amount for subsequent sprite runs (0 = off, 1 = the grid's incident light
// replaces the authored brightness); flushes the open batches.
void NukeDiligent::setSpriteVolumeLight(float amount)
{
	amount = amount < 0.f ? 0.f : (amount > 1.f ? 1.f : amount);
	if (m_impl->spriteVolLight == amount) return;
	if (m_impl->spriteBatchOpen) m_impl->FlushSprites();
	if (m_impl->spriteSixA) m_impl->FlushSpritesSix();
	m_impl->spriteVolLight = amount;
}

// Alpha mask for subsequent sprite runs (a particle's built-in Shape over its texture); a change
// flushes the open batches so each keeps the mask it was opened with.
void NukeDiligent::setSpriteMask(Texture* mask)
{
	if (m_impl->spriteMask == mask) return;
	if (m_impl->spriteBatchOpen) m_impl->FlushSprites();
	if (m_impl->spriteLitTex) m_impl->FlushSpritesLit();
	if (m_impl->spriteSixA) m_impl->FlushSpritesSix();
	m_impl->spriteMask = mask;
}

// Six-way lit smoke run (two lightmaps); falls back to the unlit run without the PSO or maps.
void NukeDiligent::drawSpriteRunSixWay(Texture* lightA, Texture* lightB, const float* verts, int vertCount)
{
	m_impl->lastInstBind.pso = nullptr;
	if (!m_impl->spriteSixPSO || !lightA || !lightB) { drawSpriteRun(lightA, verts, vertCount); return; }
	if (!verts || vertCount <= 0 || !m_impl->cameraPassActive) return;
	if (m_impl->spriteBatchOpen) m_impl->FlushSprites();    // kind switch: keep paint order
	if (m_impl->spriteLitTex) m_impl->FlushSpritesLit();
	if (m_impl->spriteSixA && (lightA != m_impl->spriteSixA || lightB != m_impl->spriteSixB)) m_impl->FlushSpritesSix();
	m_impl->spriteSixA = lightA; m_impl->spriteSixB = lightB; m_impl->spriteSixMask = m_impl->spriteMask;
	std::vector<float>& b = m_impl->spriteSixVerts;
	b.insert(b.end(), verts, verts + (size_t)vertCount * 9);
}

// Bulk-append pre-baked quads already in the batch vertex layout (9 floats per vertex).
// tex may be null — untextured runs draw as tinted white quads.
void NukeDiligent::drawSpriteRun(Texture* tex, const float* verts, int vertCount)
{
	m_impl->lastInstBind.pso = nullptr;   // sprite pipeline replaces the instanced VB/PSO state
	if (!m_impl->spritePSO || !verts || vertCount <= 0) return;
	if (!m_impl->cameraPassActive) return;   // no camera targets bound -> nowhere valid to draw
	if (m_impl->spriteLitTex) m_impl->FlushSpritesLit();   // kind switch: keep paint order
	if (m_impl->spriteSixA) m_impl->FlushSpritesSix();
	if (m_impl->spriteBatchOpen && (tex != m_impl->spriteBatchTex || !Impl::SameSpriteParams(m_impl->spriteParams, m_impl->spriteBatchParams)))
		m_impl->FlushSprites();
	m_impl->spriteBatchTex = tex; m_impl->spriteBatchMask = m_impl->spriteMask; m_impl->spriteBatchParams = m_impl->spriteParams;
	m_impl->spriteBatchOpen = true;
	std::vector<float>& b = m_impl->spriteBatchVerts;
	b.insert(b.end(), verts, verts + (size_t)vertCount * 9);
}

// Bulk-append quads drawn with the normal-mapped Lambert pipeline; falls back to the unlit
// run when the lit PSO or normal map is unavailable.
void NukeDiligent::drawSpriteRunLit(Texture* tex, Texture* normal, const float* verts, int vertCount,
                                    bool normalFlipY)
{
	m_impl->lastInstBind.pso = nullptr;   // sprite pipeline replaces the instanced VB/PSO state
	if (!m_impl->spriteLitPSO || !normal) { drawSpriteRun(tex, verts, vertCount); return; }
	if (!tex || !verts || vertCount <= 0) return;
	if (!m_impl->cameraPassActive) return;
	if (m_impl->spriteBatchOpen) m_impl->FlushSprites();    // kind switch: keep paint order
	if (m_impl->spriteSixA) m_impl->FlushSpritesSix();
	if (m_impl->spriteLitTex && (tex != m_impl->spriteLitTex || normal != m_impl->spriteLitNormal))
		m_impl->FlushSpritesLit();
	m_impl->spriteLitTex = tex; m_impl->spriteLitNormal = normal; m_impl->spriteLitFlipY = normalFlipY; m_impl->spriteLitMask = m_impl->spriteMask;
	std::vector<float>& b = m_impl->spriteLitVerts;
	b.insert(b.end(), verts, verts + (size_t)vertCount * 9);
}

// Draw the accumulated batch (one texture) in a single call. Must run while the camera targets
// are still bound — i.e. before the MSAA resolve.
// Split the open batch at a water surface: quads on the FAR side of it from the camera are drawn
// now (they belong to the pre-water scene the surface refracts); quads on the camera's side wait
// for FlushSpritesDeferred and stand in front of the surface, which writes depth. From above the
// far side is below the level; from under water it is above. Paint order is kept on both sides.
static void SplitQuads(std::vector<float>& verts, float y, bool camBelow, std::vector<float>& farSide, std::vector<float>& nearSide)
{
	const size_t quad = 6 * 9;
	for (size_t q = 0; q + quad <= verts.size(); q += quad)
	{
		const float cy = 0.5f * (verts[q + 1] + verts[q + 2 * 9 + 1]);   // TL.y + BR.y
		const bool below = cy < y;
		std::vector<float>& dst = (below != camBelow) ? farSide : nearSide;
		dst.insert(dst.end(), verts.begin() + q, verts.begin() + q + quad);
	}
	verts.clear();
}
void NukeDiligent::Impl::FlushSpritesBelow(float y, bool camBelow)
{
	if (!spriteBatchOpen || spriteBatchVerts.empty()) { FlushSprites(); return; }
	std::vector<float> below, above;
	SplitQuads(spriteBatchVerts, y, camBelow, below, above);
	Texture* tex = spriteBatchTex;
	if (!above.empty()) { DeferredSprites d; d.tex = tex; d.mask = spriteBatchMask; d.verts = std::move(above); spriteDeferred.push_back(std::move(d)); }
	spriteBatchVerts = std::move(below);
	FlushSprites();   // draws what is left (or clears an empty batch)
}
void NukeDiligent::Impl::FlushSpritesSixBelow(float y, bool camBelow)
{
	if (spriteSixVerts.empty() || !spriteSixA || !spriteSixB) { FlushSpritesSix(); return; }
	std::vector<float> below, above;
	SplitQuads(spriteSixVerts, y, camBelow, below, above);
	if (!above.empty()) { DeferredSprites d; d.tex = spriteSixA; d.texB = spriteSixB; d.mask = spriteSixMask; d.six = true; d.verts = std::move(above); spriteDeferred.push_back(std::move(d)); }
	spriteSixVerts = std::move(below);
	FlushSpritesSix();
}
void NukeDiligent::Impl::FlushSpritesDeferred()
{
	if (spriteDeferred.empty()) return;
	FlushSprites(); FlushSpritesSix();   // whatever is still open goes first (paint order)
	std::vector<DeferredSprites> list = std::move(spriteDeferred);
	spriteDeferred.clear();
	for (DeferredSprites& d : list)
	{
		if (d.six) { spriteSixA = d.tex; spriteSixB = d.texB; spriteSixMask = d.mask; spriteSixVerts = std::move(d.verts); FlushSpritesSix(); }
		else       { spriteBatchTex = d.tex; spriteBatchMask = d.mask; spriteBatchOpen = true; spriteBatchVerts = std::move(d.verts); FlushSprites(); }
	}
}

void NukeDiligent::Impl::FlushSprites()
{
	if (!spritePSO || !spriteStamp.current(samples, SceneFmt()) || spriteBatchVerts.empty())
	{ spriteBatchVerts.clear(); spriteBatchTex = nullptr; spriteBatchOpen = false; return; }
	// The sprite PSO needs a D32 depth buffer; outside a camera pass the bound target has none.
	if (!cameraPassActive) { spriteBatchVerts.clear(); spriteBatchTex = nullptr; spriteBatchOpen = false; return; }
	ITextureView* srv = spriteBatchTex ? GetTexSRV(spriteBatchTex)
	                                   : (whiteTex ? whiteTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE) : nullptr);
	// Draw-path diagnostics (NUKE_TM_DIAG=1).
	static const bool diag = []{ const char* e = std::getenv("NUKE_TM_DIAG"); return e && *e == '1'; }();
	if (diag)
	{
		static int n = 0;
		if (n < 6)
		{
			++n;
			std::cout << "[NukeDiligent]\tDIAG sprite flush " << spriteBatchVerts.size() / 9
			          << " verts, srv " << (srv ? "ok" : "NULL") << std::endl;
			const std::vector<float>& b = spriteBatchVerts;
			for (size_t v = 0; v + 9 <= b.size() && v < 27; v += 9)
				std::cout << "[NukeDiligent]\tDIAG v" << v / 9 << " pos(" << b[v] << "," << b[v+1] << "," << b[v+2]
				          << ") uv(" << b[v+3] << "," << b[v+4] << ") col(" << b[v+5] << "," << b[v+6]
				          << "," << b[v+7] << "," << b[v+8] << ")" << std::endl;
			if (b.size() >= 9 * 9)
			{
				size_t v = ((b.size() / 9) - 1) * 9;   // last vert (the sprite's, when appended after tiles)
				std::cout << "[NukeDiligent]\tDIAG vLAST pos(" << b[v] << "," << b[v+1] << "," << b[v+2]
				          << ") uv(" << b[v+3] << "," << b[v+4] << ")" << std::endl;
			}
		}
	}
	if (!srv) { spriteBatchVerts.clear(); spriteBatchTex = nullptr; spriteBatchOpen = false; return; }

	const int vertCount = (int)(spriteBatchVerts.size() / 9);
	if (!spriteVB || spriteVBSize < vertCount)
	{
		Trash(spriteVB);   // grows mid-frame; earlier draws this frame reference the old buffer
		spriteVB.Release();
		while (spriteVBSize < vertCount) spriteVBSize = spriteVBSize ? spriteVBSize * 2 : 384;   // 384 = 64 quads
		BufferDesc bd; bd.Name = "Sprite VB"; bd.BindFlags = BIND_VERTEX_BUFFER;
		bd.Usage = USAGE_DYNAMIC; bd.CPUAccessFlags = CPU_ACCESS_WRITE; bd.Size = (Uint64)spriteVBSize * 9 * sizeof(float);
		device->CreateBuffer(bd, nullptr, &spriteVB);
		if (!spriteVB) { spriteBatchVerts.clear(); spriteBatchTex = nullptr; return; }
	}
	{ MapHelper<float>    mv(context, spriteVB, MAP_WRITE, MAP_FLAG_DISCARD); std::memcpy(mv, spriteBatchVerts.data(), spriteBatchVerts.size() * sizeof(float)); }
	{
		// Soft particles need the single-sample depth prepass; without it the fade disables.
		// The froxel grid (own-column fog + light) binds the same way, white3D when off.
		bool soft = false; float soft2[4];
		BindSpriteVolume(spriteSRB, spriteVolIntegVar, spriteVolLightVar, spriteDepthVar, soft2, soft);
		MapHelper<SpriteCBData> cb(context, spriteCB, MAP_WRITE, MAP_FLAG_DISCARD);
		if (cb != nullptr)
		{
			cb->vp = curView * curProj;
			cb->soft[0] = spriteSoftDist; cb->soft[1] = curNear; cb->soft[2] = curFar; cb->soft[3] = soft ? 1.f : 0.f;
			memcpy(cb->soft2, soft2, sizeof(soft2));
			FillSpriteCB(*cb, spriteBatchParams, nullptr);   // world quads: SDF applies, no clip
		}
	}
	if (spriteTexVar) spriteTexVar->Set(srv);
	if (spriteMaskVar)
	{
		ITextureView* msrv = spriteBatchMask ? GetTexSRV(spriteBatchMask) : nullptr;
		spriteMaskVar->Set(msrv ? msrv : whiteTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
	}

	Uint64 offset = 0; IBuffer* vbs[] = { spriteVB };
	context->SetVertexBuffers(0, 1, vbs, &offset, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, SET_VERTEX_BUFFERS_FLAG_RESET);
	context->SetPipelineState(spritePSO);
	context->CommitShaderResources(spriteSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	DrawAttribs da; da.NumVertices = (Uint32)vertCount; da.Flags = DRAW_FLAG_VERIFY_ALL;
	context->Draw(da);
	CoverAppend(spriteBatchTex, spriteBatchMask, spriteBatchVerts);

	spriteBatchVerts.clear();
	spriteBatchTex = nullptr;
	spriteBatchOpen = false;
}

// Draw the accumulated lit batch (one diffuse+normal pair); the plane TBN comes from the
// first quad's corners, since a run shares one plane.
void NukeDiligent::Impl::FlushSpritesLit()
{
	if (spriteLitVerts.empty() || !spriteLitTex) { spriteLitVerts.clear(); spriteLitTex = nullptr; spriteLitNormal = nullptr; return; }
	auto drop = [&]{ spriteLitVerts.clear(); spriteLitTex = nullptr; spriteLitNormal = nullptr; };
	if (!spriteLitPSO || !cameraPassActive) { drop(); return; }
	ITextureView* srv  = GetTexSRV(spriteLitTex);
	ITextureView* nsrv = GetTexSRV(spriteLitNormal);
	if (!srv || !nsrv) { drop(); return; }

	// Per-batch TBN from the first quad (verts: TL, TR, BR, ... — 9 floats each).
	{
		const float* v = spriteLitVerts.data();
		float3 tl(v[0], v[1], v[2]), tr(v[9], v[10], v[11]), br(v[18], v[19], v[20]);
		float3 T = tr - tl, B = tr - br;
		float tl2 = length(T), bl2 = length(B);
		T = (tl2 > 1e-6f) ? T / tl2 : float3(1, 0, 0);
		B = (bl2 > 1e-6f) ? B / bl2 : float3(0, 1, 0);
		float3 N = cross(T, B);
		float nl = length(N); N = (nl > 1e-6f) ? N / nl : float3(0, 0, 1);
		MapHelper<float> cb(context, spriteLitCB, MAP_WRITE, MAP_FLAG_DISCARD);
		if (cb)
		{
			float* d = cb;
			d[0] = T.x; d[1] = T.y; d[2]  = T.z; d[3]  = 0;
			d[4] = B.x; d[5] = B.y; d[6]  = B.z; d[7]  = 0;
			d[8] = N.x; d[9] = N.y; d[10] = N.z; d[11] = spriteLitFlipY ? 1.0f : -1.0f;
		}
	}

	const int vertCount = (int)(spriteLitVerts.size() / 9);
	if (!spriteVB || spriteVBSize < vertCount)   // shared VB with the unlit flush (sequential use)
	{
		Trash(spriteVB);
		spriteVB.Release();
		while (spriteVBSize < vertCount) spriteVBSize = spriteVBSize ? spriteVBSize * 2 : 384;
		BufferDesc bd; bd.Name = "Sprite VB"; bd.BindFlags = BIND_VERTEX_BUFFER;
		bd.Usage = USAGE_DYNAMIC; bd.CPUAccessFlags = CPU_ACCESS_WRITE; bd.Size = (Uint64)spriteVBSize * 9 * sizeof(float);
		device->CreateBuffer(bd, nullptr, &spriteVB);
		if (!spriteVB) { drop(); return; }
	}
	{ MapHelper<float>    mv(context, spriteVB, MAP_WRITE, MAP_FLAG_DISCARD); std::memcpy(mv, spriteLitVerts.data(), spriteLitVerts.size() * sizeof(float)); }
	{
		MapHelper<SpriteCBData> cb(context, spriteCB, MAP_WRITE, MAP_FLAG_DISCARD);
		if (cb != nullptr) { memset(&*cb, 0, sizeof(SpriteCBData)); cb->vp = curView * curProj; cb->clip[0] = 1.f; }
	}

	// One SRB per (diffuse, normal) pair: MUTABLE vars are set once, avoiding dynamic descriptors.
	RefCntAutoPtr<IShaderResourceBinding>& srb = spriteLitSRBs[{srv, nsrv}];
	if (!srb)
	{
		spriteLitPSO->CreateShaderResourceBinding(&srb, true);
		if (!srb) { spriteLitSRBs.erase({srv, nsrv}); drop(); return; }
		if (auto* v = srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Sprite")) v->Set(srv);
		if (auto* v = srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Normal")) v->Set(nsrv);
	}

	Uint64 offset = 0; IBuffer* vbs[] = { spriteVB };
	context->SetVertexBuffers(0, 1, vbs, &offset, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, SET_VERTEX_BUFFERS_FLAG_RESET);
	if (auto* mv2 = srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Mask"))
	{
		ITextureView* msrv = spriteLitMask ? GetTexSRV(spriteLitMask) : nullptr;
		mv2->Set(msrv ? msrv : whiteTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
	}
	context->SetPipelineState(spriteLitPSO);
	context->CommitShaderResources(srb, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	DrawAttribs da; da.NumVertices = (Uint32)vertCount; da.Flags = DRAW_FLAG_VERIFY_ALL;
	context->Draw(da);
	CoverAppend(spriteLitTex, spriteLitMask, spriteLitVerts);
	drop();
}

// Draw the accumulated six-way batch (one lightmap pair): the billboard frame from the first
// quad (right, up, toward the eye) in SpriteLitCB, lights from worldFrameCB, the froxel grid
// like the unlit run.
void NukeDiligent::Impl::FlushSpritesSix()
{
	auto drop = [&]{ spriteSixVerts.clear(); spriteSixA = nullptr; spriteSixB = nullptr; };
	if (spriteSixVerts.empty() || !spriteSixA || !spriteSixB) { drop(); return; }
	if (!spriteSixPSO || !cameraPassActive) { drop(); return; }
	ITextureView* srvA = GetTexSRV(spriteSixA);
	ITextureView* srvB = GetTexSRV(spriteSixB);
	if (!srvA || !srvB) { drop(); return; }
	{
		const float* v = spriteSixVerts.data();
		float3 tl(v[0], v[1], v[2]), tr(v[9], v[10], v[11]), br(v[18], v[19], v[20]);
		float3 T = tr - tl, B = tr - br;
		float tl2 = length(T), bl2 = length(B);
		T = (tl2 > 1e-6f) ? T / tl2 : float3(1, 0, 0);
		B = (bl2 > 1e-6f) ? B / bl2 : float3(0, 1, 0);
		float3 N = cross(T, B);
		float nl = length(N); N = (nl > 1e-6f) ? N / nl : float3(0, 0, 1);
		float3 toEye(curCamPos[0] - tl.x, curCamPos[1] - tl.y, curCamPos[2] - tl.z);
		if (dot(N, toEye) < 0.f) N = -N;   // "front" = toward the eye
		MapHelper<float> cb(context, spriteLitCB, MAP_WRITE, MAP_FLAG_DISCARD);
		if (cb)
		{
			float* d = cb;
			d[0] = T.x; d[1] = T.y; d[2]  = T.z; d[3]  = 0;
			d[4] = B.x; d[5] = B.y; d[6]  = B.z; d[7]  = 0;
			d[8] = N.x; d[9] = N.y; d[10] = N.z; d[11] = 0;
		}
	}
	const int vertCount = (int)(spriteSixVerts.size() / 9);
	if (!spriteVB || spriteVBSize < vertCount)
	{
		Trash(spriteVB);
		spriteVB.Release();
		while (spriteVBSize < vertCount) spriteVBSize = spriteVBSize ? spriteVBSize * 2 : 384;
		BufferDesc bd; bd.Name = "Sprite VB"; bd.BindFlags = BIND_VERTEX_BUFFER;
		bd.Usage = USAGE_DYNAMIC; bd.CPUAccessFlags = CPU_ACCESS_WRITE; bd.Size = (Uint64)spriteVBSize * 9 * sizeof(float);
		device->CreateBuffer(bd, nullptr, &spriteVB);
		if (!spriteVB) { drop(); return; }
	}
	{ MapHelper<float> mv(context, spriteVB, MAP_WRITE, MAP_FLAG_DISCARD); std::memcpy(mv, spriteSixVerts.data(), spriteSixVerts.size() * sizeof(float)); }
	RefCntAutoPtr<IShaderResourceBinding>& srb = spriteSixSRBs[{srvA, srvB}];
	if (!srb)
	{
		spriteSixPSO->CreateShaderResourceBinding(&srb, true);
		if (!srb) { spriteSixSRBs.erase({srvA, srvB}); drop(); return; }
		if (auto* v = srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Sprite"))  v->Set(srvA);
		if (auto* v = srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_SpriteB")) v->Set(srvB);
	}
	{
		bool soft = false; float soft2[4];
		BindSpriteVolume(srb, srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_VolInteg"), srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_VolLight"),
		                 srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_SceneDepth"), soft2, soft);
		MapHelper<SpriteCBData> cb(context, spriteCB, MAP_WRITE, MAP_FLAG_DISCARD);
		if (cb != nullptr)
		{
			memset(&*cb, 0, sizeof(SpriteCBData)); cb->clip[0] = 1.f;
			cb->vp = curView * curProj;
			cb->soft[0] = spriteSoftDist; cb->soft[1] = curNear; cb->soft[2] = curFar; cb->soft[3] = soft ? 1.f : 0.f;
			memcpy(cb->soft2, soft2, sizeof(soft2));
		}
	}
	Uint64 offset = 0; IBuffer* vbs[] = { spriteVB };
	context->SetVertexBuffers(0, 1, vbs, &offset, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, SET_VERTEX_BUFFERS_FLAG_RESET);
	if (auto* mv2 = srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Mask"))
	{
		ITextureView* msrv = spriteSixMask ? GetTexSRV(spriteSixMask) : nullptr;
		mv2->Set(msrv ? msrv : whiteTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
	}
	context->SetPipelineState(spriteSixPSO);
	context->CommitShaderResources(srb, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	DrawAttribs da; da.NumVertices = (Uint32)vertCount; da.Flags = DRAW_FLAG_VERIFY_ALL;
	context->Draw(da);
	CoverAppend(spriteSixA, spriteSixMask, spriteSixVerts);
	drop();
}

// ---- sprite coverage for the RT reflection composite ------------------------------------------

bool NukeDiligent::Impl::RTReflectWanted() const
{
	if (!rtSupported || !gbufActive) return false;
	for (const auto& cs : postChain)
	{
		auto pit = postPipes.find(cs.pipeline);
		if (pit != postPipes.end() && pit->second.isRTRef) return true;
	}
	return false;
}

// Remember a flushed batch (its quads + texture) for the coverage draw at endCamera.
void NukeDiligent::Impl::CoverAppend(Texture* tex, Texture* mask, const std::vector<float>& verts)
{
	if (!coverWanted || !coverPSO || verts.empty()) return;
	CoverBatch b; b.tex = tex; b.mask = mask; b.first = (uint32_t)(coverVerts.size() / 9); b.count = (uint32_t)(verts.size() / 9);
	coverVerts.insert(coverVerts.end(), verts.begin(), verts.end());
	coverBatches.push_back(b);
}

// endCamera, before the resolve: redraw this pass's sprite quads into the R8 union mask,
// depth-tested against the G-buffer depth (opaque prepass + the water's G-pass), then rebind
// the camera targets. coverSRV = the mask for the tracer (null when nothing covers).
void NukeDiligent::Impl::DrawSpriteCoverage()
{
	coverSRV = nullptr;
	auto done = [&]{ coverVerts.clear(); coverBatches.clear(); };
	if (!coverWanted || coverBatches.empty() || !coverPSO || !coverSRB || !gbufActive || !gbufDSV || curRTW <= 0 || curRTH <= 0) { done(); return; }
	const uint64_t key = ((uint64_t)(uint32_t)curRTW << 32) | (uint32_t)curRTH;
	SizedTexSet& s = coverCache[key];
	if (!s.a)
	{
		TextureDesc td; td.Name = "Sprite Coverage"; td.Type = RESOURCE_DIM_TEX_2D;
		td.Width = (Uint32)curRTW; td.Height = (Uint32)curRTH; td.Format = TEX_FORMAT_R8_UNORM;
		td.BindFlags = BIND_RENDER_TARGET | BIND_SHADER_RESOURCE;
		device->CreateTexture(td, nullptr, &s.a);
	}
	s.lastUsed = ++sizedClock;
	EvictSized(coverCache, key);
	if (!s.a) { done(); return; }
	const int vertCount = (int)(coverVerts.size() / 9);
	if (!coverVB || coverVBSize < vertCount)
	{
		Trash(coverVB); coverVB.Release();
		while (coverVBSize < vertCount) coverVBSize = coverVBSize ? coverVBSize * 2 : 384;
		BufferDesc bd; bd.Name = "Sprite Cover VB"; bd.BindFlags = BIND_VERTEX_BUFFER;
		bd.Usage = USAGE_DYNAMIC; bd.CPUAccessFlags = CPU_ACCESS_WRITE; bd.Size = (Uint64)coverVBSize * 9 * sizeof(float);
		device->CreateBuffer(bd, nullptr, &coverVB);
		if (!coverVB) { done(); return; }
	}
	{ MapHelper<float> mv(context, coverVB, MAP_WRITE, MAP_FLAG_DISCARD); std::memcpy(mv, coverVerts.data(), coverVerts.size() * sizeof(float)); }
	{
		MapHelper<SpriteCBData> cb(context, spriteCB, MAP_WRITE, MAP_FLAG_DISCARD);
		if (cb != nullptr) { memset(&*cb, 0, sizeof(SpriteCBData)); cb->vp = curView * curProj; cb->clip[0] = 1.f; }
	}
	ITextureView* rtv = s.a->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET);
	context->SetRenderTargets(1, &rtv, gbufDSV, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	const float zero[4] = { 0, 0, 0, 0 };
	context->ClearRenderTarget(rtv, zero, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	Viewport vp; vp.TopLeftX = 0; vp.TopLeftY = 0; vp.Width = (float)curRTW; vp.Height = (float)curRTH; vp.MinDepth = 0; vp.MaxDepth = 1;
	context->SetViewports(1, &vp, curRTW, curRTH);
	Uint64 offset = 0; IBuffer* vbs[] = { coverVB };
	context->SetVertexBuffers(0, 1, vbs, &offset, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, SET_VERTEX_BUFFERS_FLAG_RESET);
	context->SetPipelineState(coverPSO);
	ITextureView* white = whiteTex ? whiteTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE) : nullptr;
	for (const CoverBatch& b : coverBatches)
	{
		ITextureView* srv = b.tex ? GetTexSRV(b.tex) : white;
		if (!srv) srv = white;
		if (!srv) continue;
		if (coverTexVar) coverTexVar->Set(srv);
		if (coverMaskVar) { ITextureView* msrv = b.mask ? GetTexSRV(b.mask) : nullptr; coverMaskVar->Set(msrv ? msrv : white); }
		context->CommitShaderResources(coverSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		DrawAttribs da; da.NumVertices = b.count; da.StartVertexLocation = b.first; da.Flags = DRAW_FLAG_VERIFY_ALL;
		context->Draw(da);
	}
	context->SetRenderTargets(1, &curRTV, curDSV, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	context->SetViewports(1, &vp, curRTW, curRTH);
	coverSRV = s.a->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
	done();
}

// ---- screen-space (Canvas HUD) sprites --------------------------------------------------------

void NukeDiligent::drawSpriteScreen(Texture* tex, const float rect[4], const float refSize[2],
                                    const float uv[4], const float tint[4], int afterPost)
{
	if (!tex) return;
	drawSpriteScreenEx(tex, rect, refSize, uv, tint, afterPost, 0);
}

void NukeDiligent::drawSpriteScreenEx(Texture* tex, const float rect[4], const float refSize[2],
                                      const float uv[4], const float tint[4], int afterPost, int scaleMode)
{
	if (!tex) return;
	if (!m_impl->cameraPassActive) return;   // canvas sprites need the camera targets bound
	if (afterPost) m_impl->AppendScreenSprite(m_impl->spriteScrPostVerts, m_impl->spriteScrPostRuns, tex, rect, refSize, uv, tint, scaleMode);
	else           m_impl->AppendScreenSprite(m_impl->spriteScrPreVerts,  m_impl->spriteScrPreRuns,  tex, rect, refSize, uv, tint, scaleMode);
}

// Build one NDC quad from a reference-pixel rect (centre + size) per the canvas scale mode and
// append it, plus a per-texture run marker, to a screen batch.
void NukeDiligent::Impl::AppendScreenSprite(std::vector<float>& verts, std::vector<SprRun>& runs, Texture* tex,
                                            const float rect[4], const float refSize[2], const float uv[4], const float tint[4],
                                            int scaleMode)
{
	const float tw = (float)(curRTW > 0 ? curRTW : 1), th = (float)(curRTH > 0 ? curRTH : 1);
	const float refW = refSize[0] > 1.f ? refSize[0] : 1.f, refH = refSize[1] > 1.f ? refSize[1] : 1.f;
	const float sw = tw / refW, sh = th / refH;   // per-axis reference->target scale
	float sx, sy;
	switch (scaleMode)
	{
		default:
		case 0: sx = sy = (sh < sw ? sh : sw); break;   // Fit: uniform, whole canvas visible (letterboxed)
		case 1: sx = sw; sy = sh;              break;   // Stretch: canvas corners = screen corners
		case 2: sx = sy = (sh > sw ? sh : sw); break;   // Expand: uniform, covers the screen (may crop)
		case 3: sx = sy = sw;                  break;   // FitWidth
		case 4: sx = sy = sh;                  break;   // FitHeight
	}
	const float cx = rect[0] * sx / (tw * 0.5f), cy = rect[1] * sy / (th * 0.5f);
	const float hw = rect[2] * 0.5f * sx / (tw * 0.5f), hh = rect[3] * 0.5f * sy / (th * 0.5f);
	const float u0 = uv[0], v0 = uv[1], u1 = uv[2], v1 = uv[3];
	auto push = [&](float x, float y, float u, float vv)
	{
		verts.push_back(x); verts.push_back(y); verts.push_back(0.0f);
		verts.push_back(u); verts.push_back(vv);
		verts.push_back(tint[0]); verts.push_back(tint[1]); verts.push_back(tint[2]); verts.push_back(tint[3]);
	};
	push(cx - hw, cy + hh, u0, v0); push(cx + hw, cy + hh, u1, v0); push(cx + hw, cy - hh, u1, v1);   // TL,TR,BR
	push(cx - hw, cy + hh, u0, v0); push(cx + hw, cy - hh, u1, v1); push(cx - hw, cy - hh, u0, v1);   // TL,BR,BL
	// The clip rect goes through the same reference->target mapping, to target pixels (top-left origin).
	float clipPx[4] = { 1.f, 0.f, 0.f, 0.f };
	if (spriteParams.clip)
	{
		const float* c = spriteParams.clipRect;
		const float x0 = (c[0] * sx / (tw * 0.5f) + 1.f) * 0.5f * tw, x1 = (c[2] * sx / (tw * 0.5f) + 1.f) * 0.5f * tw;
		const float y0 = (1.f - c[3] * sy / (th * 0.5f)) * 0.5f * th, y1 = (1.f - c[1] * sy / (th * 0.5f)) * 0.5f * th;
		clipPx[0] = x0; clipPx[1] = y0; clipPx[2] = x1; clipPx[3] = y1;
	}
	if (runs.empty() || runs.back().tex != tex || !SameSpriteParams(runs.back().prm, spriteParams)
	    || memcmp(runs.back().clipPx, clipPx, sizeof(clipPx)) != 0)
	{
		SprRun nr; nr.tex = tex; nr.count = 0; nr.prm = spriteParams; memcpy(nr.clipPx, clipPx, sizeof(clipPx));
		runs.push_back(nr);
	}
	runs.back().count += 6;
}

// Replay a screen batch: identity transform (verts are already NDC), one draw per texture run.
void NukeDiligent::Impl::FlushScreen(std::vector<float>& verts, std::vector<SprRun>& runs, IPipelineState* pso,
                                     IShaderResourceBinding* srb, IShaderResourceVariable* texVar, const float4x4* vp)
{
	if (!pso || verts.empty() || runs.empty()) { verts.clear(); runs.clear(); return; }
	const int vertCount = (int)(verts.size() / 9);
	if (!spriteVB || spriteVBSize < vertCount)
	{
		Trash(spriteVB);   // grows mid-frame; earlier draws this frame reference the old buffer
		spriteVB.Release();
		while (spriteVBSize < vertCount) spriteVBSize = spriteVBSize ? spriteVBSize * 2 : 384;
		BufferDesc bd; bd.Name = "Sprite VB"; bd.BindFlags = BIND_VERTEX_BUFFER;
		bd.Usage = USAGE_DYNAMIC; bd.CPUAccessFlags = CPU_ACCESS_WRITE; bd.Size = (Uint64)spriteVBSize * 9 * sizeof(float);
		device->CreateBuffer(bd, nullptr, &spriteVB);
		if (!spriteVB) { verts.clear(); runs.clear(); return; }
	}
	{ MapHelper<float>    mv(context, spriteVB, MAP_WRITE, MAP_FLAG_DISCARD); std::memcpy(mv, verts.data(), verts.size() * sizeof(float)); }
	Uint64 offset = 0; IBuffer* vbs[] = { spriteVB };
	context->SetVertexBuffers(0, 1, vbs, &offset, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, SET_VERTEX_BUFFERS_FLAG_RESET);
	context->SetPipelineState(pso);
	int base = 0;
	for (const SprRun& r : runs)
	{
		ITextureView* srv = r.tex ? GetTexSRV(r.tex) : nullptr;
		if (srv && texVar)
		{
			{   // per run: identity transform + this run's SDF / clip params
				MapHelper<SpriteCBData> cb(context, spriteCB, MAP_WRITE, MAP_FLAG_DISCARD);
				if (cb != nullptr) { memset(&*cb, 0, sizeof(SpriteCBData)); cb->vp = vp ? *vp : float4x4::Identity(); FillSpriteCB(*cb, r.prm, r.clipPx); }
			}
			texVar->Set(srv);
			context->CommitShaderResources(srb, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
			DrawAttribs da; da.NumVertices = (Uint32)r.count; da.StartVertexLocation = (Uint32)base; da.Flags = DRAW_FLAG_VERIFY_ALL;
			context->Draw(da);
		}
		base += r.count;
	}
	verts.clear(); runs.clear();
}

// Draw the before-post screen sprites; they reuse the depth-tested in-scene sprite PSO, so the
// camera targets must be bound.
void NukeDiligent::Impl::FlushScreenPre()
{
	if (!cameraPassActive) { spriteScrPreVerts.clear(); spriteScrPreRuns.clear(); return; }
	FlushScreen(spriteScrPreVerts, spriteScrPreRuns, spritePSO, spriteSRB, spriteTexVar);
}

void NukeDiligent::Impl::FlushScreenPost(bool toBackbuffer)
{
	if (toBackbuffer) FlushScreen(spriteScrPostVerts, spriteScrPostRuns, spriteScreenPSOBB, spriteScreenSRBBB, spriteScreenTexVarBB);
	else              FlushScreen(spriteScrPostVerts, spriteScrPostRuns, spriteScreenPSO,   spriteScreenSRB,   spriteScreenTexVar);
	spriteOvlVerts.clear(); spriteOvlRuns.clear();   // a camera that never reached the overlay flush drops them
}

void NukeDiligent::Impl::FlushWorldOverlay(bool toBackbuffer)
{
	if (toBackbuffer) FlushScreen(spriteOvlVerts, spriteOvlRuns, spriteScreenPSOBB, spriteScreenSRBBB, spriteScreenTexVarBB, &spriteOvlVP);
	else              FlushScreen(spriteOvlVerts, spriteOvlRuns, spriteScreenPSO,   spriteScreenSRB,   spriteScreenTexVar,   &spriteOvlVP);
}

// A depth-based post stage (DOF, motion blur) reads the G-buffer depth: module surfaces (the
// water) must write themselves into it, or the effect sees the geometry BEHIND them.
bool NukeDiligent::Impl::PostWantsDepth() const
{
	if (!gbufActive) return false;
	for (const auto& cs : postChain)
	{
		auto pit = postPipes.find(cs.pipeline);
		if (pit != postPipes.end() && (pit->second.isDOF || pit->second.isMotion || pit->second.isUpscale)) return true;   // upscale: its depth input + the water flag of the reactive mask
	}
	return false;
}
