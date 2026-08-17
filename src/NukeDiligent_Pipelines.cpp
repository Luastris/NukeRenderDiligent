#include "NukeDiligentImpl.h"

// Overlay-slot texture names, generated to stay in lock-step with kOvSlots (the HLSL declares
// g_Ov{slot}{Alb|Nrm|MR|Mask} + g_Mask3D; the G-buffer set has no albedos).
const std::vector<std::string>& NukeDiligent::Impl::OvTexNames()
{
	static const std::vector<std::string> names = []{
		std::vector<std::string> v;
		static const char* kind[4] = { "Alb", "Nrm", "MR", "Mask" };
		for (int s = 0; s < kOvSlots; ++s)
			for (int k = 0; k < 4; ++k) v.push_back("g_Ov" + std::to_string(s) + kind[k]);
		v.push_back("g_Mask3D");
		v.push_back("g_Detail");
		v.push_back("g_DetailNrm");
		return v;
	}();
	return names;
}

const std::vector<std::string>& NukeDiligent::Impl::OvGbufNames()
{
	static const std::vector<std::string> names = []{
		std::vector<std::string> v;
		static const char* kind[3] = { "Nrm", "MR", "Mask" };
		for (int s = 0; s < kOvSlots; ++s)
			for (int k = 0; k < 3; ++k) v.push_back("g_Ov" + std::to_string(s) + kind[k]);
		v.push_back("g_Mask3D");
		v.push_back("g_DetailNrm");
		return v;
	}();
	return names;
}

static const char GAMMA_TO_LINEAR[] = "((Gamma) < 0.04045 ? (Gamma) / 12.92 : pow(max((Gamma) + 0.055, 0.0) / 1.055, 2.4))";
static const char SRGBA_TO_LINEAR[] =
    "col.r = GAMMA_TO_LINEAR(col.r); col.g = GAMMA_TO_LINEAR(col.g); "
    "col.b = GAMMA_TO_LINEAR(col.b); col.a = 1.0 - GAMMA_TO_LINEAR(1.0 - col.a);";

void NukeDiligent::Impl::CreateUIPipeline(TEXTURE_FORMAT bbFmt, TEXTURE_FORMAT dsFmt)
{
	baseVertexSupported = (device->GetAdapterInfo().DrawCommand.CapFlags & DRAW_COMMAND_CAP_FLAG_BASE_VERTEX) != 0;

	ShaderCreateInfo ShaderCI;
	ShaderCI.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;

	const bool srgb = GetTextureFormatAttribs(bbFmt).ComponentType == COMPONENT_TYPE_UNORM_SRGB;
	ShaderMacro macrosSrgb[] = {{"GAMMA_TO_LINEAR(Gamma)", GAMMA_TO_LINEAR}, {"SRGBA_TO_LINEAR(col)", SRGBA_TO_LINEAR}};
	ShaderMacro macrosNone[] = {{"SRGBA_TO_LINEAR(col)", ""}};
	ShaderCI.Macros = srgb ? ShaderMacroArray{macrosSrgb, 2} : ShaderMacroArray{macrosNone, 1};

	std::string vsSrc = shaderSource("ui.vs");
	std::string psSrc = shaderSource("ui.ps");
	RefCntAutoPtr<IShader> vs, ps;
	ShaderCI.Desc = {"UI VS", SHADER_TYPE_VERTEX, true};
	ShaderCI.Source = vsSrc.c_str();
	CreateShaderCached(ShaderCI, &vs);
	ShaderCI.Desc = {"UI PS", SHADER_TYPE_PIXEL, true};
	ShaderCI.Source = psSrc.c_str();
	CreateShaderCached(ShaderCI, &ps);

	GraphicsPipelineStateCreateInfo PSOCreateInfo;
	PSOCreateInfo.PSODesc.Name = "UI PSO";
	auto& GP = PSOCreateInfo.GraphicsPipeline;
	GP.NumRenderTargets  = 1;
	GP.RTVFormats[0]     = bbFmt;
	// UI binds a null DSV, so the PSO must declare DSVFormat = UNKNOWN or Diligent reports a
	// depth-format mismatch every frame. dsFmt is intentionally unused.
	(void)dsFmt;
	GP.DSVFormat         = TEX_FORMAT_UNKNOWN;
	GP.PrimitiveTopology = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	GP.RasterizerDesc.CullMode      = CULL_MODE_NONE;
	GP.RasterizerDesc.ScissorEnable = True;
	GP.DepthStencilDesc.DepthEnable = False;
	auto& RT0 = GP.BlendDesc.RenderTargets[0];
	RT0.BlendEnable    = True;
	RT0.SrcBlend       = BLEND_FACTOR_ONE;
	RT0.DestBlend      = BLEND_FACTOR_INV_SRC_ALPHA;
	RT0.BlendOp        = BLEND_OPERATION_ADD;
	RT0.SrcBlendAlpha  = BLEND_FACTOR_ONE;
	RT0.DestBlendAlpha = BLEND_FACTOR_INV_SRC_ALPHA;
	RT0.BlendOpAlpha   = BLEND_OPERATION_ADD;
	RT0.RenderTargetWriteMask = COLOR_MASK_ALL;
	PSOCreateInfo.pVS = vs;
	PSOCreateInfo.pPS = ps;

	LayoutElement layout[] = {
		{0, 0, 2, VT_FLOAT32},     // pos
		{1, 0, 2, VT_FLOAT32},     // uv
		{2, 0, 4, VT_UINT8, True}, // col (normalized RGBA8)
	};
	GP.InputLayout.NumElements    = 3;
	GP.InputLayout.LayoutElements = layout;

	// MUTABLE, not DYNAMIC: the UI binds through a per-texture SRB cache, so commits consume no
	// dynamic GPU descriptors (a DYNAMIC var allocates fresh ones on every commit and drains the heap).
	ShaderResourceVariableDesc vars[] = {{SHADER_TYPE_PIXEL, "Texture", SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE}};
	PSOCreateInfo.PSODesc.ResourceLayout.Variables    = vars;
	PSOCreateInfo.PSODesc.ResourceLayout.NumVariables = 1;
	SamplerDesc sam;
	sam.AddressU = sam.AddressV = sam.AddressW = TEXTURE_ADDRESS_CLAMP;
	ImmutableSamplerDesc samplers[] = {{SHADER_TYPE_PIXEL, "Texture", sam}};
	PSOCreateInfo.PSODesc.ResourceLayout.ImmutableSamplers    = samplers;
	PSOCreateInfo.PSODesc.ResourceLayout.NumImmutableSamplers = 1;

	CreateGraphicsPipelineStateCached(PSOCreateInfo, &uiPSO);

	BufferDesc cbd;
	cbd.Name = "UI projection CB";
	cbd.Size = sizeof(float4x4);
	cbd.Usage = USAGE_DYNAMIC;
	cbd.BindFlags = BIND_UNIFORM_BUFFER;
	cbd.CPUAccessFlags = CPU_ACCESS_WRITE;
	device->CreateBuffer(cbd, nullptr, &uiCB);
	uiPSO->GetStaticVariableByName(SHADER_TYPE_VERTEX, "Constants")->Set(uiCB);
	// No single SRB: DrawUILists gets one from the per-texture cache (UISRBFor).
}

void NukeDiligent::Impl::CreateWorldPipeline()
{
	// Shared constant buffers (bound as static vars on EVERY world PSO).
	BufferDesc cbd;
	cbd.Name = "World CB"; cbd.Size = sizeof(float4x4) * 3; cbd.Usage = USAGE_DYNAMIC;   // wvp, world, prevWVP (gbuffer velocity)
	cbd.BindFlags = BIND_UNIFORM_BUFFER; cbd.CPUAccessFlags = CPU_ACCESS_WRITE;
	device->CreateBuffer(cbd, nullptr, &worldCB);

	BufferDesc mcbd;
	mcbd.Name = "World MatCB"; mcbd.Size = kMatCBBytes; mcbd.Usage = USAGE_DYNAMIC;
	mcbd.BindFlags = BIND_UNIFORM_BUFFER; mcbd.CPUAccessFlags = CPU_ACCESS_WRITE;
	device->CreateBuffer(mcbd, nullptr, &worldMatCB);

	BufferDesc fcbd;
	fcbd.Name = "World FrameCB"; fcbd.Size = sizeof(FrameCBData); fcbd.Usage = USAGE_DYNAMIC;
	fcbd.BindFlags = BIND_UNIFORM_BUFFER; fcbd.CPUAccessFlags = CPU_ACCESS_WRITE;
	device->CreateBuffer(fcbd, nullptr, &worldFrameCB);

	// Bend CB for the instanced vertex shaders: 59 float4s (g_WindV, g_WindT, g_WindP, g_Push[8], g_Vol[48]).
	BufferDesc bcbd;
	bcbd.Name = "Bend CB"; bcbd.Size = sizeof(float) * 4 * 59; bcbd.Usage = USAGE_DYNAMIC;
	bcbd.BindFlags = BIND_UNIFORM_BUFFER; bcbd.CPUAccessFlags = CPU_ACCESS_WRITE;
	device->CreateBuffer(bcbd, nullptr, &bendCB);

	// Per-draw flags (x = receiveShadows) get their OWN cbuffer, not a MatCB slot: custom shaders
	// parse MatCB for prop offsets, so growing its header would shift every user shader's props.
	BufferDesc dfbd;
	dfbd.Name = "DrawFlags CB"; dfbd.Size = sizeof(float) * 4; dfbd.Usage = USAGE_DYNAMIC;
	dfbd.BindFlags = BIND_UNIFORM_BUFFER; dfbd.CPUAccessFlags = CPU_ACCESS_WRITE;
	device->CreateBuffer(dfbd, nullptr, &drawFlagsCB);

	// Foliage bend compute (bend.cs): bends merged chunk meshes so their BLAS sways with the wind.
	// Shares nukebend.hlsl with the raster VS shaders and BendCB with setWind. RT-only.
	if (rtSupported)
	{
		std::string cs = shaderSource("bend.cs");
		if (!cs.empty())
		{
			ShaderCreateInfo sci; sci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
			sci.pShaderSourceStreamFactory = shaderFactory;   // resolves #include "nukebend.hlsl"
			sci.Desc = {"Foliage Bend CS", SHADER_TYPE_COMPUTE, true};
			sci.Source = cs.c_str();
			RefCntAutoPtr<IShader> csh; CreateShaderCached(sci, &csh);
			if (csh)
			{
				BufferDesc cpb; cpb.Name = "BendCS Params"; cpb.Size = sizeof(float) * 4;
				cpb.Usage = USAGE_DYNAMIC; cpb.BindFlags = BIND_UNIFORM_BUFFER; cpb.CPUAccessFlags = CPU_ACCESS_WRITE;
				device->CreateBuffer(cpb, nullptr, &bendCSParamsCB);
				ComputePipelineStateCreateInfo cci; cci.PSODesc.Name = "Foliage Bend PSO";
				ShaderResourceVariableDesc cvars[] = {
					{SHADER_TYPE_COMPUTE, "g_SrcPos",    SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
					{SHADER_TYPE_COMPUTE, "g_BendData",  SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
					{SHADER_TYPE_COMPUTE, "g_BendPivot", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
					{SHADER_TYPE_COMPUTE, "g_DstPos",    SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
				};
				cci.PSODesc.ResourceLayout.Variables = cvars; cci.PSODesc.ResourceLayout.NumVariables = 4;
				cci.pCS = csh;
				device->CreateComputePipelineState(cci, &bendCSPSO);
				if (bendCSPSO)
				{
					if (auto* v = bendCSPSO->GetStaticVariableByName(SHADER_TYPE_COMPUTE, "BendCB"))       v->Set(bendCB);
					if (auto* v = bendCSPSO->GetStaticVariableByName(SHADER_TYPE_COMPUTE, "BendCSParams")) v->Set(bendCSParamsCB);
					bendCSPSO->CreateShaderResourceBinding(&bendCSSRB, true);
					if (bendCSSRB)
					{
						if (auto* v = bendCSSRB->GetVariableByName(SHADER_TYPE_COMPUTE, "BendCB"))       v->Set(bendCB);
						if (auto* v = bendCSSRB->GetVariableByName(SHADER_TYPE_COMPUTE, "BendCSParams")) v->Set(bendCSParamsCB);
					}
				}
				cout << "[NukeDiligent]	foliage bend CS " << (bendCSPSO && bendCSSRB ? "ready" : "FAILED") << endl;
			}
		}
	}

	// GPU skinning compute (skin.cs): morphs + LBS into the skinned instance's buffers.
	// Not RT-gated — every skinned character uses it; per-instance SRBs bind the streams.
	{
		std::string cs = shaderSource("skin.cs");
		if (!cs.empty())
		{
			ShaderCreateInfo sci; sci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
			sci.pShaderSourceStreamFactory = shaderFactory;
			sci.Desc = {"Skin CS", SHADER_TYPE_COMPUTE, true};
			sci.Source = cs.c_str();
			RefCntAutoPtr<IShader> csh; CreateShaderCached(sci, &csh);
			if (csh)
			{
				BufferDesc cpb; cpb.Name = "SkinCS Params"; cpb.Size = sizeof(uint32_t) * 4;
				cpb.Usage = USAGE_DYNAMIC; cpb.BindFlags = BIND_UNIFORM_BUFFER; cpb.CPUAccessFlags = CPU_ACCESS_WRITE;
				device->CreateBuffer(cpb, nullptr, &skinCSParamsCB);
				ComputePipelineStateCreateInfo cci; cci.PSODesc.Name = "Skin PSO";
				ShaderResourceVariableDesc cvars[] = {
					{SHADER_TYPE_COMPUTE, "g_Palette",     SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
					{SHADER_TYPE_COMPUTE, "g_BindPos",     SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
					{SHADER_TYPE_COMPUTE, "g_BindNrm",     SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
					{SHADER_TYPE_COMPUTE, "g_BoneIdx",     SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
					{SHADER_TYPE_COMPUTE, "g_BoneWgt",     SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
					{SHADER_TYPE_COMPUTE, "g_MorphDelta",  SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
					{SHADER_TYPE_COMPUTE, "g_MorphWeight", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
					{SHADER_TYPE_COMPUTE, "g_PosOut",      SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
					{SHADER_TYPE_COMPUTE, "g_NrmOut",      SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
					{SHADER_TYPE_COMPUTE, "g_PosPrev",     SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
				};
				cci.PSODesc.ResourceLayout.Variables = cvars; cci.PSODesc.ResourceLayout.NumVariables = 10;
				cci.pCS = csh;
				device->CreateComputePipelineState(cci, &skinCSPSO);
				if (skinCSPSO)
				{
					if (auto* v = skinCSPSO->GetStaticVariableByName(SHADER_TYPE_COMPUTE, "SkinCSParams")) v->Set(skinCSParamsCB);
					skinCSPSO->CreateShaderResourceBinding(&skinCSSRB, true);
				}
				cout << "[NukeDiligent]	skin CS " << (skinCSPSO && skinCSSRB ? "ready" : "FAILED") << endl;
			}
		}
	}

	// 1x1 white fallback texture (bound when a material has no texture).
	uint32_t white = 0xFFFFFFFFu;
	TextureDesc wd; wd.Type = RESOURCE_DIM_TEX_2D; wd.Width = 1; wd.Height = 1;
	wd.Format = TEX_FORMAT_RGBA8_UNORM; wd.BindFlags = BIND_SHADER_RESOURCE; wd.Usage = USAGE_IMMUTABLE;
	TextureSubResData wsr; wsr.pData = &white; wsr.Stride = 4;
	TextureData wdat; wdat.pSubResources = &wsr; wdat.NumSubresources = 1;
	device->CreateTexture(wd, &wdat, &whiteTex);

	// 1x1 flat normal (R=128,G=128,B=255 -> +Z), bound when a material has no normal map.
	uint32_t flatN = 0xFFFF8080u;   // RGBA8 little-endian: R=0x80 G=0x80 B=0xFF A=0xFF
	TextureSubResData nsr; nsr.pData = &flatN; nsr.Stride = 4;
	TextureData ndat; ndat.pSubResources = &nsr; ndat.NumSubresources = 1;
	device->CreateTexture(wd, &ndat, &flatNormTex);

	// Probe sampler (linear, clamp) attached to each probe/fallback cube SRV: combined-texture-sampler
	// mode reads it from the view, so g_Probe needs no immutable sampler in the world PSO.
	{
		SamplerDesc ps; ps.MinFilter = FILTER_TYPE_LINEAR; ps.MagFilter = FILTER_TYPE_LINEAR; ps.MipFilter = FILTER_TYPE_LINEAR;
		ps.AddressU = TEXTURE_ADDRESS_CLAMP; ps.AddressV = TEXTURE_ADDRESS_CLAMP; ps.AddressW = TEXTURE_ADDRESS_CLAMP;
		device->CreateSampler(ps, &probeSampler);
	}
	// 1x1 black cube bound to g_Probe when no reflection probe is active (the var must stay valid).
	{
		uint16_t blackF16[4] = { 0, 0, 0, 0x3C00 };   // half (0,0,0,1)
		TextureDesc cd; cd.Name = "Fallback Cube"; cd.Type = RESOURCE_DIM_TEX_CUBE; cd.Width = 1; cd.Height = 1;
		cd.ArraySize = 6; cd.MipLevels = 1; cd.Format = HDR_FMT; cd.BindFlags = BIND_SHADER_RESOURCE; cd.Usage = USAGE_IMMUTABLE;
		TextureSubResData subs[6]; for (int f = 0; f < 6; ++f) { subs[f].pData = blackF16; subs[f].Stride = 8; }
		TextureData cdat; cdat.pSubResources = subs; cdat.NumSubresources = 6;
		device->CreateTexture(cd, &cdat, &fallbackCube);
		if (fallbackCube) { fallbackCubeSRV = fallbackCube->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
			if (fallbackCubeSRV && probeSampler) fallbackCubeSRV->SetSampler(probeSampler); }
	}

	// Clamp requested MSAA to what the device supports for BOTH color (RGBA8) and depth (D32).
	// SampleCounts is a bitmask whose bit VALUE equals the sample count.
	{
		Uint32 colorSC = (Uint32)device->GetTextureFormatInfoExt(TEX_FORMAT_RGBA8_UNORM).SampleCounts;
		Uint32 depthSC = (Uint32)device->GetTextureFormatInfoExt(TEX_FORMAT_D32_FLOAT).SampleCounts;
		Uint32 want = samples;
		while (want > 1 && !((colorSC & want) && (depthSC & want))) want >>= 1;
		samples = (Uint8)(want < 1 ? 1 : want);
		std::cout << "[NukeDiligent]\tMSAA samples = " << (int)samples << std::endl;
	}

	defaultWorldHandle = MakeWorldPSO(shaderSource("world.vs"), shaderSource("world.ps"), "World");

	BuildOutlinePipelines();   // selection outline (stencil mark + scaled draw)
	CreateShadowResources();   // directional shadow map + depth PSO
	CreateSkyResources();      // procedural sky pipeline
	CreateDebugResources();    // debug/gizmo line pipeline
	CreateSpriteResources();   // 2D sprite quad pipeline (SceneFmt + MSAA -> rebuild with them)
	CreateDecalResources();    // screen-space decal pipeline (SceneFmt + MSAA)
	CreatePostResources();     // final tonemap / post-process pass
	const TEXTURE_FORMAT fmt0 = SceneFmt();
	skyStamp.stamp(samples, fmt0); debugStamp.stamp(samples, fmt0); spriteStamp.stamp(samples, fmt0);
	decalStamp.stamp(samples, fmt0); outlineStamp.stamp(samples, fmt0);
	// The renderer's own pipelines join the same warm-up pump the modules use: everything built
	// after this point (materials arriving with a world, an MSAA change) lands off the draw path.
	{
		WarmEntry e;
		e.name = "NukeDiligent";
		e.fn = [](void* u) { return ((Impl*)u)->WarmEnginePipelines(); };
		e.user = this;
		warmups.push_back(e);
	}
}

// Build the selection-outline pipelines: mask (mesh -> RGBA8, alpha=1) and edge (fullscreen
// edge-detect drawing a constant-pixel-thickness border).
void NukeDiligent::Impl::BuildOutlinePipelines()
{
	// Rebuild path (MSAA change re-calls this): release prior objects or Create asserts.
	outlineMaskPSO.Release(); outlineMaskSRB.Release();
	outlineEdgePSO.Release(); outlineEdgeSRB.Release(); outlineEdgeCB.Release();
	ShaderCreateInfo sci; sci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;

	// mask pipeline (mesh -> mask RT)
	std::string mvs = shaderSource("outline.vs"), mps = shaderSource("outline.ps");
	if (!mvs.empty() && !mps.empty())
	{
		RefCntAutoPtr<IShader> vs, ps;
		sci.Desc = {"Outline Mask VS", SHADER_TYPE_VERTEX, true}; sci.Source = mvs.c_str(); CreateShaderCached(sci, &vs);
		sci.Desc = {"Outline Mask PS", SHADER_TYPE_PIXEL, true};  sci.Source = mps.c_str(); CreateShaderCached(sci, &ps);
		if (vs && ps)
		{
			GraphicsPipelineStateCreateInfo ci;
			ci.PSODesc.Name = "Outline Mask PSO";
			auto& gp = ci.GraphicsPipeline;
			gp.NumRenderTargets             = 1;
			gp.RTVFormats[0]                = TEX_FORMAT_RGBA8_UNORM;
			gp.DSVFormat                    = TEX_FORMAT_UNKNOWN;   // no depth
			gp.PrimitiveTopology            = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
			gp.RasterizerDesc.CullMode      = CULL_MODE_NONE;
			gp.DepthStencilDesc.DepthEnable = False;
			LayoutElement layout[] = { {0, 0, 3, VT_FLOAT32}, {1, 1, 3, VT_FLOAT32}, {2, 2, 2, VT_FLOAT32} };
			gp.InputLayout.NumElements    = 3;
			gp.InputLayout.LayoutElements = layout;
			ci.pVS = vs; ci.pPS = ps;
			CreateGraphicsPipelineStateCached(ci, &outlineMaskPSO);
			if (outlineMaskPSO)
			{
				if (auto* v = outlineMaskPSO->GetStaticVariableByName(SHADER_TYPE_VERTEX, "CB")) v->Set(worldCB);
				outlineMaskPSO->CreateShaderResourceBinding(&outlineMaskSRB, true);
			}
		}
	}

	// edge pipeline (fullscreen mask -> camera RT)
	std::string evs = shaderSource("outline_edge.vs"), eps = shaderSource("outline_edge.ps");
	if (!evs.empty() && !eps.empty())
	{
		BufferDesc cbd; cbd.Name = "Outline EdgeCB"; cbd.Size = sizeof(float) * 4;
		cbd.Usage = USAGE_DYNAMIC; cbd.BindFlags = BIND_UNIFORM_BUFFER; cbd.CPUAccessFlags = CPU_ACCESS_WRITE;
		device->CreateBuffer(cbd, nullptr, &outlineEdgeCB);

		RefCntAutoPtr<IShader> vs, ps;
		sci.Desc = {"Outline Edge VS", SHADER_TYPE_VERTEX, true}; sci.Source = evs.c_str(); CreateShaderCached(sci, &vs);
		sci.Desc = {"Outline Edge PS", SHADER_TYPE_PIXEL, true};  sci.Source = eps.c_str(); CreateShaderCached(sci, &ps);
		if (vs && ps)
		{
			GraphicsPipelineStateCreateInfo ci;
			ci.PSODesc.Name = "Outline Edge PSO";
			auto& gp = ci.GraphicsPipeline;
			gp.NumRenderTargets             = 1;
			gp.RTVFormats[0]                = SceneFmt();   // composites into the scene target
			gp.DSVFormat                    = TEX_FORMAT_UNKNOWN;
			gp.PrimitiveTopology            = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
			gp.RasterizerDesc.CullMode      = CULL_MODE_NONE;
			gp.DepthStencilDesc.DepthEnable = False;
			gp.SmplDesc.Count               = samples;   // MSAA: edge composites into the MS camera target
			gp.InputLayout.NumElements      = 0;   // fullscreen triangle from SV_VertexID, no VB
			ShaderResourceVariableDesc vars[] = {{SHADER_TYPE_PIXEL, "g_Mask", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC}};
			ci.PSODesc.ResourceLayout.Variables    = vars;
			ci.PSODesc.ResourceLayout.NumVariables = 1;
			SamplerDesc samp; samp.MinFilter = FILTER_TYPE_POINT; samp.MagFilter = FILTER_TYPE_POINT; samp.MipFilter = FILTER_TYPE_POINT;
			samp.AddressU = TEXTURE_ADDRESS_CLAMP; samp.AddressV = TEXTURE_ADDRESS_CLAMP; samp.AddressW = TEXTURE_ADDRESS_CLAMP;
			ImmutableSamplerDesc imm[] = {{SHADER_TYPE_PIXEL, "g_Mask", samp}};
			ci.PSODesc.ResourceLayout.ImmutableSamplers    = imm;
			ci.PSODesc.ResourceLayout.NumImmutableSamplers = 1;
			ci.pVS = vs; ci.pPS = ps;
			CreateGraphicsPipelineStateCached(ci, &outlineEdgePSO);
			if (outlineEdgePSO)
			{
				if (auto* c = outlineEdgePSO->GetStaticVariableByName(SHADER_TYPE_PIXEL, "EdgeCB")) c->Set(outlineEdgeCB);
				outlineEdgePSO->CreateShaderResourceBinding(&outlineEdgeSRB, true);
				if (outlineEdgeSRB) outlineEdgeMaskVar = outlineEdgeSRB->GetVariableByName(SHADER_TYPE_PIXEL, "g_Mask");
			}
		}
	}
}

// Lazily (re)create the selection mask RT to match the current camera target size.
void NukeDiligent::Impl::EnsureOutlineMask(int w, int h)
{
	if (w <= 0 || h <= 0) return;
	if (outlineMaskTex && outlineMaskW == w && outlineMaskH == h) return;
	Trash(outlineMaskTex);   // this frame's earlier passes may still reference it
	outlineMaskRTV = nullptr; outlineMaskSRV = nullptr; outlineMaskTex.Release();
	TextureDesc td; td.Name = "Outline Mask"; td.Type = RESOURCE_DIM_TEX_2D;
	td.Width = (Uint32)w; td.Height = (Uint32)h;
	td.Format = TEX_FORMAT_RGBA8_UNORM; td.BindFlags = BIND_RENDER_TARGET | BIND_SHADER_RESOURCE;
	device->CreateTexture(td, nullptr, &outlineMaskTex);
	if (outlineMaskTex)
	{
		outlineMaskRTV = outlineMaskTex->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET);
		outlineMaskSRV = outlineMaskTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
		outlineMaskW = w; outlineMaskH = h;
	}
}

// Build (or rebuild in place) the blend-variant PSOs + SRB into `wp` at the current `samples`.
// Rebuilding in place keeps existing material->shader handles valid.
bool NukeDiligent::Impl::BuildWorldPipe(WorldPipe& wp, const std::string& vsSrc, const std::string& psSrc, const char* dbg)
{
	if (vsSrc.empty() || psSrc.empty()) return false;
	ShaderCreateInfo sci;
	sci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
	sci.pShaderSourceStreamFactory = shaderFactory;   // resolves #include "nukebend.hlsl"
	// Inline RayQuery needs DXC at SM6.5; unsupported devices keep the default FXC SM5 path.
	ShaderMacro rtMacro[] = {{"RT_ENABLED", "1"}};
	if (rtSupported)
	{
		sci.ShaderCompiler = SHADER_COMPILER_DXC;
		sci.HLSLVersion    = ShaderVersion{6, 5};
		sci.Macros         = ShaderMacroArray{rtMacro, 1};
	}
	RefCntAutoPtr<IShader> vs, ps;
	sci.Desc = {dbg, SHADER_TYPE_VERTEX, true}; sci.Source = vsSrc.c_str(); CreateShaderCached(sci, &vs);
	sci.Desc = {dbg, SHADER_TYPE_PIXEL, true};  sci.Source = psSrc.c_str(); CreateShaderCached(sci, &ps);
	if (!vs || !ps) return false;

	GraphicsPipelineStateCreateInfo ci;
	ci.PSODesc.Name = dbg;
	auto& gp = ci.GraphicsPipeline;
	gp.NumRenderTargets             = 1;
	gp.RTVFormats[0]                = SceneFmt();   // HDR (post tonemaps) or RGBA8 (world.ps tonemaps)
	gp.DSVFormat                    = TEX_FORMAT_D32_FLOAT;
	gp.PrimitiveTopology            = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	gp.RasterizerDesc.CullMode      = CULL_MODE_NONE;
	gp.DepthStencilDesc.DepthEnable = True;
	gp.SmplDesc.Count               = samples;   // MSAA: must match the (MS) camera target
	LayoutElement layout[] = {
		{0, 0, 3, VT_FLOAT32}, // position
		{1, 1, 3, VT_FLOAT32}, // normal
		{2, 2, 2, VT_FLOAT32}, // uv
		{3, 3, 4, VT_FLOAT32}, // vertex color (opt-in via the NUKE_VCOLOR marker in the VS source)
	};
	gp.InputLayout.NumElements    = vsSrc.find("NUKE_VCOLOR") != std::string::npos ? 4 : 3;
	gp.InputLayout.LayoutElements = layout;

	ShaderResourceVariableDesc varsBase[] = {
		{SHADER_TYPE_PIXEL, "g_Tex",        SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
		{SHADER_TYPE_PIXEL, "g_Normal",     SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
		{SHADER_TYPE_PIXEL, "g_MetalRough", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
		{SHADER_TYPE_PIXEL, "g_Occlusion",  SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
		{SHADER_TYPE_PIXEL, "g_Emissive",   SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
		{SHADER_TYPE_PIXEL, "g_Spec",       SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
		{SHADER_TYPE_PIXEL, "g_WipeMask",   SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},   // luma-wipe mask (LM-3)
		{SHADER_TYPE_PIXEL, "g_Height",     SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},   // POM/displacement height (LM-3)
		{SHADER_TYPE_PIXEL, "g_Shadow",     SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
		{SHADER_TYPE_PIXEL, "g_ShadowCube", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
		{SHADER_TYPE_PIXEL, "g_Probe",      SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},   // reflection probe cubemap
		{SHADER_TYPE_PIXEL, "g_TLAS",       SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},   // RT scene (only present when rtSupported)
		{SHADER_TYPE_PIXEL, "g_RTInst",     SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},   // per-instance RT data (shadow footprints)
	};
	// The last two base entries are RT-only — drop them when the device has no ray tracing.
	const Uint32 kNumBase = (Uint32)(sizeof(varsBase) / sizeof(varsBase[0]));
	std::vector<ShaderResourceVariableDesc> vars(varsBase, varsBase + (rtSupported ? kNumBase : kNumBase - 2));
	// Overlay slots (LM-3 states/layers), OvTexNames() order; one shared sampler on g_Ov0Alb.
	for (const std::string& n : OvTexNames())
		vars.push_back({SHADER_TYPE_PIXEL, n.c_str(), SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC});
	// BRDF pack (LM-6): flow map + the pre-transparent scene snapshot (shared sampler too).
	vars.push_back({SHADER_TYPE_PIXEL, "g_Flow",      SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC});
	vars.push_back({SHADER_TYPE_PIXEL, "g_MskStamp",  SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC});
	vars.push_back({SHADER_TYPE_PIXEL, "g_SceneRefr", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC});
	ci.PSODesc.ResourceLayout.Variables    = vars.data();
	ci.PSODesc.ResourceLayout.NumVariables = (Uint32)vars.size();
	SamplerDesc samp; samp.MinFilter = FILTER_TYPE_LINEAR; samp.MagFilter = FILTER_TYPE_LINEAR; samp.MipFilter = FILTER_TYPE_LINEAR;
	samp.AddressU = TEXTURE_ADDRESS_WRAP; samp.AddressV = TEXTURE_ADDRESS_WRAP; samp.AddressW = TEXTURE_ADDRESS_WRAP;
	// Combined-sampler mode on D3D12 samples texture X via X_sampler, so each map needs its OWN
	// immutable sampler, and only for the maps this shader actually declares (unassigned ones warn).
	static const char* const kMapTex[] = { "g_Tex", "g_Normal", "g_MetalRough", "g_Occlusion", "g_Emissive", "g_Spec", "g_WipeMask", "g_Height",
	                                       "g_Ov0Alb" };   // ONE sampler serves the whole overlay block (D3D11 sampler cap)
	ImmutableSamplerDesc immSamp[9]; Uint32 nImm = 0;
	ps->GetStatus(true);   // async compile (cache miss): reflection needs the FINISHED shader
	const Uint32 nRes = ps->GetResourceCount();
	for (const char* nm : kMapTex)
		for (Uint32 r = 0; r < nRes; ++r)
		{
			ShaderResourceDesc rd; ps->GetResourceDesc(r, rd);
			if (rd.Type == SHADER_RESOURCE_TYPE_TEXTURE_SRV && std::string(rd.Name) == nm)
			{ immSamp[nImm++] = ImmutableSamplerDesc{SHADER_TYPE_PIXEL, nm, samp}; break; }
		}
	ci.PSODesc.ResourceLayout.ImmutableSamplers    = immSamp;
	ci.PSODesc.ResourceLayout.NumImmutableSamplers = nImm;
	ci.pVS = vs; ci.pPS = ps;

	auto setStatics = [&](IPipelineState* pso) {
		if (auto* v = pso->GetStaticVariableByName(SHADER_TYPE_VERTEX, "CB"))      v->Set(worldCB);
		if (auto* m = pso->GetStaticVariableByName(SHADER_TYPE_PIXEL,  "MatCB"))   m->Set(worldMatCB);
		if (auto* f = pso->GetStaticVariableByName(SHADER_TYPE_PIXEL,  "FrameCB")) f->Set(worldFrameCB);
		if (auto* b = pso->GetStaticVariableByName(SHADER_TYPE_VERTEX, "BendCB"))  b->Set(bendCB);   // instanced variants only (7.4)
		if (auto* d = pso->GetStaticVariableByName(SHADER_TYPE_PIXEL,  "DrawFlagsCB")) d->Set(drawFlagsCB);   // receiveShadows etc.
	};

	// Rebuild path: release first — Diligent asserts on Create over a non-null ref.
	wp.pso.Release(); wp.psoBlend.Release(); wp.psoAdd.Release(); wp.psoWire.Release(); wp.srb.Release();
	memset(wp.lastBind, 0, sizeof(wp.lastBind)); memset(wp.lastBindI, 0, sizeof(wp.lastBindI));   // fresh SRBs hold nothing

	// 1) Opaque — blend off, depth write on.
	CreateGraphicsPipelineStateCached(ci, &wp.pso);
	if (!wp.pso) { cout << "[NukeDiligent]\tPSO build failed for shader '" << dbg << "'" << endl; return false; }
	setStatics(wp.pso);

	// 2) Transparent — straight-alpha blend, depth test on but no depth write.
	{
		auto& rt = ci.GraphicsPipeline.BlendDesc.RenderTargets[0];
		rt.BlendEnable = True;
		rt.SrcBlend = BLEND_FACTOR_SRC_ALPHA; rt.DestBlend = BLEND_FACTOR_INV_SRC_ALPHA; rt.BlendOp = BLEND_OPERATION_ADD;
		rt.SrcBlendAlpha = BLEND_FACTOR_ONE;  rt.DestBlendAlpha = BLEND_FACTOR_INV_SRC_ALPHA; rt.BlendOpAlpha = BLEND_OPERATION_ADD;
		ci.GraphicsPipeline.DepthStencilDesc.DepthWriteEnable = False;
		ci.PSODesc.Name = "World (blend)";
		CreateGraphicsPipelineStateCached(ci, &wp.psoBlend);
		if (wp.psoBlend) setStatics(wp.psoBlend);
	}
	// 3) Additive — src*a + dst, depth write off.
	{
		auto& rt = ci.GraphicsPipeline.BlendDesc.RenderTargets[0];
		rt.BlendEnable = True;
		rt.SrcBlend = BLEND_FACTOR_SRC_ALPHA; rt.DestBlend = BLEND_FACTOR_ONE; rt.BlendOp = BLEND_OPERATION_ADD;
		rt.SrcBlendAlpha = BLEND_FACTOR_ONE;  rt.DestBlendAlpha = BLEND_FACTOR_ONE; rt.BlendOpAlpha = BLEND_OPERATION_ADD;
		ci.PSODesc.Name = "World (add)";
		CreateGraphicsPipelineStateCached(ci, &wp.psoAdd);
		if (wp.psoAdd) setStatics(wp.psoAdd);
	}
	// 4) Wireframe — opaque state with line fill.
	{
		ci.GraphicsPipeline.BlendDesc.RenderTargets[0] = RenderTargetBlendDesc{};
		ci.GraphicsPipeline.DepthStencilDesc.DepthWriteEnable = True;
		ci.GraphicsPipeline.RasterizerDesc.FillMode = FILL_MODE_WIREFRAME;
		ci.PSODesc.Name = "World (wire)";
		CreateGraphicsPipelineStateCached(ci, &wp.psoWire);
		if (wp.psoWire) setStatics(wp.psoWire);
	}

	// 4b) Vertex-color variants (opt-in: sources handle NUKE_VCTINT). Same resource layout as
	//     the plain PSOs, so wp.srb serves them; chosen per draw when the mesh has a color
	//     stream and the material asks for tint/overlay-mask.
	wp.psoVcol.Release(); wp.psoVcolBlend.Release(); wp.psoVcolAdd.Release();
	if (vsSrc.find("NUKE_VCTINT") != std::string::npos && psSrc.find("NUKE_VCTINT") != std::string::npos)
	{
		const std::string vsV = "#define NUKE_VCTINT 1\n" + vsSrc;
		const std::string psV = "#define NUKE_VCTINT 1\n" + psSrc;
		ShaderCreateInfo sciV; sciV.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
		sciV.pShaderSourceStreamFactory = shaderFactory;
		RefCntAutoPtr<IShader> vsv, psv;
		sciV.Desc = {"World VS (vcol)", SHADER_TYPE_VERTEX, true}; sciV.Source = vsV.c_str(); CreateShaderCached(sciV, &vsv);
		sciV.Desc = {"World PS (vcol)", SHADER_TYPE_PIXEL,  true}; sciV.Source = psV.c_str(); CreateShaderCached(sciV, &psv);
		if (vsv && psv)
		{
			LayoutElement layoutV[] = {
				{0, 0, 3, VT_FLOAT32}, {1, 1, 3, VT_FLOAT32}, {2, 2, 2, VT_FLOAT32},
				{3, 3, 4, VT_FLOAT32},   // vertex color stream
			};
			ci.GraphicsPipeline.InputLayout.LayoutElements = layoutV;
			ci.GraphicsPipeline.InputLayout.NumElements    = 4;
			ci.pVS = vsv; ci.pPS = psv;
			ci.GraphicsPipeline.RasterizerDesc.FillMode = FILL_MODE_SOLID;
			ci.GraphicsPipeline.BlendDesc.RenderTargets[0] = RenderTargetBlendDesc{};
			ci.GraphicsPipeline.DepthStencilDesc.DepthWriteEnable = True;
			ci.PSODesc.Name = "World (vcol)";
			CreateGraphicsPipelineStateCached(ci, &wp.psoVcol);
			if (wp.psoVcol) setStatics(wp.psoVcol);
			{
				auto& rt = ci.GraphicsPipeline.BlendDesc.RenderTargets[0];
				rt.BlendEnable = True; rt.SrcBlend = BLEND_FACTOR_SRC_ALPHA; rt.DestBlend = BLEND_FACTOR_INV_SRC_ALPHA;
				rt.BlendOp = BLEND_OPERATION_ADD;
				rt.SrcBlendAlpha = BLEND_FACTOR_ONE; rt.DestBlendAlpha = BLEND_FACTOR_INV_SRC_ALPHA; rt.BlendOpAlpha = BLEND_OPERATION_ADD;
				ci.GraphicsPipeline.DepthStencilDesc.DepthWriteEnable = False;
				ci.PSODesc.Name = "World (vcol blend)";
				CreateGraphicsPipelineStateCached(ci, &wp.psoVcolBlend);
				if (wp.psoVcolBlend) setStatics(wp.psoVcolBlend);
			}
			{
				auto& rt = ci.GraphicsPipeline.BlendDesc.RenderTargets[0];
				rt.BlendEnable = True; rt.SrcBlend = BLEND_FACTOR_SRC_ALPHA; rt.DestBlend = BLEND_FACTOR_ONE;
				rt.BlendOp = BLEND_OPERATION_ADD;
				rt.SrcBlendAlpha = BLEND_FACTOR_ONE; rt.DestBlendAlpha = BLEND_FACTOR_ONE; rt.BlendOpAlpha = BLEND_OPERATION_ADD;
				ci.PSODesc.Name = "World (vcol add)";
				CreateGraphicsPipelineStateCached(ci, &wp.psoVcolAdd);
				if (wp.psoVcolAdd) setStatics(wp.psoVcolAdd);
			}
			// Restore the shared state for the blocks below.
			ci.GraphicsPipeline.BlendDesc.RenderTargets[0] = RenderTargetBlendDesc{};
			ci.GraphicsPipeline.DepthStencilDesc.DepthWriteEnable = True;
			ci.GraphicsPipeline.InputLayout.LayoutElements = layout;
			ci.GraphicsPipeline.InputLayout.NumElements    = vsSrc.find("NUKE_VCOLOR") != std::string::npos ? 4 : 3;
			ci.pVS = vs; ci.pPS = ps;
		}
	}

	// 5) Displacement tessellation — opt-in (VS handles NUKE_TESS + PS declares g_Disp) and
	//    device-gated. Opaque state, patch-list topology; the DS displaces along the normal by
	//    g_Height and emits the same PSIn, so the SAME pixel shader lights the result.
	wp.psoTess.Release(); wp.srbTess.Release();
	if (device->GetDeviceInfo().Features.Tessellation
	    && vsSrc.find("NUKE_TESS") != std::string::npos && psSrc.find("g_Disp") != std::string::npos)
	{
		const std::string hsSrc = shaderSource("world.hs"), dsSrc = shaderSource("world.ds");
		if (!hsSrc.empty() && !dsSrc.empty())
		{
			const std::string vsT = "#define NUKE_TESS 1\n" + vsSrc;
			ShaderCreateInfo sciT; sciT.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
			sciT.pShaderSourceStreamFactory = shaderFactory;
			// ALL tess-PSO stages through DXC (water precedent): glslang hull/domain SPIR-V
			// breaks on NVIDIA — the driver dies inside pipeline creation.
			sciT.ShaderCompiler = SHADER_COMPILER_DXC;
			RefCntAutoPtr<IShader> vst, hst, dst, pst;
			sciT.Desc = {"World VS (tess)", SHADER_TYPE_VERTEX, true}; sciT.Source = vsT.c_str();   CreateShaderCached(sciT, &vst);
			sciT.Desc = {"World HS",        SHADER_TYPE_HULL,   true}; sciT.Source = hsSrc.c_str(); CreateShaderCached(sciT, &hst);
			sciT.Desc = {"World DS",        SHADER_TYPE_DOMAIN, true}; sciT.Source = dsSrc.c_str(); CreateShaderCached(sciT, &dst);
			sciT.Desc = {"World PS (tess)", SHADER_TYPE_PIXEL,  true}; sciT.Source = psSrc.c_str(); CreateShaderCached(sciT, &pst);
			if (vst && hst && dst && pst)
			{
				ci.GraphicsPipeline.RasterizerDesc.FillMode = FILL_MODE_SOLID;
				ci.GraphicsPipeline.BlendDesc.RenderTargets[0] = RenderTargetBlendDesc{};
				ci.GraphicsPipeline.DepthStencilDesc.DepthWriteEnable = True;
				ci.GraphicsPipeline.PrimitiveTopology = PRIMITIVE_TOPOLOGY_3_CONTROL_POINT_PATCHLIST;
				std::vector<ShaderResourceVariableDesc> varsT(vars);
				varsT.push_back({SHADER_TYPE_DOMAIN, "g_Height", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC});
				ci.PSODesc.ResourceLayout.Variables    = varsT.data();
				ci.PSODesc.ResourceLayout.NumVariables = (Uint32)varsT.size();
				std::vector<ImmutableSamplerDesc> immT(immSamp, immSamp + nImm);
				immT.push_back(ImmutableSamplerDesc{SHADER_TYPE_DOMAIN, "g_Height", samp});
				ci.PSODesc.ResourceLayout.ImmutableSamplers    = immT.data();
				ci.PSODesc.ResourceLayout.NumImmutableSamplers = (Uint32)immT.size();
				ci.PSODesc.Name = "World (tess)";
				ci.pVS = vst; ci.pHS = hst; ci.pDS = dst; ci.pPS = pst;
				CreateGraphicsPipelineStateCached(ci, &wp.psoTess);
				ci.pHS = nullptr; ci.pDS = nullptr; ci.pVS = vs; ci.pPS = ps;
				ci.GraphicsPipeline.PrimitiveTopology = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
				ci.PSODesc.ResourceLayout.Variables    = vars.data();
				ci.PSODesc.ResourceLayout.NumVariables = (Uint32)vars.size();
				ci.PSODesc.ResourceLayout.ImmutableSamplers    = immSamp;
				ci.PSODesc.ResourceLayout.NumImmutableSamplers = nImm;
				cout << "[NukeDiligent]\tworld tess PSO " << (wp.psoTess ? "ready" : "FAILED") << " ('" << dbg << "')" << endl;
				if (wp.psoTess)
				{
					setStatics(wp.psoTess);
					// HS/DS see MatCB (tess factor + displacement params) and the DS projects via CB.
					if (auto* m = wp.psoTess->GetStaticVariableByName(SHADER_TYPE_HULL,   "MatCB")) m->Set(worldMatCB);
					if (auto* m = wp.psoTess->GetStaticVariableByName(SHADER_TYPE_DOMAIN, "MatCB")) m->Set(worldMatCB);
					if (auto* c = wp.psoTess->GetStaticVariableByName(SHADER_TYPE_DOMAIN, "CB"))    c->Set(worldCB);
					wp.psoTess->CreateShaderResourceBinding(&wp.srbTess, true);
					// Vulkan: cbuffers may reflect MUTABLE — bind through the SRB as well.
					if (auto* d = wp.srbTess->GetVariableByName(SHADER_TYPE_PIXEL,  "DrawFlagsCB")) d->Set(drawFlagsCB);
					if (auto* m = wp.srbTess->GetVariableByName(SHADER_TYPE_HULL,   "MatCB")) m->Set(worldMatCB);
					if (auto* m = wp.srbTess->GetVariableByName(SHADER_TYPE_DOMAIN, "MatCB")) m->Set(worldMatCB);
					if (auto* c = wp.srbTess->GetVariableByName(SHADER_TYPE_DOMAIN, "CB"))    c->Set(worldCB);
				}
			}
		}
	}

	wp.pso->CreateShaderResourceBinding(&wp.srb, true);
	// Vulkan: cbuffers may reflect as MUTABLE, so bind through the SRB as well as the statics.
	if (auto* d = wp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "DrawFlagsCB")) d->Set(drawFlagsCB);
	wp.texVar  = wp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Tex");
	wp.normVar = wp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Normal");
	wp.mrVar   = wp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_MetalRough");
	wp.aoVar   = wp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Occlusion");
	wp.emVar   = wp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Emissive");
	wp.specVar = wp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Spec");
	wp.wipeVar = wp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_WipeMask");
	wp.heightVar = wp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Height");
	wp.shadowVar = wp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Shadow");
	wp.cubeVar   = wp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_ShadowCube");
	wp.probeVar  = wp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Probe");
	wp.tlasVar   = wp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_TLAS");
	wp.rtInstVar = wp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_RTInst");
	for (int k = 0; k < kOvTexCount; ++k)
		wp.ovVar[k] = wp.srb->GetVariableByName(SHADER_TYPE_PIXEL, OvTexNames()[k].c_str());
	wp.flowVar = wp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Flow");
	wp.mskVar = wp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_MskStamp");
	wp.refrVar = wp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_SceneRefr");

	// Instanced variants — only for shaders that opt in by handling NUKE_INSTANCED. Same sources
	// with the define prepended; the layout gains 5 per-instance float4 attributes in slot 3.
	wp.psoInst.Release(); wp.psoInstBlend.Release(); wp.psoInstAdd.Release(); wp.psoInstWire.Release(); wp.srbInst.Release();
	wp.texVarI = wp.normVarI = wp.mrVarI = wp.aoVarI = wp.emVarI = wp.specVarI = wp.wipeVarI = wp.heightVarI = nullptr;
	wp.shadowVarI = wp.cubeVarI = wp.probeVarI = wp.tlasVarI = nullptr;
	memset(wp.ovVarI, 0, sizeof(wp.ovVarI));
	if (vsSrc.find("NUKE_INSTANCED") != std::string::npos && psSrc.find("NUKE_INSTANCED") != std::string::npos)
	{
		const std::string vsI = "#define NUKE_INSTANCED 1\n" + vsSrc;
		const std::string psI = "#define NUKE_INSTANCED 1\n" + psSrc;
		RefCntAutoPtr<IShader> vsi, psi;
		sci.Desc = {dbg, SHADER_TYPE_VERTEX, true}; sci.Source = vsI.c_str(); CreateShaderCached(sci, &vsi);
		sci.Desc = {dbg, SHADER_TYPE_PIXEL, true};  sci.Source = psI.c_str(); CreateShaderCached(sci, &psi);
		if (vsi && psi)
		{
			LayoutElement layoutI[] = {
				{0, 0, 3, VT_FLOAT32},   // position
				{1, 1, 3, VT_FLOAT32},   // normal
				{2, 2, 2, VT_FLOAT32},   // uv
				{3, 3, 4, VT_FLOAT32, False, LAYOUT_ELEMENT_AUTO_OFFSET, LAYOUT_ELEMENT_AUTO_STRIDE, INPUT_ELEMENT_FREQUENCY_PER_INSTANCE},   // world row 0
				{4, 3, 4, VT_FLOAT32, False, LAYOUT_ELEMENT_AUTO_OFFSET, LAYOUT_ELEMENT_AUTO_STRIDE, INPUT_ELEMENT_FREQUENCY_PER_INSTANCE},   // world row 1
				{5, 3, 4, VT_FLOAT32, False, LAYOUT_ELEMENT_AUTO_OFFSET, LAYOUT_ELEMENT_AUTO_STRIDE, INPUT_ELEMENT_FREQUENCY_PER_INSTANCE},   // world row 2
				{6, 3, 4, VT_FLOAT32, False, LAYOUT_ELEMENT_AUTO_OFFSET, LAYOUT_ELEMENT_AUTO_STRIDE, INPUT_ELEMENT_FREQUENCY_PER_INSTANCE},   // tint
				{7, 3, 4, VT_FLOAT32, False, LAYOUT_ELEMENT_AUTO_OFFSET, LAYOUT_ELEMENT_AUTO_STRIDE, INPUT_ELEMENT_FREQUENCY_PER_INSTANCE},   // custom
			};
			gp.InputLayout.LayoutElements = layoutI;
			gp.InputLayout.NumElements    = 8;
			ci.pVS = vsi; ci.pPS = psi;

			// 1) Opaque — reset the state the wireframe variant left behind.
			ci.GraphicsPipeline.BlendDesc.RenderTargets[0] = RenderTargetBlendDesc{};
			ci.GraphicsPipeline.DepthStencilDesc.DepthWriteEnable = True;
			ci.GraphicsPipeline.RasterizerDesc.FillMode = FILL_MODE_SOLID;
			ci.PSODesc.Name = "World (inst)";
			CreateGraphicsPipelineStateCached(ci, &wp.psoInst);
			if (wp.psoInst)
			{
				setStatics(wp.psoInst);
				// 2) Transparent.
				{
					auto& rt = ci.GraphicsPipeline.BlendDesc.RenderTargets[0];
					rt.BlendEnable = True;
					rt.SrcBlend = BLEND_FACTOR_SRC_ALPHA; rt.DestBlend = BLEND_FACTOR_INV_SRC_ALPHA; rt.BlendOp = BLEND_OPERATION_ADD;
					rt.SrcBlendAlpha = BLEND_FACTOR_ONE;  rt.DestBlendAlpha = BLEND_FACTOR_INV_SRC_ALPHA; rt.BlendOpAlpha = BLEND_OPERATION_ADD;
					ci.GraphicsPipeline.DepthStencilDesc.DepthWriteEnable = False;
					ci.PSODesc.Name = "World (inst blend)";
					CreateGraphicsPipelineStateCached(ci, &wp.psoInstBlend);
					if (wp.psoInstBlend) setStatics(wp.psoInstBlend);
				}
				// 3) Additive.
				{
					auto& rt = ci.GraphicsPipeline.BlendDesc.RenderTargets[0];
					rt.BlendEnable = True;
					rt.SrcBlend = BLEND_FACTOR_SRC_ALPHA; rt.DestBlend = BLEND_FACTOR_ONE; rt.BlendOp = BLEND_OPERATION_ADD;
					rt.SrcBlendAlpha = BLEND_FACTOR_ONE;  rt.DestBlendAlpha = BLEND_FACTOR_ONE; rt.BlendOpAlpha = BLEND_OPERATION_ADD;
					ci.PSODesc.Name = "World (inst add)";
					CreateGraphicsPipelineStateCached(ci, &wp.psoInstAdd);
					if (wp.psoInstAdd) setStatics(wp.psoInstAdd);
				}
				// 4) Wireframe.
				{
					ci.GraphicsPipeline.BlendDesc.RenderTargets[0] = RenderTargetBlendDesc{};
					ci.GraphicsPipeline.DepthStencilDesc.DepthWriteEnable = True;
					ci.GraphicsPipeline.RasterizerDesc.FillMode = FILL_MODE_WIREFRAME;
					ci.PSODesc.Name = "World (inst wire)";
					CreateGraphicsPipelineStateCached(ci, &wp.psoInstWire);
					if (wp.psoInstWire) setStatics(wp.psoInstWire);
				}
				wp.psoInst->CreateShaderResourceBinding(&wp.srbInst, true);
				// Vulkan: cbuffers may reflect as MUTABLE, and one unbound descriptor invalidates
				// the whole set — bind BendCB through the SRB as well as the statics.
				if (auto* b = wp.srbInst->GetVariableByName(SHADER_TYPE_VERTEX, "BendCB")) b->Set(bendCB);
				if (auto* d = wp.srbInst->GetVariableByName(SHADER_TYPE_PIXEL, "DrawFlagsCB")) d->Set(drawFlagsCB);
				wp.texVarI  = wp.srbInst->GetVariableByName(SHADER_TYPE_PIXEL, "g_Tex");
				wp.normVarI = wp.srbInst->GetVariableByName(SHADER_TYPE_PIXEL, "g_Normal");
				wp.mrVarI   = wp.srbInst->GetVariableByName(SHADER_TYPE_PIXEL, "g_MetalRough");
				wp.aoVarI   = wp.srbInst->GetVariableByName(SHADER_TYPE_PIXEL, "g_Occlusion");
				wp.emVarI   = wp.srbInst->GetVariableByName(SHADER_TYPE_PIXEL, "g_Emissive");
				wp.specVarI = wp.srbInst->GetVariableByName(SHADER_TYPE_PIXEL, "g_Spec");
				wp.wipeVarI = wp.srbInst->GetVariableByName(SHADER_TYPE_PIXEL, "g_WipeMask");
				wp.heightVarI = wp.srbInst->GetVariableByName(SHADER_TYPE_PIXEL, "g_Height");
				wp.shadowVarI = wp.srbInst->GetVariableByName(SHADER_TYPE_PIXEL, "g_Shadow");
				wp.cubeVarI   = wp.srbInst->GetVariableByName(SHADER_TYPE_PIXEL, "g_ShadowCube");
				wp.probeVarI  = wp.srbInst->GetVariableByName(SHADER_TYPE_PIXEL, "g_Probe");
				wp.tlasVarI   = wp.srbInst->GetVariableByName(SHADER_TYPE_PIXEL, "g_TLAS");
				wp.rtInstVarI = wp.srbInst->GetVariableByName(SHADER_TYPE_PIXEL, "g_RTInst");
				for (int k = 0; k < kOvTexCount; ++k)
					wp.ovVarI[k] = wp.srbInst->GetVariableByName(SHADER_TYPE_PIXEL, OvTexNames()[k].c_str());
				wp.flowVarI = wp.srbInst->GetVariableByName(SHADER_TYPE_PIXEL, "g_Flow");
				wp.mskVarI = wp.srbInst->GetVariableByName(SHADER_TYPE_PIXEL, "g_MskStamp");
				wp.refrVarI = wp.srbInst->GetVariableByName(SHADER_TYPE_PIXEL, "g_SceneRefr");
			}
			else
				cout << "[NukeDiligent]\tinstanced PSO build failed for shader '" << dbg << "'" << endl;
		}
	}
	return true;
}

// Registers a world pipeline for a shader pair. The build itself is left to the warm-up pump:
// a material arriving with a world load must not compile inside the frame that loads it. Until
// it is ready the draw uses the default world pipeline, so the object is shaded, not missing.
uint64_t NukeDiligent::Impl::MakeWorldPSO(const std::string& vsSrc, const std::string& psSrc, const char* dbg)
{
	WorldPipe wp;
	wp.vsSrc = vsSrc; wp.psSrc = psSrc; wp.dbg = dbg;   // kept so the warm-up can (re)build it
	// The default pipeline is the fallback for everything else, so it cannot be deferred.
	const bool isDefault = (defaultWorldHandle == 0);
	if (isDefault)
	{
		if (!BuildWorldPipe(wp, vsSrc, psSrc, dbg)) return 0;
		wp.builtSamples = samples; wp.builtFmt = SceneFmt();
	}
	uint64_t h = nextShaderHandle++;
	worldPipes[h] = std::move(wp);
	// Custom shaders can register AFTER the warm pump finished its sweep (module OnLoad, shader
	// hot-reload) — re-arm the renderer's own entry or the new pipe never builds and every draw
	// silently falls back to the default pipeline.
	for (WarmEntry& e : warmups) if (e.user == this) e.done = false;
	return h;
}

// Builds one stale pipeline set per call and returns false to be resumed next frame. Driven by
// the warm-up pump — nothing here may run from a draw.
bool NukeDiligent::Impl::WarmEnginePipelines()
{
	const TEXTURE_FORMAT fmt = SceneFmt();
	for (auto& kv : worldPipes)
	{
		WorldPipe& wp = kv.second;
		if (wp.buildFailed || (wp.builtSamples == samples && wp.builtFmt == fmt)) continue;
		if (!BuildWorldPipe(wp, wp.vsSrc, wp.psSrc, wp.dbg.c_str()))
		{
			wp.buildFailed = true;   // latch: the draw keeps falling back to the default pipe
			std::cout << "[NukeDiligent]\tworld pipeline FAILED: " << wp.dbg << std::endl;
			continue;
		}
		wp.builtSamples = samples; wp.builtFmt = fmt;
		return false;   // one pipeline per frame
	}
	if (!skyStamp.current(samples, fmt))     { CreateSkyResources();    skyStamp.stamp(samples, fmt);     return false; }
	if (!debugStamp.current(samples, fmt))   { CreateDebugResources();  debugStamp.stamp(samples, fmt);   return false; }
	if (!spriteStamp.current(samples, fmt))  { CreateSpriteResources(); spriteStamp.stamp(samples, fmt);  return false; }
	if (!decalStamp.current(samples, fmt))   { CreateDecalResources();  decalStamp.stamp(samples, fmt);   return false; }
	if (!outlineStamp.current(samples, fmt)) { BuildOutlinePipelines(); outlineStamp.stamp(samples, fmt); return false; }
	return true;
}

// Sample count or scene format changed. The targets follow immediately — they define what the
// passes render into — while every pipeline built against the old pair is marked stale and
// rebuilt by the warm-up over the next frames. Draws skip what is not current yet, so the
// change costs a moment of things fading back in instead of a frozen frame.
void NukeDiligent::Impl::RebuildForMSAA()
{
	skyStamp = debugStamp = spriteStamp = decalStamp = outlineStamp = PipeStamp{};
	// The default world pipeline is everyone's fallback: it is rebuilt here, not deferred.
	auto dit = worldPipes.find(defaultWorldHandle);
	if (dit != worldPipes.end())
	{
		WorldPipe& wp = dit->second;
		if (BuildWorldPipe(wp, wp.vsSrc, wp.psSrc, wp.dbg.c_str()))
		{
			wp.builtSamples = samples; wp.builtFmt = SceneFmt();
		}
	}
	for (WarmEntry& e : warmups) e.done = false;   // every builder gets another look
	TrashRT(backbufferMS);
	backbufferMS = RT{};   // recreated on next target-0 camera
	for (auto& kv : rts)
		if (kv.second.w > 0 && kv.second.h > 0)
		{
			RT old = kv.second;              // last frame's UI draw data may still be in flight
			kv.second = MakeRT(kv.second.w, kv.second.h);
			TrashRT(old);
		}
	// Cached UI SRBs key views that were just replaced — park them all; the cache refills on next draw.
	for (auto& kv : uiSRBCache) Trash(kv.second.srb);
	uiSRBCache.clear();
}

uint64_t NukeDiligent::createShaderPipeline(const char* name, const char* vs, const char* ps)
{
	if (!vs || !ps) return 0;
	uint64_t h = m_impl->MakeWorldPSO(vs, ps, "Shader");   // world-type PSO (layout/CBs) from custom VS+PS
	// A shader shipping "<name>.surf.hlsl" gets an auto-generated RT closest-hit group.
	if (h && name && *name && m_impl->rtSupported && m_impl->shaderFactory)
	{
		RefCntAutoPtr<IFileStream> stream;
		m_impl->shaderFactory->CreateInputStream2((std::string(name) + ".surf.hlsl").c_str(),
		                                          CREATE_SHADER_SOURCE_INPUT_STREAM_FLAG_SILENT, &stream);
		if (stream)
		{
			std::string& slot = m_impl->rtSurfShaders[name];
			if (slot != ps) { slot = ps; m_impl->rtPipelineDirty = true; }   // (re)build the RT pipeline to add/refresh this hit group
		}
	}
	return h;
}
