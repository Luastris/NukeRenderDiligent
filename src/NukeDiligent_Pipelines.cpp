#include "NukeDiligentImpl.h"

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
	device->CreateShader(ShaderCI, &vs);
	ShaderCI.Desc = {"UI PS", SHADER_TYPE_PIXEL, true};
	ShaderCI.Source = psSrc.c_str();
	device->CreateShader(ShaderCI, &ps);

	GraphicsPipelineStateCreateInfo PSOCreateInfo;
	PSOCreateInfo.PSODesc.Name = "UI PSO";
	auto& GP = PSOCreateInfo.GraphicsPipeline;
	GP.NumRenderTargets  = 1;
	GP.RTVFormats[0]     = bbFmt;
	// UI draws with NO depth bound (SetRenderTargets passes a null DSV) and never tests/writes depth,
	// so the PSO must declare DSVFormat = UNKNOWN — otherwise Diligent warns every frame about a
	// depth-format mismatch (bound UNKNOWN vs PSO D32_FLOAT). dsFmt is intentionally unused.
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

	// MUTABLE (not DYNAMIC): the UI binds textures through a per-texture SRB CACHE, so
	// commits consume NO dynamic GPU descriptors (a dynamic var allocated fresh ones on
	// EVERY CommitShaderResources - dozens per frame, per window; the heap bled out).
	ShaderResourceVariableDesc vars[] = {{SHADER_TYPE_PIXEL, "Texture", SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE}};
	PSOCreateInfo.PSODesc.ResourceLayout.Variables    = vars;
	PSOCreateInfo.PSODesc.ResourceLayout.NumVariables = 1;
	SamplerDesc sam;
	sam.AddressU = sam.AddressV = sam.AddressW = TEXTURE_ADDRESS_CLAMP;
	ImmutableSamplerDesc samplers[] = {{SHADER_TYPE_PIXEL, "Texture", sam}};
	PSOCreateInfo.PSODesc.ResourceLayout.ImmutableSamplers    = samplers;
	PSOCreateInfo.PSODesc.ResourceLayout.NumImmutableSamplers = 1;

	device->CreateGraphicsPipelineState(PSOCreateInfo, &uiPSO);

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

	// PS lighting buffer (camera pos + ambient + light array), bound as a static var on every world PSO.
	BufferDesc fcbd;
	fcbd.Name = "World FrameCB"; fcbd.Size = sizeof(FrameCBData); fcbd.Usage = USAGE_DYNAMIC;
	fcbd.BindFlags = BIND_UNIFORM_BUFFER; fcbd.CPUAccessFlags = CPU_ACCESS_WRITE;
	device->CreateBuffer(fcbd, nullptr, &worldFrameCB);

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

	// Probe sampler (linear, clamp) — attached to each probe/fallback cube SRV (combined-texture-samplers mode
	// reads it from the view), so g_Probe needs NO immutable sampler in the world PSO (custom shaders may omit it).
	{
		SamplerDesc ps; ps.MinFilter = FILTER_TYPE_LINEAR; ps.MagFilter = FILTER_TYPE_LINEAR; ps.MipFilter = FILTER_TYPE_LINEAR;
		ps.AddressU = TEXTURE_ADDRESS_CLAMP; ps.AddressV = TEXTURE_ADDRESS_CLAMP; ps.AddressW = TEXTURE_ADDRESS_CLAMP;
		device->CreateSampler(ps, &probeSampler);
	}
	// 1x1 black cube — bound to g_Probe when no reflection probe is active (so the var is always valid).
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

	// Clamp the requested MSAA to what the device supports for BOTH color (RGBA8) and depth (D32),
	// stepping 8->4->2->1. SampleCounts is a bitmask where the bit value equals the sample count.
	{
		Uint32 colorSC = (Uint32)device->GetTextureFormatInfoExt(TEX_FORMAT_RGBA8_UNORM).SampleCounts;
		Uint32 depthSC = (Uint32)device->GetTextureFormatInfoExt(TEX_FORMAT_D32_FLOAT).SampleCounts;
		Uint32 want = samples;
		while (want > 1 && !((colorSC & want) && (depthSC & want))) want >>= 1;
		samples = (Uint8)(want < 1 ? 1 : want);
		std::cout << "[NukeDiligent]\tMSAA samples = " << (int)samples << std::endl;
	}

	// Built-in "world" pipeline from the engine shaders.
	defaultWorldHandle = MakeWorldPSO(shaderSource("world.vs"), shaderSource("world.ps"), "World");

	BuildOutlinePipelines();   // selection outline (stencil mark + scaled draw)
	CreateShadowResources();   // directional shadow map + depth PSO
	CreateSkyResources();      // procedural sky pipeline
	CreateDebugResources();    // debug/gizmo line pipeline
	CreateSpriteResources();   // 2D sprite quad pipeline (SceneFmt + MSAA -> rebuild with them)
	CreateDecalResources();    // screen-space decal pipeline (SceneFmt + MSAA)
	CreatePostResources();     // final tonemap / post-process pass
}

// Selection-outline pipelines: (1) MASK = render the mesh flat into an RGBA8 mask (alpha=1);
// (2) EDGE = fullscreen edge-detect over the mask, drawing a constant-pixel-thickness border.
void NukeDiligent::Impl::BuildOutlinePipelines()
{
	// rebuild-safe (MSAA change re-calls this) — release prior objects so Create doesn't assert.
	outlineMaskPSO.Release(); outlineMaskSRB.Release();
	outlineEdgePSO.Release(); outlineEdgeSRB.Release(); outlineEdgeCB.Release();
	ShaderCreateInfo sci; sci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;

	// --- mask pipeline (mesh -> mask RT) ---
	std::string mvs = shaderSource("outline.vs"), mps = shaderSource("outline.ps");
	if (!mvs.empty() && !mps.empty())
	{
		RefCntAutoPtr<IShader> vs, ps;
		sci.Desc = {"Outline Mask VS", SHADER_TYPE_VERTEX, true}; sci.Source = mvs.c_str(); device->CreateShader(sci, &vs);
		sci.Desc = {"Outline Mask PS", SHADER_TYPE_PIXEL, true};  sci.Source = mps.c_str(); device->CreateShader(sci, &ps);
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
			device->CreateGraphicsPipelineState(ci, &outlineMaskPSO);
			if (outlineMaskPSO)
			{
				if (auto* v = outlineMaskPSO->GetStaticVariableByName(SHADER_TYPE_VERTEX, "CB")) v->Set(worldCB);
				outlineMaskPSO->CreateShaderResourceBinding(&outlineMaskSRB, true);
			}
		}
	}

	// --- edge pipeline (fullscreen mask -> camera RT) ---
	std::string evs = shaderSource("outline_edge.vs"), eps = shaderSource("outline_edge.ps");
	if (!evs.empty() && !eps.empty())
	{
		// constants buffer (texel size + thickness)
		BufferDesc cbd; cbd.Name = "Outline EdgeCB"; cbd.Size = sizeof(float) * 4;
		cbd.Usage = USAGE_DYNAMIC; cbd.BindFlags = BIND_UNIFORM_BUFFER; cbd.CPUAccessFlags = CPU_ACCESS_WRITE;
		device->CreateBuffer(cbd, nullptr, &outlineEdgeCB);

		RefCntAutoPtr<IShader> vs, ps;
		sci.Desc = {"Outline Edge VS", SHADER_TYPE_VERTEX, true}; sci.Source = evs.c_str(); device->CreateShader(sci, &vs);
		sci.Desc = {"Outline Edge PS", SHADER_TYPE_PIXEL, true};  sci.Source = eps.c_str(); device->CreateShader(sci, &ps);
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
			device->CreateGraphicsPipelineState(ci, &outlineEdgePSO);
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

// Build (or REBUILD) the 3 blend-variant PSOs + SRB into `wp` using the current `samples`. Used by both
// MakeWorldPSO (first build) and setMSAA (rebuild in place so material->shader handles stay valid).
bool NukeDiligent::Impl::BuildWorldPipe(WorldPipe& wp, const std::string& vsSrc, const std::string& psSrc, const char* dbg)
{
	if (vsSrc.empty() || psSrc.empty()) return false;
	ShaderCreateInfo sci;
	sci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
	// Ray tracing: compile the world shaders via DXC at SM6.5 with RT_ENABLED so world.ps can use inline RayQuery
	// (RT shadows). D3D11 / unsupported GPUs keep the default FXC SM5 path (shadow maps).
	ShaderMacro rtMacro[] = {{"RT_ENABLED", "1"}};
	if (rtSupported)
	{
		sci.ShaderCompiler = SHADER_COMPILER_DXC;
		sci.HLSLVersion    = ShaderVersion{6, 5};
		sci.Macros         = ShaderMacroArray{rtMacro, 1};
	}
	RefCntAutoPtr<IShader> vs, ps;
	sci.Desc = {dbg, SHADER_TYPE_VERTEX, true}; sci.Source = vsSrc.c_str(); device->CreateShader(sci, &vs);
	sci.Desc = {dbg, SHADER_TYPE_PIXEL, true};  sci.Source = psSrc.c_str(); device->CreateShader(sci, &ps);
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
	// Opaque base state: blend off, depth write on. The transparent/additive variants below flip blend +
	// depth write (the engine sorts those back-to-front so straight-alpha blending composites correctly).
	LayoutElement layout[] = {
		{0, 0, 3, VT_FLOAT32}, // position
		{1, 1, 3, VT_FLOAT32}, // normal
		{2, 2, 2, VT_FLOAT32}, // uv
	};
	gp.InputLayout.NumElements    = 3;
	gp.InputLayout.LayoutElements = layout;

	ShaderResourceVariableDesc vars[] = {
		{SHADER_TYPE_PIXEL, "g_Tex",        SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
		{SHADER_TYPE_PIXEL, "g_Normal",     SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
		{SHADER_TYPE_PIXEL, "g_MetalRough", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
		{SHADER_TYPE_PIXEL, "g_Occlusion",  SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
		{SHADER_TYPE_PIXEL, "g_Emissive",   SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
		{SHADER_TYPE_PIXEL, "g_Spec",       SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
		{SHADER_TYPE_PIXEL, "g_Shadow",     SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
		{SHADER_TYPE_PIXEL, "g_ShadowCube", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
		{SHADER_TYPE_PIXEL, "g_Probe",      SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},   // reflection probe cubemap
		{SHADER_TYPE_PIXEL, "g_TLAS",       SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},   // RT scene (only present when rtSupported)
	};
	ci.PSODesc.ResourceLayout.Variables    = vars;
	// g_TLAS (last entry) only exists when rtSupported -> drop it from the list otherwise.
	const Uint32 kNumVars = (Uint32)(sizeof(vars) / sizeof(vars[0]));
	ci.PSODesc.ResourceLayout.NumVariables = rtSupported ? kNumVars : kNumVars - 1;
	SamplerDesc samp; samp.MinFilter = FILTER_TYPE_LINEAR; samp.MagFilter = FILTER_TYPE_LINEAR; samp.MipFilter = FILTER_TYPE_LINEAR;
	samp.AddressU = TEXTURE_ADDRESS_WRAP; samp.AddressV = TEXTURE_ADDRESS_WRAP; samp.AddressW = TEXTURE_ADDRESS_WRAP;
	// One immutable sampler per material-map texture. Combined-sampler mode is strict on D3D12: a texture X is
	// sampled via X_sampler, so each needs its own immutable sampler (keyed by the texture name). Register only the
	// maps THIS shader actually declares (query the compiled PS) — a custom shader without them must not carry
	// unassigned immutable samplers (Diligent warns). Shadow + probe samplers are attached to their SRVs elsewhere.
	static const char* const kMapTex[] = { "g_Tex", "g_Normal", "g_MetalRough", "g_Occlusion", "g_Emissive", "g_Spec" };
	ImmutableSamplerDesc immSamp[6]; Uint32 nImm = 0;
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
	};

	// Release any previous objects first (rebuild path) — Diligent asserts on Create over a non-null ref.
	wp.pso.Release(); wp.psoBlend.Release(); wp.psoAdd.Release(); wp.srb.Release();

	// 1) Opaque — blend off, depth write on (base ci as configured above).
	device->CreateGraphicsPipelineState(ci, &wp.pso);
	if (!wp.pso) { cout << "[NukeDiligent]\tPSO build failed for shader '" << dbg << "'" << endl; return false; }
	setStatics(wp.pso);

	// 2) Transparent — straight-alpha blend, depth test on but NO depth write (sorted back-to-front by engine).
	{
		auto& rt = ci.GraphicsPipeline.BlendDesc.RenderTargets[0];
		rt.BlendEnable = True;
		rt.SrcBlend = BLEND_FACTOR_SRC_ALPHA; rt.DestBlend = BLEND_FACTOR_INV_SRC_ALPHA; rt.BlendOp = BLEND_OPERATION_ADD;
		rt.SrcBlendAlpha = BLEND_FACTOR_ONE;  rt.DestBlendAlpha = BLEND_FACTOR_INV_SRC_ALPHA; rt.BlendOpAlpha = BLEND_OPERATION_ADD;
		ci.GraphicsPipeline.DepthStencilDesc.DepthWriteEnable = False;
		ci.PSODesc.Name = "World (blend)";
		device->CreateGraphicsPipelineState(ci, &wp.psoBlend);
		if (wp.psoBlend) setStatics(wp.psoBlend);
	}
	// 3) Additive — src*a + dst, depth write off.
	{
		auto& rt = ci.GraphicsPipeline.BlendDesc.RenderTargets[0];
		rt.BlendEnable = True;
		rt.SrcBlend = BLEND_FACTOR_SRC_ALPHA; rt.DestBlend = BLEND_FACTOR_ONE; rt.BlendOp = BLEND_OPERATION_ADD;
		rt.SrcBlendAlpha = BLEND_FACTOR_ONE;  rt.DestBlendAlpha = BLEND_FACTOR_ONE; rt.BlendOpAlpha = BLEND_OPERATION_ADD;
		ci.PSODesc.Name = "World (add)";
		device->CreateGraphicsPipelineState(ci, &wp.psoAdd);
		if (wp.psoAdd) setStatics(wp.psoAdd);
	}

	wp.pso->CreateShaderResourceBinding(&wp.srb, true);
	wp.texVar  = wp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Tex");
	wp.normVar = wp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Normal");
	wp.mrVar   = wp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_MetalRough");
	wp.aoVar   = wp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Occlusion");
	wp.emVar   = wp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Emissive");
	wp.specVar = wp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Spec");
	wp.shadowVar = wp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Shadow");
	wp.cubeVar   = wp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_ShadowCube");
	wp.probeVar  = wp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Probe");
	wp.tlasVar   = wp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_TLAS");
	return true;
}

uint64_t NukeDiligent::Impl::MakeWorldPSO(const std::string& vsSrc, const std::string& psSrc, const char* dbg)
{
	WorldPipe wp;
	wp.vsSrc = vsSrc; wp.psSrc = psSrc; wp.dbg = dbg;   // kept so setMSAA can rebuild this pipeline
	if (!BuildWorldPipe(wp, vsSrc, psSrc, dbg)) return 0;
	uint64_t h = nextShaderHandle++;
	worldPipes[h] = std::move(wp);
	return h;
}

// After `samples` changes, rebuild everything whose PSO sample count or texture sample count depends on it:
// all world pipelines (in place, so material->shader handles stay valid), the sky + outline-edge PSOs, the
// MS backbuffer, and every render target. Shadow maps / outline mask / UI are single-sample and untouched.
void NukeDiligent::Impl::RebuildForMSAA()
{
	for (auto& kv : worldPipes)
		BuildWorldPipe(kv.second, kv.second.vsSrc, kv.second.psSrc, kv.second.dbg.c_str());
	CreateSkyResources();
	CreateDebugResources();
	CreateSpriteResources();   // sprite PSO sample count / SceneFmt depends on samples+HDR
	CreateDecalResources();    // decal PSO sample count / SceneFmt depends on samples+HDR
	BuildOutlinePipelines();
	backbufferMS = RT{};   // recreated on next target-0 camera
	for (auto& kv : rts)
		if (kv.second.w > 0 && kv.second.h > 0) kv.second = MakeRT(kv.second.w, kv.second.h);
}

uint64_t NukeDiligent::createShaderPipeline(const char* name, const char* vs, const char* ps)
{
	if (!vs || !ps) return 0;
	uint64_t h = m_impl->MakeWorldPSO(vs, ps, "Shader");   // world-type PSO (layout/CBs) from custom VS+PS
	// A shader that ships "<name>.surf.hlsl" gets an auto-generated RT closest-hit (per-shader reflections).
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
