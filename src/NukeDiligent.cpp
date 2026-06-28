// IMPORTANT include order: Windows-pulling headers (GLFW native + Diligent D3D)
// MUST come before the engine headers. Several engine headers do a global
// `using namespace std;`, which brings std::byte into scope; if the Windows SDK
// headers (objidl.h/oaidl.h, pulled by Diligent's D3D backend) are processed
// after that, `byte` becomes ambiguous (std::byte vs Windows ::byte). Processing
// the Windows headers first avoids the clash.

#include <cmath>   // acosf (spot shadow FOV)

// GLFW: window creation + native Win32 handle
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <shellapi.h>   // ExtractIconEx (set the window icon from the .exe)
#include <dwmapi.h>     // DwmSetWindowAttribute (dark title bar)
#ifndef DWMWA_USE_IMMERSIVE_DARK_MODE
#define DWMWA_USE_IMMERSIVE_DARK_MODE 20   // Win10 2004+ (build 19041+)
#endif

// Diligent Engine
#include "EngineFactoryD3D11.h"
#include "RenderDevice.h"
#include "DeviceContext.h"
#include "SwapChain.h"
#include "PipelineState.h"
#include "ShaderResourceBinding.h"
#include "Buffer.h"
#include "Texture.h"
#include "Shader.h"
#include "RefCntAutoPtr.hpp"
#include "MapHelper.hpp"
#include "BasicMath.hpp"
#include "GraphicsAccessories.hpp"
// --- Windows-only HDR10 display output (DXGI / D3D11). Other platforms: no D3D11, HDR output is a no-op. ---
#ifdef _WIN32
#include <d3d11.h>           // ID3D11View etc. (needed BEFORE the Diligent D3D11 interface headers below)
#include <dxgi1_6.h>         // IDXGISwapChain3/4 + IDXGIOutput6 (HDR10 colour space + display detection)
#include "SwapChainD3D11.h"  // Diligent ISwapChainD3D11::GetDXGISwapChain (HDR10 swap-chain access)
#endif

// Engine headers last (they do `using namespace std;` internally).
#include "NukeDiligent.h"
#include <interface/RenderModule.h>

#include <cstring>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include <string>

using namespace Diligent;
// NOTE: deliberately NOT `using namespace std` — it pulls std::byte, which clashes
// with the Windows SDK's ::byte (rpcndr.h) and makes `byte` ambiguous.
using std::cout;
using std::endl;

struct NukeDiligent::Impl
{
	RefCntAutoPtr<IRenderDevice>  device;
	RefCntAutoPtr<IDeviceContext> context;
	RefCntAutoPtr<ISwapChain>     swapChain;

	std::vector<boost::function<void(void)>> onGUI;
	std::vector<boost::function<void(void)>> onRender;
	std::vector<boost::function<void()>>     onClose;

	// --- generic 2D (UI) draw-list renderer ---
	RefCntAutoPtr<IPipelineState>         uiPSO;
	RefCntAutoPtr<IShaderResourceBinding> uiSRB;
	IShaderResourceVariable*              uiTexVar = nullptr;
	RefCntAutoPtr<IBuffer>                uiVB, uiIB, uiCB;
	int uiVBSize = 0;
	int uiIBSize = 0;
	bool baseVertexSupported = false;
	// Keep created textures alive; handle == ITextureView*.
	std::unordered_map<uint64_t, RefCntAutoPtr<ITexture>> textures;

	// --- render targets (cameras draw into these; the UI can sample them) ---
	struct RT
	{
		RefCntAutoPtr<ITexture> color, depth;     // color = HDR (RGBA16F) single-sample: geometry target (no MSAA) / resolve dest
		RefCntAutoPtr<ITexture> colorMS, depthMS; // multisampled HDR render targets (when samples > 1)
		RefCntAutoPtr<ITexture> post;             // LDR (RGBA8) post-process output — what the UI samples
		ITextureView* rtv = nullptr;              // geometry RTV (MS when samples>1, else color)
		ITextureView* dsv = nullptr;              // geometry DSV
		ITextureView* hdrSRV = nullptr;           // color's SRV (post-pass input)
		ITextureView* postRTV = nullptr;          // post's RTV (post-pass output)
		ITextureView* srv = nullptr;              // post's SRV (final LDR result shown by the UI / sampled as a texture)
		int w = 0, h = 0;
	};
	std::unordered_map<uint64_t, RT> rts;
	uint64_t rtCounter = 0;

	// --- MSAA --------------------------------------------------------------------------------------
	Uint8 samples = 4;                 // hardware multisample count for all geometry passes (1 = off)
	int   pendingSamples = -1;         // requested sample count; applied at the START of render() (never mid-frame)
	RT    backbufferMS;                // MS color+depth for camera target 0 (Player), resolved to the backbuffer
	// Resolve bookkeeping for the current camera pass (set in beginCamera, consumed in endCamera).
	bool      curMSAA = false;
	ITexture* curResolveSrc = nullptr; // MS HDR color to resolve from
	ITexture* curResolveDst = nullptr; // single-sample HDR destination
	ITextureView* curPostSrc = nullptr; // HDR SRV the post pass reads (after resolve)
	ITextureView* curPostDst = nullptr; // LDR RTV the post pass writes (RT's post / the backbuffer)
	void EnsureBackbufferMS(int w, int h);

	// --- Post-process ------------------------------------------------------------------------------
	RefCntAutoPtr<IPipelineState>         postPSO;       // -> RT targets (RGBA8, SDR sRGB)
	RefCntAutoPtr<IShaderResourceBinding> postSRB;
	RefCntAutoPtr<IPipelineState>         postPSOBB;     // -> the backbuffer (matches the swap-chain format; PQ when HDR10)
	RefCntAutoPtr<IShaderResourceBinding> postSRBBB;
	RefCntAutoPtr<IBuffer>                postCB;
	IShaderResourceVariable*              postHdrVar = nullptr;
	IShaderResourceVariable*              postHdrVarBB = nullptr;
	float                                 exposure = 1.0f;
	bool                                  hdrOutput = false;   // requested HDR10 display output (Player only, before init)
	bool                                  hdr10Active = false; // an HDR10 swap chain is actually live (display is HDR)
	void CreatePostResources();
	void RunPostPass(ITextureView* hdrSRV, ITextureView* dstRTV, int w, int h, bool toBackbuffer);
	void SetupHDROutput();   // after swap-chain creation: set the HDR10 colour space if the display supports it
	static constexpr TEXTURE_FORMAT HDR_FMT = TEX_FORMAT_RGBA16_FLOAT;
	bool hdr = true;                   // HDR pipeline on (scene = RGBA16F, post tonemaps) / off (RGBA8, world.ps tonemaps)
	int  pendingHDR = -1;              // requested hdr (0/1); applied with pendingSamples at the start of render()
	TEXTURE_FORMAT SceneFmt() const { return hdr ? HDR_FMT : TEX_FORMAT_RGBA8_UNORM; }

	// --- 3D world pipelines (one per shader; all share the layout + CBs + white fallback) ---
	struct WorldPipe
	{
		RefCntAutoPtr<IPipelineState>         pso;        // opaque (blend off, depth write on)
		RefCntAutoPtr<IPipelineState>         psoBlend;   // transparent: alpha blend, depth test on / write off
		RefCntAutoPtr<IPipelineState>         psoAdd;     // additive: add blend, depth write off
		RefCntAutoPtr<IShaderResourceBinding> srb;
		IShaderResourceVariable*              texVar  = nullptr;  // PS "g_Tex"        (base color, dynamic)
		IShaderResourceVariable*              normVar = nullptr;  // PS "g_Normal"     (normal map, dynamic)
		IShaderResourceVariable*              mrVar   = nullptr;  // PS "g_MetalRough" (dynamic)
		IShaderResourceVariable*              aoVar   = nullptr;  // PS "g_Occlusion"  (dynamic)
		IShaderResourceVariable*              emVar   = nullptr;  // PS "g_Emissive"   (dynamic)
		IShaderResourceVariable*              shadowVar = nullptr;// PS "g_Shadow"      (dynamic)
		IShaderResourceVariable*              cubeVar   = nullptr;// PS "g_ShadowCube" (dynamic)
		std::string vsSrc, psSrc, dbg;   // kept so the pipeline can be rebuilt (e.g. on an MSAA change)
	};
	std::unordered_map<uint64_t, WorldPipe> worldPipes;   // shader handle -> pipeline
	uint64_t                              defaultWorldHandle = 0;   // builtin "world" pipeline
	uint64_t                              nextShaderHandle   = 1;   // handles handed to the engine
	RefCntAutoPtr<IBuffer>                worldCB;     // VS: WVP + World   (shared)
	RefCntAutoPtr<IBuffer>                worldMatCB;  // PS: color + params + custom shader props (shared)
	static const uint32_t                 kMatCBBytes = 256;   // MatCB capacity (color/params + props)
	RefCntAutoPtr<ITexture>               whiteTex;    // 1x1 fallback when a material has no texture
	RefCntAutoPtr<ITexture>               flatNormTex; // 1x1 (0.5,0.5,1) flat normal fallback
	RefCntAutoPtr<IBuffer>                worldFrameCB;// PS b1: camera pos + ambient + light array (shared)
	// PBR lighting buffer layout (matches FrameCB in world.ps.hlsl). Each float4 = 16 bytes.
	static const int                      kMaxLights = 16;
	struct GPULight { float posType[4]; float dirRange[4]; float colorIntensity[4]; float spot[4]; };
	struct FrameCBData { float camPos[4]; float ambient[4]; float lightCount[4]; GPULight lights[kMaxLights];
	                     float shadowVP[16 * 4]; float shadowParams[4];        // 4 = SHADOW_SLOTS
	                     float skyTop[4]; float skyHorizon[4]; float skyGround[4]; float skyParams[4]; };  // IBL
	float                                 curCamPos[3] = {0, 0, 0};  // set in beginCamera (PBR view dir)
	uint64_t                              curTarget = 0;             // RT id bound by beginCamera (feedback guard)
	std::vector<NukeLight>                lights;      // scene lights (setLights); empty -> default sun

	// --- Shadow maps (directional + spot share a 2D array; one slice per shadow-casting light) -----
	static const int                      SHADOW_RES   = 2048;
	static const int                      SHADOW_SLOTS = 4;
	RefCntAutoPtr<ITexture>               shadowTex;             // Texture2DArray, D32, SHADOW_SLOTS slices
	RefCntAutoPtr<ITextureView>           shadowSliceDSV[SHADOW_SLOTS];   // per-slice depth targets
	ITextureView*                         shadowSRV = nullptr;   // whole-array SRV (sampled in the world pass)
	RefCntAutoPtr<IPipelineState>         shadowPSO;
	RefCntAutoPtr<IShaderResourceBinding> shadowSRB;
	IShaderResourceVariable*              shadowPsTexVar = nullptr;   // shadow PS "g_Tex" (alpha)
	RefCntAutoPtr<IBuffer>                shadowVSCB;    // VS: g_LightWVP (per shadow draw)
	RefCntAutoPtr<IBuffer>                shadowPSCB;    // PS: g_Alpha    (per shadow draw)
	RefCntAutoPtr<ISampler>               shadowCmpSampler;   // PCF comparison sampler (set on shadowSRV)
	// Procedural sky / environment.
	NukeSky                               sky;
	RefCntAutoPtr<IPipelineState>         skyPSO;
	RefCntAutoPtr<IShaderResourceBinding> skySRB;
	RefCntAutoPtr<IBuffer>                skyCB;
	IShaderResourceVariable*              skyStarVar = nullptr;   // sky PS "g_StarTex" (optional star panorama)
	IShaderResourceVariable*              skyMoonVar = nullptr;   // sky PS "g_MoonTex" (optional moon disk)
	void CreateSkyResources();
	void DrawSky();

	int                                   numShadowSlots = 0;          // assigned 2D slots this frame
	int                                   lightSlot[kMaxLights];       // per-light 2D shadow slot (-1 = none)
	float4x4                              slotVP[SHADOW_SLOTS];        // world->light-clip per 2D slot
	float4x4                              curShadowVP;                 // VP of the pass being rendered
	// Point-light shadows: a cube depth array (6 faces per cube), sampled by direction.
	static const int                      MAX_POINT_SHADOWS = 2;
	RefCntAutoPtr<ITexture>               shadowCubeTex;               // TextureCubeArray, D32
	RefCntAutoPtr<ITextureView>           cubeFaceDSV[MAX_POINT_SHADOWS * 6];
	ITextureView*                         shadowCubeSRV = nullptr;
	RefCntAutoPtr<ISampler>               shadowCubeCmpSampler;
	int                                   numCubes = 0;
	float4x4                              cubeFaceVP[MAX_POINT_SHADOWS * 6];
	int                                   lightCube[kMaxLights];       // per-light cube index (-1 = none)
	void CreateShadowResources();
	// Build a world-type PSO (fixed layout/CBs) from VS+PS source; store it under a handle.
	uint64_t MakeWorldPSO(const std::string& vsSrc, const std::string& psSrc, const char* dbg);
	bool     BuildWorldPipe(WorldPipe& wp, const std::string& vsSrc, const std::string& psSrc, const char* dbg);
	void     RebuildForMSAA();   // rebuild all sample-count-dependent pipelines + targets after `samples` changes
	struct MeshGPU { RefCntAutoPtr<IBuffer> pos, nrm, uv; int numVerts = 0; };
	std::unordered_map<Mesh*, MeshGPU>          meshCache;
	MeshGPU* GetMeshGPU(Mesh* mesh);   // get-or-build the GPU vertex buffers (pos/nrm/uv) for a mesh
	std::unordered_map<Texture*, RefCntAutoPtr<ITexture>> texCache;   // engine Texture -> GPU texture
	std::unordered_map<Texture*, std::vector<RefCntAutoPtr<ITexture>>> animTex;   // GIF: one Texture2D per frame
	float4x4 curView, curProj;   // set in beginCamera, used in renderObject

	// Selection outline (editor): post-process. Pass 1 renders the selected mesh into a mask RT;
	// pass 2 is a fullscreen edge-detect that draws a CONSTANT-pixel-thickness border around the mask
	// (independent of distance/size, works for any geometry incl. flat planes).
	RefCntAutoPtr<IPipelineState>         outlineMaskPSO, outlineEdgePSO;
	RefCntAutoPtr<IShaderResourceBinding> outlineMaskSRB, outlineEdgeSRB;
	IShaderResourceVariable*              outlineEdgeMaskVar = nullptr;   // edge PS "g_Mask" (dynamic)
	RefCntAutoPtr<ITexture>               outlineMaskTex;
	ITextureView*                         outlineMaskRTV = nullptr;
	ITextureView*                         outlineMaskSRV = nullptr;
	int                                   outlineMaskW = 0, outlineMaskH = 0;
	RefCntAutoPtr<IBuffer>                outlineEdgeCB;      // texel size + thickness
	ITextureView*                         curRTV = nullptr;   // current camera color target (outline rebind)
	int                                   curRTW = 0, curRTH = 0;
	ITextureView*                         uiRTV = nullptr;    // explicit 2D target (bindRenderTarget); null = backbuffer
	Uint32                                uiTW = 0, uiTH = 0; // its size (0 = use swapchain)
	void BuildOutlinePipelines();
	void EnsureOutlineMask(int w, int h);

	// Shader sources pushed by the engine (the renderer does NO file IO). name -> HLSL.
	std::unordered_map<std::string, std::string> shaderSrc;
	std::string shaderSource(const char* name)
	{
		auto it = shaderSrc.find(name);
		if (it == shaderSrc.end() || it->second.empty())
			{ cout << "[NukeDiligent]\tmissing shader source '" << name << "'" << endl; return std::string(); }
		return it->second;
	}

	ITextureView* GetTexSRV(Texture* t);   // get-or-create a GPU texture from an engine Texture

	void CreateUIPipeline(TEXTURE_FORMAT bbFmt, TEXTURE_FORMAT dsFmt);
	void CreateWorldPipeline();
	RT   MakeRT(int w, int h);
};

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

	ShaderResourceVariableDesc vars[] = {{SHADER_TYPE_PIXEL, "Texture", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC}};
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

	uiPSO->CreateShaderResourceBinding(&uiSRB, true);
	uiTexVar = uiSRB->GetVariableByName(SHADER_TYPE_PIXEL, "Texture");
}

void NukeDiligent::Impl::CreateWorldPipeline()
{
	// Shared constant buffers (bound as static vars on EVERY world PSO).
	BufferDesc cbd;
	cbd.Name = "World CB"; cbd.Size = sizeof(float4x4) * 2; cbd.Usage = USAGE_DYNAMIC;
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
		{SHADER_TYPE_PIXEL, "g_Shadow",     SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
		{SHADER_TYPE_PIXEL, "g_ShadowCube", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
	};
	ci.PSODesc.ResourceLayout.Variables    = vars;
	ci.PSODesc.ResourceLayout.NumVariables = 7;
	SamplerDesc samp; samp.MinFilter = FILTER_TYPE_LINEAR; samp.MagFilter = FILTER_TYPE_LINEAR; samp.MipFilter = FILTER_TYPE_LINEAR;
	samp.AddressU = TEXTURE_ADDRESS_WRAP; samp.AddressV = TEXTURE_ADDRESS_WRAP; samp.AddressW = TEXTURE_ADDRESS_WRAP;
	ImmutableSamplerDesc immSamp[] = {{SHADER_TYPE_PIXEL, "g_Tex", samp}};   // shared sampler for material maps;
	ci.PSODesc.ResourceLayout.ImmutableSamplers    = immSamp;                // the shadow comparison sampler is
	ci.PSODesc.ResourceLayout.NumImmutableSamplers = 1;                      // attached to the shadow SRV instead
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
	wp.shadowVar = wp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Shadow");
	wp.cubeVar   = wp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_ShadowCube");
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
	BuildOutlinePipelines();
	backbufferMS = RT{};   // recreated on next target-0 camera
	for (auto& kv : rts)
		if (kv.second.w > 0 && kv.second.h > 0) kv.second = MakeRT(kv.second.w, kv.second.h);
}

void NukeDiligent::Impl::CreateShadowResources()
{
	for (int s = 0; s < SHADOW_SLOTS; ++s) lightSlot[s] = -1;
	TextureDesc td; td.Name = "Shadow Maps"; td.Type = RESOURCE_DIM_TEX_2D_ARRAY;
	td.Width = SHADOW_RES; td.Height = SHADOW_RES; td.ArraySize = SHADOW_SLOTS; td.MipLevels = 1;
	td.Format = TEX_FORMAT_D32_FLOAT;
	td.BindFlags = BIND_DEPTH_STENCIL | BIND_SHADER_RESOURCE;
	device->CreateTexture(td, nullptr, &shadowTex);
	if (shadowTex)
	{
		shadowSRV = shadowTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);   // whole array
		// PCF comparison sampler, attached to the SRV (combined-texture-sampler mode reads it from the view).
		SamplerDesc cmp; cmp.MinFilter = FILTER_TYPE_COMPARISON_LINEAR; cmp.MagFilter = FILTER_TYPE_COMPARISON_LINEAR;
		cmp.MipFilter = FILTER_TYPE_COMPARISON_POINT;
		cmp.AddressU = TEXTURE_ADDRESS_CLAMP; cmp.AddressV = TEXTURE_ADDRESS_CLAMP; cmp.AddressW = TEXTURE_ADDRESS_CLAMP;
		cmp.ComparisonFunc = COMPARISON_FUNC_LESS_EQUAL;
		device->CreateSampler(cmp, &shadowCmpSampler);
		if (shadowSRV && shadowCmpSampler) shadowSRV->SetSampler(shadowCmpSampler);
		// one depth-stencil view per array slice (each shadow pass renders into its slice)
		for (int s = 0; s < SHADOW_SLOTS; ++s)
		{
			TextureViewDesc vd; vd.Name = "Shadow slice DSV"; vd.ViewType = TEXTURE_VIEW_DEPTH_STENCIL;
			vd.TextureDim = RESOURCE_DIM_TEX_2D_ARRAY; vd.FirstArraySlice = (Uint32)s; vd.NumArraySlices = 1;
			shadowTex->CreateView(vd, &shadowSliceDSV[s]);
		}
	}

	// Point-light shadows: a cube depth array (6 faces per cube), sampled by direction in the world pass.
	for (int i = 0; i < kMaxLights; ++i) lightCube[i] = -1;
	TextureDesc ctd; ctd.Name = "Shadow Cubes"; ctd.Type = RESOURCE_DIM_TEX_CUBE_ARRAY;
	ctd.Width = SHADOW_RES / 2; ctd.Height = SHADOW_RES / 2; ctd.ArraySize = MAX_POINT_SHADOWS * 6; ctd.MipLevels = 1;
	ctd.Format = TEX_FORMAT_D32_FLOAT; ctd.BindFlags = BIND_DEPTH_STENCIL | BIND_SHADER_RESOURCE;
	device->CreateTexture(ctd, nullptr, &shadowCubeTex);
	if (shadowCubeTex)
	{
		shadowCubeSRV = shadowCubeTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
		SamplerDesc cmp; cmp.MinFilter = FILTER_TYPE_COMPARISON_LINEAR; cmp.MagFilter = FILTER_TYPE_COMPARISON_LINEAR;
		cmp.MipFilter = FILTER_TYPE_COMPARISON_POINT;
		cmp.AddressU = TEXTURE_ADDRESS_CLAMP; cmp.AddressV = TEXTURE_ADDRESS_CLAMP; cmp.AddressW = TEXTURE_ADDRESS_CLAMP;
		cmp.ComparisonFunc = COMPARISON_FUNC_LESS_EQUAL;
		device->CreateSampler(cmp, &shadowCubeCmpSampler);
		if (shadowCubeSRV && shadowCubeCmpSampler) shadowCubeSRV->SetSampler(shadowCubeCmpSampler);
		for (int f = 0; f < MAX_POINT_SHADOWS * 6; ++f)   // a DSV per cube face (a single array slice)
		{
			TextureViewDesc vd; vd.Name = "Shadow cube face DSV"; vd.ViewType = TEXTURE_VIEW_DEPTH_STENCIL;
			vd.TextureDim = RESOURCE_DIM_TEX_2D_ARRAY; vd.FirstArraySlice = (Uint32)f; vd.NumArraySlices = 1;
			shadowCubeTex->CreateView(vd, &cubeFaceDSV[f]);
		}
	}

	BufferDesc cbd; cbd.Usage = USAGE_DYNAMIC; cbd.BindFlags = BIND_UNIFORM_BUFFER; cbd.CPUAccessFlags = CPU_ACCESS_WRITE;
	cbd.Name = "ShadowVSCB"; cbd.Size = sizeof(float4x4);    device->CreateBuffer(cbd, nullptr, &shadowVSCB);
	cbd.Name = "ShadowPSCB"; cbd.Size = sizeof(float) * 4;   device->CreateBuffer(cbd, nullptr, &shadowPSCB);

	std::string vs = shaderSource("shadow.vs"), ps = shaderSource("shadow.ps");
	if (vs.empty() || ps.empty()) { cout << "[NukeDiligent]\tshadow shaders missing" << endl; return; }
	ShaderCreateInfo sci; sci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
	RefCntAutoPtr<IShader> vsh, psh;
	sci.Desc = {"Shadow VS", SHADER_TYPE_VERTEX, true}; sci.Source = vs.c_str(); device->CreateShader(sci, &vsh);
	sci.Desc = {"Shadow PS", SHADER_TYPE_PIXEL, true};  sci.Source = ps.c_str(); device->CreateShader(sci, &psh);
	if (!vsh || !psh) return;

	GraphicsPipelineStateCreateInfo ci; ci.PSODesc.Name = "Shadow PSO";
	auto& gp = ci.GraphicsPipeline;
	gp.NumRenderTargets = 0;
	gp.RTVFormats[0]    = TEX_FORMAT_UNKNOWN;
	gp.DSVFormat        = TEX_FORMAT_D32_FLOAT;
	gp.PrimitiveTopology = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	gp.RasterizerDesc.CullMode        = CULL_MODE_NONE;
	gp.RasterizerDesc.DepthClipEnable = True;
	gp.DepthStencilDesc.DepthEnable      = True;
	gp.DepthStencilDesc.DepthWriteEnable = True;
	gp.DepthStencilDesc.DepthFunc        = COMPARISON_FUNC_LESS;
	LayoutElement layout[] = { {0, 0, 3, VT_FLOAT32}, {1, 1, 3, VT_FLOAT32}, {2, 2, 2, VT_FLOAT32} };
	gp.InputLayout.NumElements = 3; gp.InputLayout.LayoutElements = layout;
	ShaderResourceVariableDesc vars[] = {
		{SHADER_TYPE_VERTEX, "ShadowVSCB", SHADER_RESOURCE_VARIABLE_TYPE_STATIC},
		{SHADER_TYPE_PIXEL,  "ShadowPSCB", SHADER_RESOURCE_VARIABLE_TYPE_STATIC},
		{SHADER_TYPE_PIXEL,  "g_Tex",      SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
	};
	ci.PSODesc.ResourceLayout.Variables = vars; ci.PSODesc.ResourceLayout.NumVariables = 3;
	SamplerDesc samp; samp.MinFilter = FILTER_TYPE_LINEAR; samp.MagFilter = FILTER_TYPE_LINEAR; samp.MipFilter = FILTER_TYPE_LINEAR;
	ImmutableSamplerDesc imm[] = {{SHADER_TYPE_PIXEL, "g_Tex", samp}};
	ci.PSODesc.ResourceLayout.ImmutableSamplers = imm; ci.PSODesc.ResourceLayout.NumImmutableSamplers = 1;
	ci.pVS = vsh; ci.pPS = psh;
	device->CreateGraphicsPipelineState(ci, &shadowPSO);
	if (shadowPSO)
	{
		// Bind the per-draw cbuffers. Try static (on the PSO) and mutable (on the SRB) — whichever the
		// reflected variable actually is.
		if (auto* v = shadowPSO->GetStaticVariableByName(SHADER_TYPE_VERTEX, "ShadowVSCB")) v->Set(shadowVSCB);
		if (auto* v = shadowPSO->GetStaticVariableByName(SHADER_TYPE_PIXEL,  "ShadowPSCB")) v->Set(shadowPSCB);
		shadowPSO->CreateShaderResourceBinding(&shadowSRB, true);
		if (auto* v = shadowSRB->GetVariableByName(SHADER_TYPE_VERTEX, "ShadowVSCB")) v->Set(shadowVSCB);
		if (auto* v = shadowSRB->GetVariableByName(SHADER_TYPE_PIXEL,  "ShadowPSCB")) v->Set(shadowPSCB);
		shadowPsTexVar = shadowSRB->GetVariableByName(SHADER_TYPE_PIXEL, "g_Tex");
	}
	cout << "[NukeDiligent]\tshadow map " << SHADOW_RES << "x" << SHADOW_RES << (shadowPSO ? " ready" : " FAILED") << endl;
}

void NukeDiligent::Impl::CreateSkyResources()
{
	skyPSO.Release(); skySRB.Release(); skyCB.Release();   // rebuild-safe (MSAA change re-calls this)
	std::string vs = shaderSource("sky.vs"), ps = shaderSource("sky.ps");
	if (vs.empty() || ps.empty()) { cout << "[NukeDiligent]\tsky shaders missing" << endl; return; }
	ShaderCreateInfo sci; sci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
	RefCntAutoPtr<IShader> v, p;
	sci.Desc = {"Sky VS", SHADER_TYPE_VERTEX, true}; sci.Source = vs.c_str(); device->CreateShader(sci, &v);
	sci.Desc = {"Sky PS", SHADER_TYPE_PIXEL, true};  sci.Source = ps.c_str(); device->CreateShader(sci, &p);
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
	device->CreateGraphicsPipelineState(ci, &skyPSO);
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

ITextureView* NukeDiligent::Impl::GetTexSRV(Texture* t)
{
	if (!t) return nullptr;
	if (t->renderTexture)   // sample a RenderTexture = the camera's render-target color view
	{
		// Feedback guard: never sample the RT we're currently rendering INTO (an object that displays the
		// RT, caught in that camera's own view, would bind it as SRV while it's the RTV -> Diligent drops
		// the render target and the whole pass renders nothing but the clear color).
		if (t->rtId != 0 && t->rtId == curTarget) return nullptr;
		auto rit = rts.find(t->rtId);
		return rit != rts.end() ? rit->second.srv : nullptr;
	}
	if (t->pixels.empty() || t->width <= 0 || t->height <= 0) return nullptr;

	if (t->frameCount > 1)   // animated (GIF): a separate Texture2D per frame; return the current frame's SRV
	{
		auto av = animTex.find(t);
		if (av == animTex.end())
		{
			const int w = t->width, h = t->height, n = t->frameCount;
			const bool bc = (t->format == Texture::FMT_BC1 || t->format == Texture::FMT_BC3);
			const int  bb = (t->format == Texture::FMT_BC1) ? 8 : 16;
			const TEXTURE_FORMAT fmt = bc ? (t->format == Texture::FMT_BC1 ? TEX_FORMAT_BC1_UNORM : TEX_FORMAT_BC3_UNORM) : TEX_FORMAT_RGBA8_UNORM;
			const size_t frameBytes = bc ? (size_t)((w + 3) / 4) * ((h + 3) / 4) * bb : (size_t)w * h * 4;
			const Uint64 stride     = bc ? (Uint64)((w + 3) / 4) * bb : (Uint64)w * 4;
			if (t->pixels.size() < frameBytes * n) return nullptr;
			std::vector<RefCntAutoPtr<ITexture>> frames(n);
			for (int k = 0; k < n; ++k)
			{
				TextureDesc td; td.Type = RESOURCE_DIM_TEX_2D; td.Width = w; td.Height = h; td.MipLevels = 1;
				td.Format = fmt; td.BindFlags = BIND_SHADER_RESOURCE; td.Usage = USAGE_IMMUTABLE;
				TextureSubResData sub; sub.pData = t->pixels.data() + (size_t)k * frameBytes; sub.Stride = stride;
				TextureData data; data.pSubResources = &sub; data.NumSubresources = 1;
				device->CreateTexture(td, &data, &frames[k]);
			}
			av = animTex.emplace(t, std::move(frames)).first;
		}
		auto& fs = av->second;
		if (fs.empty()) return nullptr;
		int f = t->curFrame; if (f < 0) f = 0; if (f >= (int)fs.size()) f %= (int)fs.size();
		return fs[f] ? fs[f]->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE) : nullptr;
	}

	auto it = texCache.find(t);
	if (it == texCache.end())
	{
		RefCntAutoPtr<ITexture> tex;
		if (t->format == Texture::FMT_BC1 || t->format == Texture::FMT_BC3)
		{
			// Pre-compressed BC with a stored mip chain — upload every level (no GenerateMips for BC).
			const int  blockBytes = (t->format == Texture::FMT_BC1) ? 8 : 16;
			const int  mips = t->mipCount < 1 ? 1 : t->mipCount;
			TextureDesc td; td.Type = RESOURCE_DIM_TEX_2D; td.Width = t->width; td.Height = t->height;
			td.MipLevels = mips; td.BindFlags = BIND_SHADER_RESOURCE; td.Usage = USAGE_IMMUTABLE;
			td.Format = (t->format == Texture::FMT_BC1) ? TEX_FORMAT_BC1_UNORM : TEX_FORMAT_BC3_UNORM;
			std::vector<TextureSubResData> subs(mips);
			size_t off = 0; int mw = t->width, mh = t->height;
			for (int m = 0; m < mips; ++m)
			{
				int bx = (mw + 3) / 4, by = (mh + 3) / 4;
				subs[m].pData  = t->pixels.data() + off;
				subs[m].Stride = (Uint64)bx * blockBytes;
				off += (size_t)bx * by * blockBytes;
				mw = mw > 1 ? mw / 2 : 1; mh = mh > 1 ? mh / 2 : 1;
			}
			TextureData data; data.pSubResources = subs.data(); data.NumSubresources = (Uint32)mips;
			device->CreateTexture(td, &data, &tex);
		}
		else   // RGBA8 (non-BC fallback, e.g. odd sizes / GIF frames) — single level, no GPU mip-gen
		{
			if (t->pixels.size() < (size_t)t->width * t->height * 4) return nullptr;   // guard malformed data
			TextureDesc td; td.Type = RESOURCE_DIM_TEX_2D; td.Width = t->width; td.Height = t->height;
			td.MipLevels = 1; td.Format = TEX_FORMAT_RGBA8_UNORM;
			td.BindFlags = BIND_SHADER_RESOURCE; td.Usage = USAGE_IMMUTABLE;
			TextureSubResData sub; sub.pData = t->pixels.data(); sub.Stride = (Uint64)t->width * 4;
			TextureData data; data.pSubResources = &sub; data.NumSubresources = 1;
			device->CreateTexture(td, &data, &tex);
		}
		it = texCache.emplace(t, tex).first;
	}
	return it->second ? it->second->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE) : nullptr;
}

NukeDiligent::Impl::RT NukeDiligent::Impl::MakeRT(int w, int h)
{
	RT rt; rt.w = w; rt.h = h;
	const bool ms = samples > 1;

	// HDR (RGBA16F) color: geometry target (no MSAA) / resolve destination (MSAA). The post pass reads it.
	TextureDesc cd;
	cd.Name = "RT Color HDR"; cd.Type = RESOURCE_DIM_TEX_2D; cd.Width = (Uint32)w; cd.Height = (Uint32)h;
	cd.Format = SceneFmt(); cd.BindFlags = BIND_RENDER_TARGET | BIND_SHADER_RESOURCE;   // RGBA16F (HDR) or RGBA8 (off)
	device->CreateTexture(cd, nullptr, &rt.color);
	if (rt.color) rt.hdrSRV = rt.color->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);

	// LDR (RGBA8) post output — the tonemapped result the UI shows / a material samples.
	TextureDesc pd;
	pd.Name = "RT Color Post"; pd.Type = RESOURCE_DIM_TEX_2D; pd.Width = (Uint32)w; pd.Height = (Uint32)h;
	pd.Format = TEX_FORMAT_RGBA8_UNORM; pd.BindFlags = BIND_RENDER_TARGET | BIND_SHADER_RESOURCE;
	device->CreateTexture(pd, nullptr, &rt.post);
	if (rt.post) { rt.postRTV = rt.post->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET); rt.srv = rt.post->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE); }

	if (ms)
	{
		TextureDesc cm = cd; cm.Name = "RT Color HDR MS"; cm.SampleCount = samples; cm.BindFlags = BIND_RENDER_TARGET;
		device->CreateTexture(cm, nullptr, &rt.colorMS);
		if (rt.colorMS) rt.rtv = rt.colorMS->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET);
		TextureDesc dm; dm.Name = "RT Depth MS"; dm.Type = RESOURCE_DIM_TEX_2D; dm.Width = (Uint32)w; dm.Height = (Uint32)h;
		dm.Format = TEX_FORMAT_D32_FLOAT; dm.BindFlags = BIND_DEPTH_STENCIL; dm.SampleCount = samples;
		device->CreateTexture(dm, nullptr, &rt.depthMS);
		if (rt.depthMS) rt.dsv = rt.depthMS->GetDefaultView(TEXTURE_VIEW_DEPTH_STENCIL);
	}
	else
	{
		if (rt.color) rt.rtv = rt.color->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET);
		TextureDesc dd; dd.Name = "RT Depth"; dd.Type = RESOURCE_DIM_TEX_2D; dd.Width = (Uint32)w; dd.Height = (Uint32)h;
		dd.Format = TEX_FORMAT_D32_FLOAT; dd.BindFlags = BIND_DEPTH_STENCIL;
		device->CreateTexture(dd, nullptr, &rt.depth);
		if (rt.depth) rt.dsv = rt.depth->GetDefaultView(TEXTURE_VIEW_DEPTH_STENCIL);
	}
	return rt;
}

// (Re)create the HDR intermediate used when a camera renders to target 0 (the Player's backbuffer path):
// geometry -> this HDR target (MS if enabled) -> resolve -> HDR single -> post pass -> the swap-chain backbuffer.
// Reuses MakeRT (its unused `post` LDR texture is harmless; the post pass writes the real backbuffer instead).
void NukeDiligent::Impl::EnsureBackbufferMS(int w, int h)
{
	if (w <= 0 || h <= 0) return;
	if (backbufferMS.color && backbufferMS.w == w && backbufferMS.h == h) return;
	backbufferMS = MakeRT(w, h);
}

// Final post-process pipeline: fullscreen pass that tonemaps the HDR scene into an LDR target.
// After swap-chain creation: if the monitor is in HDR mode, switch the swap chain to the HDR10 (PQ, Rec2020)
// colour space so the backbuffer drives the display as real HDR. Falls back to plain (SDR) on failure.
void NukeDiligent::Impl::SetupHDROutput()
{
	hdr10Active = false;
#ifdef _WIN32   // DXGI HDR10 path — Windows / D3D11 only
	RefCntAutoPtr<ISwapChainD3D11> scD3D11(swapChain.RawPtr(), IID_SwapChainD3D11);
	if (!scD3D11) return;
	IDXGISwapChain* dxgi = scD3D11->GetDXGISwapChain();
	if (!dxgi) return;
	IDXGISwapChain3* sc3 = nullptr;
	if (FAILED(dxgi->QueryInterface(__uuidof(IDXGISwapChain3), (void**)&sc3)) || !sc3) return;

	bool displayHDR = false;
	IDXGIOutput* out = nullptr;
	if (SUCCEEDED(sc3->GetContainingOutput(&out)) && out)
	{
		IDXGIOutput6* out6 = nullptr;
		if (SUCCEEDED(out->QueryInterface(__uuidof(IDXGIOutput6), (void**)&out6)) && out6)
		{
			DXGI_OUTPUT_DESC1 od{};
			if (SUCCEEDED(out6->GetDesc1(&od)))
				displayHDR = (od.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020);
			out6->Release();
		}
		out->Release();
	}

	if (displayHDR)
	{
		UINT support = 0;
		if (SUCCEEDED(sc3->CheckColorSpaceSupport(DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020, &support)) &&
		    (support & DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT) &&
		    SUCCEEDED(sc3->SetColorSpace1(DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020)))
		{
			hdr10Active = true;
			IDXGISwapChain4* sc4 = nullptr;
			if (SUCCEEDED(sc3->QueryInterface(__uuidof(IDXGISwapChain4), (void**)&sc4)) && sc4)
			{
				DXGI_HDR_METADATA_HDR10 md{};   // Rec2020 primaries + D65 white, 1000-nit peak (chromaticity * 50000)
				md.RedPrimary[0]   = 34000; md.RedPrimary[1]   = 16000;
				md.GreenPrimary[0] = 13250; md.GreenPrimary[1] = 34500;
				md.BluePrimary[0]  =  7500; md.BluePrimary[1]  =  3000;
				md.WhitePoint[0]   = 15635; md.WhitePoint[1]   = 16450;
				md.MaxMasteringLuminance = 1000 * 10000; md.MinMasteringLuminance = 1;
				md.MaxContentLightLevel = 1000; md.MaxFrameAverageLightLevel = 400;
				sc4->SetHDRMetaData(DXGI_HDR_METADATA_TYPE_HDR10, sizeof(md), &md);
				sc4->Release();
			}
		}
	}
	sc3->Release();
	cout << "[NukeDiligent]\tHDR10 output " << (hdr10Active ? "ACTIVE" : "off (display not in HDR mode / unsupported)") << endl;
#endif   // _WIN32
}

void NukeDiligent::Impl::CreatePostResources()
{
	postPSO.Release(); postSRB.Release(); postPSOBB.Release(); postSRBBB.Release(); postCB.Release();   // rebuild-safe
	std::string vs = shaderSource("post.vs"), ps = shaderSource("post.ps");
	if (vs.empty() || ps.empty()) { cout << "[NukeDiligent]\tpost shaders missing" << endl; return; }
	ShaderCreateInfo sci; sci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
	RefCntAutoPtr<IShader> v, p;
	sci.Desc = {"Post VS", SHADER_TYPE_VERTEX, true}; sci.Source = vs.c_str(); device->CreateShader(sci, &v);
	sci.Desc = {"Post PS", SHADER_TYPE_PIXEL, true};  sci.Source = ps.c_str(); device->CreateShader(sci, &p);
	if (!v || !p) return;

	BufferDesc cbd; cbd.Name = "PostCB"; cbd.Size = sizeof(float) * 4; cbd.Usage = USAGE_DYNAMIC;
	cbd.BindFlags = BIND_UNIFORM_BUFFER; cbd.CPUAccessFlags = CPU_ACCESS_WRITE;
	device->CreateBuffer(cbd, nullptr, &postCB);

	// Build one post PSO per output format: RGBA8 for RT targets, swap-chain format for the backbuffer.
	auto buildPost = [&](TEXTURE_FORMAT fmt, const char* name,
	                     RefCntAutoPtr<IPipelineState>& pso, RefCntAutoPtr<IShaderResourceBinding>& srb, IShaderResourceVariable*& var)
	{
		GraphicsPipelineStateCreateInfo ci; ci.PSODesc.Name = name;
		auto& gp = ci.GraphicsPipeline;
		gp.NumRenderTargets = 1; gp.RTVFormats[0] = fmt;
		gp.DSVFormat = TEX_FORMAT_UNKNOWN;
		gp.PrimitiveTopology = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		gp.RasterizerDesc.CullMode = CULL_MODE_NONE;
		gp.DepthStencilDesc.DepthEnable = False;
		gp.InputLayout.NumElements = 0;
		ShaderResourceVariableDesc vars[] = {{SHADER_TYPE_PIXEL, "g_HDR", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC}};
		ci.PSODesc.ResourceLayout.Variables = vars; ci.PSODesc.ResourceLayout.NumVariables = 1;
		SamplerDesc samp; samp.MinFilter = FILTER_TYPE_LINEAR; samp.MagFilter = FILTER_TYPE_LINEAR; samp.MipFilter = FILTER_TYPE_LINEAR;
		samp.AddressU = TEXTURE_ADDRESS_CLAMP; samp.AddressV = TEXTURE_ADDRESS_CLAMP;
		ImmutableSamplerDesc imm[] = {{SHADER_TYPE_PIXEL, "g_HDR", samp}};
		ci.PSODesc.ResourceLayout.ImmutableSamplers = imm; ci.PSODesc.ResourceLayout.NumImmutableSamplers = 1;
		ci.pVS = v; ci.pPS = p;
		device->CreateGraphicsPipelineState(ci, &pso);
		if (pso)
		{
			if (auto* c = pso->GetStaticVariableByName(SHADER_TYPE_PIXEL, "PostCB")) c->Set(postCB);
			pso->CreateShaderResourceBinding(&srb, true);
			var = srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_HDR");
		}
	};
	buildPost(TEX_FORMAT_RGBA8_UNORM, "Post PSO", postPSO, postSRB, postHdrVar);
	TEXTURE_FORMAT bbFmt = swapChain ? swapChain->GetDesc().ColorBufferFormat : TEX_FORMAT_RGBA8_UNORM;
	buildPost(bbFmt, "Post PSO BB", postPSOBB, postSRBBB, postHdrVarBB);
}

// Tonemap HDR -> output into dstRTV. toBackbuffer picks the backbuffer PSO + (when HDR10 is live) PQ encoding.
void NukeDiligent::Impl::RunPostPass(ITextureView* hdrSRV, ITextureView* dstRTV, int w, int h, bool toBackbuffer)
{
	IPipelineState*          pso = toBackbuffer ? postPSOBB : postPSO;
	IShaderResourceBinding*  srb = toBackbuffer ? postSRBBB : postSRB;
	IShaderResourceVariable* var = toBackbuffer ? postHdrVarBB : postHdrVar;
	if (!pso || !srb || !hdrSRV || !dstRTV) return;
	const float mode = !hdr ? 0.0f : ((toBackbuffer && hdr10Active) ? 2.0f : 1.0f);   // 0=passthrough,1=sRGB SDR,2=HDR10 PQ
	{ MapHelper<float> cb(context, postCB, MAP_WRITE, MAP_FLAG_DISCARD); cb[0] = exposure; cb[1] = mode; cb[2] = 0; cb[3] = 0; }
	context->SetRenderTargets(1, &dstRTV, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	Viewport vp; vp.TopLeftX = 0; vp.TopLeftY = 0; vp.Width = (float)w; vp.Height = (float)h; vp.MinDepth = 0; vp.MaxDepth = 1;
	context->SetViewports(1, &vp, w, h);
	if (var) var->Set(hdrSRV);
	context->SetPipelineState(pso);
	context->CommitShaderResources(srb, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	DrawAttribs da{3, DRAW_FLAG_VERIFY_STATES};
	context->Draw(da);
}

static void glfw_error(int code, const char* desc)
{
	fprintf(stderr, "[NukeDiligent] GLFW error %d: %s\n", code, desc);
}

// GLFW input -> iRender neutral callbacks. The renderer forwards raw input via
// the interface; whoever set the callbacks (the UI module / editor) interprets it.
static void cb_cursorpos(GLFWwindow* w, double x, double y)
{
	auto* self = static_cast<NukeDiligent*>(glfwGetWindowUserPointer(w));
	if (self && self->_UImove) self->_UImove((int)x, (int)y);
}
static void cb_mousebtn(GLFWwindow* w, int button, int action, int /*mods*/)
{
	auto* self = static_cast<NukeDiligent*>(glfwGetWindowUserPointer(w));
	if (!self || !self->_UImouse) return;
	double x = 0, y = 0; glfwGetCursorPos(w, &x, &y);
	self->_UImouse(button, action == GLFW_PRESS ? 1 : 0, (int)x, (int)y);
}
static void cb_scroll(GLFWwindow* w, double /*xo*/, double yo)
{
	auto* self = static_cast<NukeDiligent*>(glfwGetWindowUserPointer(w));
	if (!self || !self->_UImouseWheel) return;
	double x = 0, y = 0; glfwGetCursorPos(w, &x, &y);
	self->_UImouseWheel(0, (int)yo, (int)x, (int)y);
}
static void cb_key(GLFWwindow* w, int key, int /*scancode*/, int action, int mods)
{
	auto* self = static_cast<NukeDiligent*>(glfwGetWindowUserPointer(w));
	if (self && self->_UIkey) self->_UIkey(key, action, mods);
}
static void cb_char(GLFWwindow* w, unsigned int cp)
{
	auto* self = static_cast<NukeDiligent*>(glfwGetWindowUserPointer(w));
	if (self && self->_UIchar) self->_UIchar(cp);
}

NukeDiligent::NukeDiligent() : m_impl(new Impl()) {}
NukeDiligent::~NukeDiligent() { delete m_impl; }

void NukeDiligent::setShaderSource(const char* name, const char* source)
{
	if (name && source) m_impl->shaderSrc[name] = source;
}

uint64_t NukeDiligent::createShaderPipeline(const char* vs, const char* ps)
{
	if (!vs || !ps) return 0;
	return m_impl->MakeWorldPSO(vs, ps, "Shader");   // world-type PSO (layout/CBs) from custom VS+PS
}

int NukeDiligent::init(const WindowDesc& desc)
{
	int w = desc.w, h = desc.h;
	cout << "[NukeDiligent]\tinit(" << w << ", " << h << ")" << endl;
	if (w <= 0 || h <= 0) { cout << "[NukeDiligent]\tbad size, using 1280x720" << endl; w = 1280; h = 720; }

	glfwSetErrorCallback(glfw_error);
	if (!glfwInit()) { cout << "[NukeDiligent]\tglfwInit failed" << endl; return 1; }

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // Diligent owns the graphics API
	glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);     // create hidden; show after the dark title-bar attr is set
	// Window properties from the neutral WindowDesc (config-driven, see iRender).
	glfwWindowHint(GLFW_DECORATED, desc.decorated ? GLFW_TRUE : GLFW_FALSE);
	glfwWindowHint(GLFW_RESIZABLE, desc.resizable ? GLFW_TRUE : GLFW_FALSE);
	glfwWindowHint(GLFW_FLOATING,  desc.floating  ? GLFW_TRUE : GLFW_FALSE);
	glfwWindowHint(GLFW_MAXIMIZED, desc.maximized ? GLFW_TRUE : GLFW_FALSE);
	if (desc.transparent)
		glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, GLFW_TRUE); // NOTE: true see-through also needs swapchain alpha (DComp) — not yet wired
	GLFWmonitor* monitor = desc.fullscreen ? glfwGetPrimaryMonitor() : nullptr;
	const char*  title   = (desc.title && desc.title[0]) ? desc.title : "NukeEngine";
	m_window = glfwCreateWindow(w, h, title, monitor, nullptr);
	if (!m_window) { cout << "[NukeDiligent]\tglfwCreateWindow failed" << endl; glfwTerminate(); return 1; }

	glfwSetWindowUserPointer(m_window, this);
	glfwSetCursorPosCallback(m_window, cb_cursorpos);
	glfwSetMouseButtonCallback(m_window, cb_mousebtn);
	glfwSetScrollCallback(m_window, cb_scroll);
	glfwSetKeyCallback(m_window, cb_key);
	glfwSetCharCallback(m_window, cb_char);

	HWND hWnd = glfwGetWin32Window(m_window);

	// Give the OS window the application's own icon (the host .exe's first icon
	// group) so the editor/game window matches the taskbar/console icon. Generic:
	// no dependence on a specific resource id, works for any host exe.
	{
		wchar_t exePath[MAX_PATH];
		if (GetModuleFileNameW(nullptr, exePath, MAX_PATH))
		{
			HICON hBig = nullptr, hSmall = nullptr;
			ExtractIconExW(exePath, 0, &hBig, &hSmall, 1);
			if (hBig)   SendMessageW(hWnd, WM_SETICON, ICON_BIG,   (LPARAM)hBig);
			if (hSmall) SendMessageW(hWnd, WM_SETICON, ICON_SMALL, (LPARAM)hSmall);
		}
	}

	// Dark title bar to match the editor's dark theme (no more glowing white bar).
	// Set while the window is still hidden so the first non-client paint is dark.
	{
		BOOL dark = TRUE;
		HRESULT hr = DwmSetWindowAttribute(hWnd, DWMWA_USE_IMMERSIVE_DARK_MODE, &dark, sizeof(dark));
		cout << "[NukeDiligent]\tdark title bar hr=0x" << std::hex << hr << std::dec << endl;
	}
	if (desc.opacity < 1.0f)
		glfwSetWindowOpacity(m_window, desc.opacity);
	glfwShowWindow(m_window);

	auto* pFactory = GetEngineFactoryD3D11();
	EngineD3D11CreateInfo EngineCI;
	pFactory->CreateDeviceAndContextsD3D11(EngineCI, &m_impl->device, &m_impl->context);
	if (!m_impl->device) { cout << "[NukeDiligent]\tD3D11 device creation failed" << endl; return 1; }

	Win32NativeWindow Window{ hWnd };
	SwapChainDesc SCDesc;
	// Match the World PSO + offscreen render targets (RGBA8_UNORM). Diligent defaults the
	// backbuffer to *_SRGB, which mismatches the world PSO when rendering straight to the
	// window (the Player) — "RTV format does not match PSO" spam. Keep one format everywhere.
	// HDR10 display output (Player): a 10-bit backbuffer carries the PQ-encoded HDR signal. Plain RGBA8 otherwise.
	SCDesc.ColorBufferFormat = m_impl->hdrOutput ? TEX_FORMAT_RGB10A2_UNORM : TEX_FORMAT_RGBA8_UNORM;
	pFactory->CreateSwapChainD3D11(m_impl->device, m_impl->context, SCDesc,
	                               FullScreenModeDesc{}, Window, &m_impl->swapChain);
	if (!m_impl->swapChain) { cout << "[NukeDiligent]\tswap chain creation failed" << endl; return 1; }

	if (m_impl->hdrOutput) m_impl->SetupHDROutput();   // set the HDR10 colour space if the monitor is in HDR mode

	const SwapChainDesc& scd = m_impl->swapChain->GetDesc();
	m_impl->CreateUIPipeline(scd.ColorBufferFormat, scd.DepthBufferFormat);
	m_impl->CreateWorldPipeline();

	width  = w;
	height = h;

	cout << "[NukeDiligent]\tdevice=" << m_impl->device.RawPtr()
	     << " swapChain=" << m_impl->swapChain.RawPtr() << endl;

	if (_UIinit)
	{
		cout << "[NukeDiligent]\tUI init" << endl;
		_UIinit();
	}
	return 0;
}

int NukeDiligent::render()
{
	glfwPollEvents();

	// Follow the window: resize the swap chain (and report size to the UI via
	// width/height) when the framebuffer changes. Skip rendering when minimized.
	int fbw = 0, fbh = 0;
	glfwGetFramebufferSize(m_window, &fbw, &fbh);
	if (fbw <= 0 || fbh <= 0)
		return 1;
	if (fbw != width || fbh != height)
	{
		width  = fbw;
		height = fbh;
		m_impl->swapChain->Resize((Uint32)fbw, (Uint32)fbh);
	}

	// Apply deferred MSAA / HDR changes here — between frames, after the previous frame's draw was submitted,
	// so the RT textures the UI referenced are no longer in any pending draw list (rebuilding mid-frame frees
	// them and crashes renderDrawLists). Both flip RTV/texture formats, so one rebuild covers both.
	if (m_impl->pendingSamples > 0 || m_impl->pendingHDR >= 0)
	{
		bool changed = false;
		if (m_impl->pendingSamples > 0 && (Uint8)m_impl->pendingSamples != m_impl->samples) { m_impl->samples = (Uint8)m_impl->pendingSamples; changed = true; }
		if (m_impl->pendingHDR >= 0 && (bool)m_impl->pendingHDR != m_impl->hdr) { m_impl->hdr = m_impl->pendingHDR != 0; changed = true; }
		if (changed)
		{
			m_impl->RebuildForMSAA();
			std::cout << "[NukeDiligent]\tMSAA " << (int)m_impl->samples << "x, HDR " << (m_impl->hdr ? "on" : "off") << std::endl;
		}
		m_impl->pendingSamples = -1; m_impl->pendingHDR = -1;
	}

	// 0) Clear the backbuffer up front. In the editor it's the UI background (the world
	//    goes to off-screen RTs); in the Player the world renders straight to it, where
	//    beginCamera(target 0) overwrites this clear. Either way it must NOT be cleared
	//    again after onRender, or the Player's world-to-backbuffer would be wiped.
	ITextureView* pRTV = m_impl->swapChain->GetCurrentBackBufferRTV();
	ITextureView* pDSV = m_impl->swapChain->GetDepthBufferDSV();
	const float clearColor[] = { 0.10f, 0.12f, 0.16f, 1.0f };
	m_impl->context->SetRenderTargets(1, &pRTV, pDSV, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	m_impl->context->ClearRenderTarget(pRTV, clearColor, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	m_impl->context->ClearDepthStencil(pDSV, CLEAR_DEPTH_FLAG, 1.f, 0, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

	// 1) World passes. onRender drives World::Render, which calls beginCamera (binds +
	//    clears the target) and renderObject — to off-screen RTs and/or the backbuffer.
	for (auto& cb : m_impl->onRender) cb();

	// 2) UI pass: rebind the backbuffer WITHOUT clearing (so a world rendered straight to
	//    it survives) and draw the UI on top.
	m_impl->context->SetRenderTargets(1, &pRTV, pDSV, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	for (auto& cb : m_impl->onGUI) cb();

	m_impl->swapChain->Present();
	return 1;
}

NukeDiligent::Impl::MeshGPU* NukeDiligent::Impl::GetMeshGPU(Mesh* mesh)
{
	if (!mesh || mesh->numVerts <= 0) return nullptr;
	auto it = meshCache.find(mesh);
	if (it == meshCache.end())
	{
		if (!mesh->vertexArray || !mesh->normalArray)   // immutable buffers need init data
		{
			std::cout << "[NukeDiligent]\tmesh '" << mesh->name << "' has null vertex/normal data (numVerts="
			          << mesh->numVerts << ") — skipping" << std::endl;
			return nullptr;
		}
		// Build (and cache) GPU vertex buffers for this mesh: positions + normals + uv.
		MeshGPU g; g.numVerts = mesh->numVerts;
		const Uint64 sz3 = (Uint64)mesh->numVerts * 3 * sizeof(float);
		const Uint64 sz2 = (Uint64)mesh->numVerts * 2 * sizeof(float);
		BufferDesc bd; bd.BindFlags = BIND_VERTEX_BUFFER; bd.Usage = USAGE_IMMUTABLE;
		bd.Size = sz3; bd.Name = "mesh pos"; BufferData pdat{mesh->vertexArray, sz3}; device->CreateBuffer(bd, &pdat, &g.pos);
		bd.Size = sz3; bd.Name = "mesh nrm"; BufferData ndat{mesh->normalArray, sz3}; device->CreateBuffer(bd, &ndat, &g.nrm);
		std::vector<float> zeroUV;
		const float* uvSrc = mesh->uvArray;
		if (!uvSrc) { zeroUV.assign((size_t)mesh->numVerts * 2, 0.0f); uvSrc = zeroUV.data(); }   // mesh has no UVs
		bd.Size = sz2; bd.Name = "mesh uv"; BufferData udat{uvSrc, sz2}; device->CreateBuffer(bd, &udat, &g.uv);
		it = meshCache.emplace(mesh, std::move(g)).first;
	}
	MeshGPU& g = it->second;
	if (!g.pos || !g.nrm || !g.uv) return nullptr;
	return &g;
}

void NukeDiligent::renderObject(Mesh* mesh, Material* mat,
                                const float pos[3], const float quat[4], const float scale[3])
{
	if (m_impl->worldPipes.empty()) return;
	Impl::MeshGPU* gp = m_impl->GetMeshGPU(mesh);
	if (!gp) return;
	Impl::MeshGPU& g = *gp;

	float4x4 world = float4x4::Scale(scale[0], scale[1], scale[2])
	               * Diligent::Quaternion<float>(quat[0], quat[1], quat[2], quat[3]).ToMatrix()
	               * float4x4::Translation(pos[0], pos[1], pos[2]);
	float4x4 wvp = world * m_impl->curView * m_impl->curProj;

	struct CBData { float4x4 wvp; float4x4 world; };
	{
		MapHelper<CBData> cb(m_impl->context, m_impl->worldCB, MAP_WRITE, MAP_FLAG_DISCARD);
		cb->wvp = wvp; cb->world = world;
	}

	// Material: base color + PBR maps/params (fallbacks when none).
	float col[4] = { 1, 1, 1, 1 };
	float metallic = 0.0f, roughness = 0.6f;
	float emissive[3] = { 0, 0, 0 }, emissiveI = 0.0f;
	ITextureView* srv = nullptr; ITextureView* nsrv = nullptr;
	ITextureView* mrsrv = nullptr; ITextureView* aosrv = nullptr; ITextureView* emsrv = nullptr;
	if (mat)
	{
		col[0] = (float)mat->color.r; col[1] = (float)mat->color.g; col[2] = (float)mat->color.b; col[3] = (float)mat->color.a;
		metallic = mat->metallic; roughness = mat->roughness;
		emissive[0] = (float)mat->emissive.r; emissive[1] = (float)mat->emissive.g; emissive[2] = (float)mat->emissive.b;
		emissiveI = mat->emissiveIntensity;
		if (mat->diff) srv   = m_impl->GetTexSRV(mat->diff);
		if (mat->norm) nsrv  = m_impl->GetTexSRV(mat->norm);
		if (mat->mr)   mrsrv = m_impl->GetTexSRV(mat->mr);
		if (mat->ao)   aosrv = m_impl->GetTexSRV(mat->ao);
		if (mat->em)   emsrv = m_impl->GetTexSRV(mat->em);
	}
	// Pick the pipeline for this material's shader (fallback to the built-in "world" pipeline).
	uint64_t h = (mat && mat->shader && mat->shader->rendererHandle) ? mat->shader->rendererHandle
	                                                                  : m_impl->defaultWorldHandle;
	auto pit = m_impl->worldPipes.find(h);
	if (pit == m_impl->worldPipes.end()) pit = m_impl->worldPipes.find(m_impl->defaultWorldHandle);
	if (pit == m_impl->worldPipes.end()) return;
	Impl::WorldPipe& wp = pit->second;

	// Material constant buffer: standard color @0 + params @16, then the shader's custom props at
	// their engine-parsed offsets (Shader::props). The renderer consumes the schema the engine
	// parsed from the shader source — it never reads shader files itself.
	{
		MapHelper<Uint8> mb(m_impl->context, m_impl->worldMatCB, MAP_WRITE, MAP_FLAG_DISCARD);
		Uint8* p = mb;
		memset(p, 0, Impl::kMatCBBytes);
		memcpy(p + 0, col, sizeof(float) * 4);
		float prm[4] = { srv ? 1.0f : 0.0f, nsrv ? 1.0f : 0.0f, metallic, roughness };
		memcpy(p + 16, prm, sizeof(float) * 4);   // g_Params (hasBase, hasNormal, metallic, roughness)
		float prm2[4] = { mrsrv ? 1.0f : 0.0f, aosrv ? 1.0f : 0.0f, emsrv ? 1.0f : 0.0f, 1.0f };
		memcpy(p + 32, prm2, sizeof(float) * 4);  // g_Params2 (hasMR, hasAO, hasEm, aoStrength)
		float emv[4] = { emissive[0], emissive[1], emissive[2], emissiveI };
		memcpy(p + 48, emv, sizeof(float) * 4);   // g_Emissive2 (rgb, intensity)
		if (mat && mat->shader)
			for (const nuke::ShaderProp& sp : mat->shader->props)
			{
				// Value from the material INSTANCE's prop map (data only — no engine symbols linked);
				// unset -> the shader's HLSL default.
				auto pv = mat->props.find(sp.name);
				const float* v = (pv != mat->props.end()) ? pv->second.data() : sp.def;
				uint32_t bytes = (uint32_t)sp.components * sizeof(float);
				if (sp.offset + bytes <= Impl::kMatCBBytes) memcpy(p + sp.offset, v, bytes);
			}
	}

	if (wp.texVar)
		wp.texVar->Set(srv ? srv : m_impl->whiteTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
	ITextureView* whiteSRV = m_impl->whiteTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
	if (wp.normVar)
		wp.normVar->Set(nsrv ? nsrv : m_impl->flatNormTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
	if (wp.mrVar) wp.mrVar->Set(mrsrv ? mrsrv : whiteSRV);
	if (wp.aoVar) wp.aoVar->Set(aosrv ? aosrv : whiteSRV);
	if (wp.emVar) wp.emVar->Set(emsrv ? emsrv : whiteSRV);
	if (wp.shadowVar) wp.shadowVar->Set(m_impl->shadowSRV ? m_impl->shadowSRV : whiteSRV);
	if (wp.cubeVar && m_impl->shadowCubeSRV) wp.cubeVar->Set(m_impl->shadowCubeSRV);

	IDeviceContext* ctx = m_impl->context;
	IBuffer* vbs[]    = { g.pos, g.nrm, g.uv };
	Uint64   offs[]   = { 0, 0, 0 };
	ctx->SetVertexBuffers(0, 3, vbs, offs, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, SET_VERTEX_BUFFERS_FLAG_RESET);
	// Pick the blend variant for this material (engine sorts transparent/additive back-to-front).
	IPipelineState* pso = wp.pso;
	if (mat) { if (mat->blendMode == 1 && wp.psoBlend) pso = wp.psoBlend; else if (mat->blendMode == 2 && wp.psoAdd) pso = wp.psoAdd; }
	ctx->SetPipelineState(pso);
	ctx->CommitShaderResources(wp.srb, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	DrawAttribs da{(Uint32)g.numVerts, DRAW_FLAG_VERIFY_STATES};
	ctx->Draw(da);
}

void NukeDiligent::renderSelectionOutline(Mesh* mesh, const float pos[3], const float quat[4], const float scale[3])
{
	if (!m_impl->outlineMaskPSO || !m_impl->outlineEdgePSO || !m_impl->curRTV) return;
	Impl::MeshGPU* gp = m_impl->GetMeshGPU(mesh);
	if (!gp) return;
	Impl::MeshGPU& g = *gp;
	m_impl->EnsureOutlineMask(m_impl->curRTW, m_impl->curRTH);
	if (!m_impl->outlineMaskRTV || !m_impl->outlineMaskSRV) return;

	IDeviceContext* ctx = m_impl->context;

	// --- pass 1: render the selected mesh into the mask RT (alpha = 1 over the object) ---
	{
		float4x4 world = float4x4::Scale(scale[0], scale[1], scale[2])
		               * Diligent::Quaternion<float>(quat[0], quat[1], quat[2], quat[3]).ToMatrix()
		               * float4x4::Translation(pos[0], pos[1], pos[2]);
		float4x4 wvp = world * m_impl->curView * m_impl->curProj;
		struct CBData { float4x4 wvp; float4x4 world; };
		{ MapHelper<CBData> cb(ctx, m_impl->worldCB, MAP_WRITE, MAP_FLAG_DISCARD); cb->wvp = wvp; cb->world = world; }

		const float zero[4] = { 0, 0, 0, 0 };
		ctx->SetRenderTargets(1, &m_impl->outlineMaskRTV, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		ctx->ClearRenderTarget(m_impl->outlineMaskRTV, zero, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		IBuffer* vbs[]  = { g.pos, g.nrm, g.uv };
		Uint64   offs[] = { 0, 0, 0 };
		ctx->SetVertexBuffers(0, 3, vbs, offs, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, SET_VERTEX_BUFFERS_FLAG_RESET);
		ctx->SetPipelineState(m_impl->outlineMaskPSO);
		ctx->CommitShaderResources(m_impl->outlineMaskSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		DrawAttribs da{(Uint32)g.numVerts, DRAW_FLAG_VERIFY_STATES};
		ctx->Draw(da);
	}

	// --- pass 2: fullscreen edge-detect over the mask -> draw the border into the camera RT ---
	{
		struct EdgeData { float texel[4]; };
		{
			MapHelper<EdgeData> cb(ctx, m_impl->outlineEdgeCB, MAP_WRITE, MAP_FLAG_DISCARD);
			cb->texel[0] = (m_impl->curRTW > 0) ? 1.0f / m_impl->curRTW : 0.0f;
			cb->texel[1] = (m_impl->curRTH > 0) ? 1.0f / m_impl->curRTH : 0.0f;
			cb->texel[2] = 2.0f;   // outline thickness in pixels (constant on screen)
			cb->texel[3] = 0.0f;
		}
		ctx->SetRenderTargets(1, &m_impl->curRTV, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		ctx->SetVertexBuffers(0, 0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, SET_VERTEX_BUFFERS_FLAG_RESET);
		if (m_impl->outlineEdgeMaskVar) m_impl->outlineEdgeMaskVar->Set(m_impl->outlineMaskSRV);
		ctx->SetPipelineState(m_impl->outlineEdgePSO);
		ctx->CommitShaderResources(m_impl->outlineEdgeSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		DrawAttribs fs{3, DRAW_FLAG_VERIFY_STATES};
		ctx->Draw(fs);
	}
}

void NukeDiligent::loop()
{
	if (!m_window)
	{
		cout << "[NukeDiligent]\tloop() with no window - init failed earlier." << endl;
		return;
	}
	while (!glfwWindowShouldClose(m_window))
		render();
}

void NukeDiligent::deinit()
{
	for (auto& cb : m_impl->onClose) cb();
	m_impl->swapChain.Release();
	m_impl->context.Release();
	m_impl->device.Release();
	if (m_window) { glfwDestroyWindow(m_window); m_window = nullptr; }
	glfwTerminate();
}

void NukeDiligent::update() {}

char* NukeDiligent::getEngine()  { return (char*)"Diligent - "; }
char* NukeDiligent::getVersion() { return (char*)"0.1.0"; }

void NukeDiligent::setOnGUI(bst::function<void(void)> cb)    { m_impl->onGUI.push_back(cb); }
void NukeDiligent::setOnRender(bst::function<void(void)> cb) { m_impl->onRender.push_back(cb); }
void NukeDiligent::setOnClose(bst::function<void()> cb)      { m_impl->onClose.push_back(cb); }

// Input is routed via iRender callbacks in a later milestone.
void NukeDiligent::keyboard(int, int, int, int) {}
void NukeDiligent::mouseMove(double, double) {}
void NukeDiligent::mouseClick(int, int, int) {}
void NukeDiligent::setCursorMode(int) {}
void NukeDiligent::rawMouse(double, double) {}
void NukeDiligent::mouseEnterLeave(int) {}
void NukeDiligent::setWindowTitle(const char* title) { if (m_window && title) glfwSetWindowTitle(m_window, title); }
bool NukeDiligent::isWindowFocused() { return m_window && glfwGetWindowAttrib(m_window, GLFW_FOCUSED) != 0; }
bool NukeDiligent::isWindowMaximized() { return m_window && glfwGetWindowAttrib(m_window, GLFW_MAXIMIZED) != 0; }
void NukeDiligent::setWindowMaximized(bool m) { if (!m_window) return; if (m) glfwMaximizeWindow(m_window); else glfwRestoreWindow(m_window); }
void NukeDiligent::bindRenderTarget(uint64_t id)
{
	if (id == 0) { m_impl->uiRTV = nullptr; m_impl->uiTW = m_impl->uiTH = 0; return; }
	auto it = m_impl->rts.find(id);
	if (it == m_impl->rts.end()) { m_impl->uiRTV = nullptr; m_impl->uiTW = m_impl->uiTH = 0; return; }
	m_impl->uiRTV = it->second.rtv; m_impl->uiTW = (Uint32)it->second.w; m_impl->uiTH = (Uint32)it->second.h;
}
void NukeDiligent::invalidateTexture(Texture* t) { if (t) m_impl->texCache.erase(t); }   // re-uploaded on next GetTexSRV
void NukeDiligent::getCursorPos(double& x, double& y) { x = y = 0; if (m_window) glfwGetCursorPos(m_window, &x, &y); }
bool NukeDiligent::isMouseButtonDown(int b) { return m_window && glfwGetMouseButton(m_window, b) == GLFW_PRESS; }

// ---- Neutral UI seam: generic 2D draw (no ImGui types) ----

uint64_t NukeDiligent::createTexture2D(const void* rgba, int width, int height)
{
	if (!m_impl->device || !rgba || width <= 0 || height <= 0) return 0;
	TextureDesc Desc;
	Desc.Name      = "UI Texture";
	Desc.Type      = RESOURCE_DIM_TEX_2D;
	Desc.Width     = (Uint32)width;
	Desc.Height    = (Uint32)height;
	Desc.Format    = TEX_FORMAT_RGBA8_UNORM;
	Desc.Usage     = USAGE_DEFAULT;
	Desc.BindFlags = BIND_SHADER_RESOURCE;
	TextureSubResData mip0{rgba, (Uint64)width * 4};
	TextureData       init{&mip0, 1};
	RefCntAutoPtr<ITexture> tex;
	m_impl->device->CreateTexture(Desc, &init, &tex);
	if (!tex) return 0;
	ITextureView* view = tex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
	uint64_t handle = reinterpret_cast<uint64_t>(view);
	m_impl->textures[handle] = tex; // keep alive
	return handle;
}

void NukeDiligent::destroyTexture2D(uint64_t handle)
{
	m_impl->textures.erase(handle);
}

uint64_t NukeDiligent::createRenderTarget(int w, int h)
{
	if (w <= 0 || h <= 0) return 0;
	uint64_t id = ++m_impl->rtCounter;
	m_impl->rts[id] = m_impl->MakeRT(w, h);
	return id;
}

void NukeDiligent::resizeRenderTarget(uint64_t id, int w, int h)
{
	if (w <= 0 || h <= 0) return;
	auto it = m_impl->rts.find(id);
	if (it == m_impl->rts.end()) return;
	if (it->second.w == w && it->second.h == h) return;
	it->second = m_impl->MakeRT(w, h);
}

uint64_t NukeDiligent::getRenderTargetTexture(uint64_t id)
{
	auto it = m_impl->rts.find(id);
	return (it == m_impl->rts.end()) ? 0 : reinterpret_cast<uint64_t>(it->second.srv);
}

void NukeDiligent::beginCamera(const NukeCameraDesc& cam)
{
	m_impl->curTarget = cam.target;   // feedback guard: GetTexSRV won't sample the RT we draw into
	m_impl->curMSAA = false; m_impl->curResolveSrc = nullptr; m_impl->curResolveDst = nullptr;
	m_impl->curPostSrc = nullptr; m_impl->curPostDst = nullptr;
	const bool ms = m_impl->samples > 1;
	ITextureView* rtv = nullptr;
	ITextureView* dsv = nullptr;
	int w = 0, h = 0;
	if (cam.target == 0)
	{
		// The backbuffer path always renders through an HDR intermediate, then the post pass tonemaps it
		// into the actual (LDR) swap-chain backbuffer in endCamera.
		w = (int)m_impl->swapChain->GetDesc().Width;
		h = (int)m_impl->swapChain->GetDesc().Height;
		m_impl->EnsureBackbufferMS(w, h);
		Impl::RT& bb = m_impl->backbufferMS;
		rtv = bb.rtv; dsv = bb.dsv;
		if (ms) { m_impl->curMSAA = true; m_impl->curResolveSrc = bb.colorMS; m_impl->curResolveDst = bb.color; }
		m_impl->curPostSrc = bb.hdrSRV;
		m_impl->curPostDst = m_impl->swapChain->GetCurrentBackBufferRTV();
	}
	else
	{
		auto it = m_impl->rts.find(cam.target);
		if (it == m_impl->rts.end()) return;
		Impl::RT& rt = it->second;
		rtv = rt.rtv; dsv = rt.dsv; w = rt.w; h = rt.h;
		if (ms && rt.colorMS) { m_impl->curMSAA = true; m_impl->curResolveSrc = rt.colorMS; m_impl->curResolveDst = rt.color; }
		m_impl->curPostSrc = rt.hdrSRV;
		m_impl->curPostDst = rt.postRTV;
	}
	if (!rtv) return;
	m_impl->curRTV = rtv; m_impl->curRTW = w; m_impl->curRTH = h;   // for the selection-outline pass

	IDeviceContext* ctx = m_impl->context;
	ctx->SetRenderTargets(1, &rtv, dsv, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	ctx->ClearRenderTarget(rtv, cam.clear, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	if (dsv)
		ctx->ClearDepthStencil(dsv, CLEAR_DEPTH_FLAG, 1.f, 0, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	Viewport vp; vp.TopLeftX = 0; vp.TopLeftY = 0; vp.Width = (float)w; vp.Height = (float)h; vp.MinDepth = 0; vp.MaxDepth = 1;
	ctx->SetViewports(1, &vp, w, h);

	const float aspect = (h > 0) ? (float)w / (float)h : 1.0f;
	// View from the camera's basis (left-handed look-at).
	float3 P(cam.camPos[0], cam.camPos[1], cam.camPos[2]);
	float3 F = normalize(float3(cam.camFwd[0], cam.camFwd[1], cam.camFwd[2]));
	float3 U = float3(cam.camUp[0], cam.camUp[1], cam.camUp[2]);
	float3 R = normalize(cross(U, F));
	U = cross(F, R);
	m_impl->curView = float4x4(
		R.x, U.x, F.x, 0.f,
		R.y, U.y, F.y, 0.f,
		R.z, U.z, F.z, 0.f,
		-dot(P, R), -dot(P, U), -dot(P, F), 1.f);
	m_impl->curProj = float4x4::Projection(cam.fov, aspect, cam.nearZ, cam.farZ, false);

	// PBR lighting buffer for this pass: camera pos + ambient + scene lights (default sun if none).
	m_impl->curCamPos[0] = P.x; m_impl->curCamPos[1] = P.y; m_impl->curCamPos[2] = P.z;
	{
		MapHelper<Impl::FrameCBData> fb(m_impl->context, m_impl->worldFrameCB, MAP_WRITE, MAP_FLAG_DISCARD);
		memset(fb, 0, sizeof(Impl::FrameCBData));
		fb->camPos[0] = P.x; fb->camPos[1] = P.y; fb->camPos[2] = P.z;
		fb->ambient[0] = m_impl->sky.ambient[0]; fb->ambient[1] = m_impl->sky.ambient[1];
		fb->ambient[2] = m_impl->sky.ambient[2]; fb->ambient[3] = m_impl->sky.ambientIntensity;

		std::vector<NukeLight> src = m_impl->lights;
		if (src.empty())   // default directional "sun"
		{
			NukeLight sun; sun.type = 0;
			sun.dir[0] = -0.4f; sun.dir[1] = -0.85f; sun.dir[2] = -0.35f;
			sun.color[0] = sun.color[1] = sun.color[2] = 1.0f; sun.intensity = 3.0f;
			src.push_back(sun);
		}
		int n = (int)src.size(); if (n > Impl::kMaxLights) n = Impl::kMaxLights;
		fb->lightCount[0] = (float)n;
		for (int k = 0; k < n; ++k)
		{
			const NukeLight& L = src[k]; Impl::GPULight& g = fb->lights[k];
			g.posType[0] = L.pos[0]; g.posType[1] = L.pos[1]; g.posType[2] = L.pos[2]; g.posType[3] = (float)L.type;
			g.dirRange[0] = L.dir[0]; g.dirRange[1] = L.dir[1]; g.dirRange[2] = L.dir[2]; g.dirRange[3] = L.range;
			g.colorIntensity[0] = L.color[0]; g.colorIntensity[1] = L.color[1]; g.colorIntensity[2] = L.color[2]; g.colorIntensity[3] = L.intensity;
			g.spot[0] = L.spotInner; g.spot[1] = L.spotOuter;
			g.spot[2] = (float)m_impl->lightSlot[k];   // 2D shadow-map slot (-1 = none)
			g.spot[3] = (float)m_impl->lightCube[k];   // point-light cube index (-1 = none)
		}
		// Shadow maps: per-slot world->light-clip array + params for the world PS.
		for (int s = 0; s < Impl::SHADOW_SLOTS; ++s)
			memcpy(fb->shadowVP + s * 16, &m_impl->slotVP[s], sizeof(float) * 16);
		fb->shadowParams[0] = (float)m_impl->numShadowSlots;
		fb->shadowParams[1] = 0.f;
		fb->shadowParams[2] = 1.0f / (float)Impl::SHADOW_RES;
		fb->shadowParams[3] = 0.0015f;
		// Sky gradient for IBL (procedural-sky ambient + reflections in the world PS).
		for (int k = 0; k < 3; ++k) { fb->skyTop[k] = m_impl->sky.top[k]; fb->skyHorizon[k] = m_impl->sky.horizon[k]; fb->skyGround[k] = m_impl->sky.ground[k]; }
		fb->skyParams[0] = m_impl->sky.skyIntensity; fb->skyParams[1] = (m_impl->sky.mode == 1) ? 1.0f : 0.0f;
		fb->skyParams[2] = m_impl->hdr ? 0.0f : 1.0f;   // 1 => world.ps tonemaps inline (HDR off, no post tonemap); 0 => post does it
		fb->skyParams[3] = 0;
	}

	m_impl->DrawSky();   // procedural sky behind the scene (after clear, before geometry)
}

void NukeDiligent::setSky(const NukeSky& s) { m_impl->sky = s; }

// Desktop/Explorer file-drop -> editor import. One renderer instance, so a file-static callback is fine.
static bst::function<void(const char*)> g_onFileDrop;
static void GlfwDropCB(GLFWwindow*, int count, const char** paths)
{
	if (!g_onFileDrop) return;
	for (int i = 0; i < count; ++i) if (paths[i]) g_onFileDrop(paths[i]);
}
void NukeDiligent::setOnFileDrop(bst::function<void(const char*)> cb)
{
	g_onFileDrop = cb;
	if (m_window) glfwSetDropCallback(m_window, GlfwDropCB);
}

void NukeDiligent::setLights(const NukeLight* lights, int count)
{
	m_impl->lights.assign(lights ? lights : nullptr, (lights && count > 0) ? lights + count : nullptr);

	// Assign a shadow-map slot + world->light-clip to each shadow-casting dir/spot light (point = later).
	for (int i = 0; i < Impl::kMaxLights; ++i) m_impl->lightSlot[i] = -1;
	int slot = 0;
	for (int i = 0; i < count && slot < Impl::SHADOW_SLOTS; ++i)
	{
		const NukeLight& L = lights[i];
		if (!L.castShadows || L.type == 1) continue;   // 0 = directional, 2 = spot (point shadows: later)
		float3 d(L.dir[0], L.dir[1], L.dir[2]);
		float  len = length(d); if (len < 1e-4f) continue; d /= len;
		float3 P, F = d, U = ((d.y < 0.f ? -d.y : d.y) > 0.95f) ? float3(0, 0, 1) : float3(0, 1, 0);
		float4x4 proj;
		if (L.type == 0)   // directional: ortho centred on the origin
		{
			const float dist = 60.0f, extent = 60.0f;
			P = d * (-dist);
			proj = float4x4::Ortho(extent, extent, 0.1f, dist * 2.0f, false);
		}
		else               // spot: perspective from the light position, fov from the cone
		{
			P = float3(L.pos[0], L.pos[1], L.pos[2]);
			float fov = 2.0f * acosf(L.spotOuter < -1.f ? -1.f : (L.spotOuter > 1.f ? 1.f : L.spotOuter));
			if (fov < 0.2f) fov = 0.2f; if (fov > 3.0f) fov = 3.0f;
			float farZ = (L.range > 1.f) ? L.range : 50.0f;
			proj = float4x4::Projection(fov, 1.0f, 0.1f, farZ, false);
		}
		float3 R = normalize(cross(U, F)); U = cross(F, R);
		float4x4 view(
			R.x, U.x, F.x, 0.f,
			R.y, U.y, F.y, 0.f,
			R.z, U.z, F.z, 0.f,
			-dot(P, R), -dot(P, U), -dot(P, F), 1.f);
		m_impl->slotVP[slot] = view * proj;
		m_impl->lightSlot[i] = slot;
		++slot;
	}
	m_impl->numShadowSlots = m_impl->shadowPSO ? slot : 0;

	// Point lights: 6 perspective faces per cube (omnidirectional depth).
	for (int i = 0; i < Impl::kMaxLights; ++i) m_impl->lightCube[i] = -1;
	int cube = 0;
	static const float3 faceF[6] = { {1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1} };
	static const float3 faceU[6] = { {0,1,0}, {0,1,0}, {0,0,-1}, {0,0,1}, {0,1,0}, {0,1,0} };
	for (int i = 0; i < count && cube < Impl::MAX_POINT_SHADOWS; ++i)
	{
		const NukeLight& L = lights[i];
		if (!L.castShadows || L.type != 1) continue;   // point only
		float3 P(L.pos[0], L.pos[1], L.pos[2]);
		float  farZ = (L.range > 1.f) ? L.range : 50.0f;
		float4x4 proj = float4x4::Projection(1.5707963f, 1.0f, 0.1f, farZ, false);   // 90deg
		for (int f = 0; f < 6; ++f)
		{
			float3 F = faceF[f], U = faceU[f];
			float3 R = normalize(cross(U, F)); U = cross(F, R);
			float4x4 view(
				R.x, U.x, F.x, 0.f,
				R.y, U.y, F.y, 0.f,
				R.z, U.z, F.z, 0.f,
				-dot(P, R), -dot(P, U), -dot(P, F), 1.f);
			m_impl->cubeFaceVP[cube * 6 + f] = view * proj;
		}
		m_impl->lightCube[i] = cube;
		++cube;
	}
	m_impl->numCubes = (m_impl->shadowPSO && m_impl->shadowCubeTex) ? cube : 0;
}

int NukeDiligent::shadowPassCount() { return m_impl->numShadowSlots + m_impl->numCubes * 6; }

void NukeDiligent::beginShadowPass(int pass)
{
	if (pass < 0 || pass >= m_impl->numShadowSlots + m_impl->numCubes * 6) return;
	IDeviceContext* ctx = m_impl->context;
	ITextureView* dsv; int res;
	if (pass < m_impl->numShadowSlots)   // a 2D (directional/spot) shadow slice
	{
		m_impl->curShadowVP = m_impl->slotVP[pass];
		dsv = m_impl->shadowSliceDSV[pass]; res = Impl::SHADOW_RES;
	}
	else                                 // a point-light cube face
	{
		int c = pass - m_impl->numShadowSlots;
		m_impl->curShadowVP = m_impl->cubeFaceVP[c];
		dsv = m_impl->cubeFaceDSV[c]; res = Impl::SHADOW_RES / 2;
	}
	if (!dsv) return;
	ctx->SetRenderTargets(0, nullptr, dsv, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	ctx->ClearDepthStencil(dsv, CLEAR_DEPTH_FLAG, 1.f, 0, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	Viewport vp; vp.TopLeftX = 0; vp.TopLeftY = 0;
	vp.Width = (float)res; vp.Height = (float)res; vp.MinDepth = 0; vp.MaxDepth = 1;
	ctx->SetViewports(1, &vp, res, res);
}

void NukeDiligent::renderShadowObject(Mesh* mesh, const float pos[3], const float quat[4], const float scale[3], Material* mat)
{
	if (!m_impl->shadowPSO) return;
	Impl::MeshGPU* gp = m_impl->GetMeshGPU(mesh); if (!gp) return;
	Impl::MeshGPU& g = *gp;
	float4x4 world = float4x4::Scale(scale[0], scale[1], scale[2])
	               * Diligent::Quaternion<float>(quat[0], quat[1], quat[2], quat[3]).ToMatrix()
	               * float4x4::Translation(pos[0], pos[1], pos[2]);
	float4x4 wvp = world * m_impl->curShadowVP;

	ITextureView* base = (mat && mat->diff) ? m_impl->GetTexSRV(mat->diff) : nullptr;
	{
		MapHelper<float4x4> cb(m_impl->context, m_impl->shadowVSCB, MAP_WRITE, MAP_FLAG_DISCARD);
		*cb = wvp;
	}
	{
		MapHelper<float> cb(m_impl->context, m_impl->shadowPSCB, MAP_WRITE, MAP_FLAG_DISCARD);
		cb[0] = mat ? (float)mat->color.a : 1.0f;
		cb[1] = base ? 1.0f : 0.0f; cb[2] = 0.f; cb[3] = 0.f;
	}
	if (m_impl->shadowPsTexVar)
		m_impl->shadowPsTexVar->Set(base ? base : m_impl->whiteTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));

	IDeviceContext* ctx = m_impl->context;
	IBuffer* vbs[] = { g.pos, g.nrm, g.uv };
	Uint64   offs[] = { 0, 0, 0 };
	ctx->SetVertexBuffers(0, 3, vbs, offs, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, SET_VERTEX_BUFFERS_FLAG_RESET);
	ctx->SetPipelineState(m_impl->shadowPSO);
	ctx->CommitShaderResources(m_impl->shadowSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	DrawAttribs da{(Uint32)g.numVerts, DRAW_FLAG_VERIFY_STATES};
	ctx->Draw(da);
}

void NukeDiligent::endShadowPass() { /* the next beginCamera rebinds the camera targets */ }

void NukeDiligent::endCamera()
{
	// 1) Resolve the multisampled HDR color into the single-sample HDR texture (post-pass input).
	if (m_impl->curMSAA && m_impl->curResolveSrc && m_impl->curResolveDst)
	{
		ResolveTextureSubresourceAttribs ra;
		ra.SrcTextureTransitionMode = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
		ra.DstTextureTransitionMode = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
		ra.Format = m_impl->SceneFmt();
		m_impl->context->ResolveTextureSubresource(m_impl->curResolveSrc, m_impl->curResolveDst, ra);
	}
	// 2) Post-process: tonemap the HDR scene into the output (RT's post texture, or the backbuffer for target 0).
	if (m_impl->curPostSrc && m_impl->curPostDst)
		m_impl->RunPostPass(m_impl->curPostSrc, m_impl->curPostDst, m_impl->curRTW, m_impl->curRTH, m_impl->curTarget == 0);

	m_impl->curMSAA = false; m_impl->curResolveSrc = nullptr; m_impl->curResolveDst = nullptr;
	m_impl->curPostSrc = nullptr; m_impl->curPostDst = nullptr;
	m_impl->curTarget = 0;
}

void NukeDiligent::setMSAA(int s)
{
	int req = (s >= 8) ? 8 : (s >= 4) ? 4 : (s >= 2) ? 2 : 1;   // snap to a power of two (1 = off)
	if (!m_impl->device) { m_impl->samples = (Uint8)req; return; }   // before init: Setup will clamp + build
	Uint32 colorSC = (Uint32)m_impl->device->GetTextureFormatInfoExt(TEX_FORMAT_RGBA8_UNORM).SampleCounts;
	Uint32 depthSC = (Uint32)m_impl->device->GetTextureFormatInfoExt(TEX_FORMAT_D32_FLOAT).SampleCounts;
	while (req > 1 && !((colorSC & req) && (depthSC & req))) req >>= 1;
	// Defer the rebuild to the start of the next frame — doing it now (mid ImGui frame) would free the RT
	// textures the current frame's draw data still references (UI ImGui::Image), crashing renderDrawLists.
	m_impl->pendingSamples = req;
}

int NukeDiligent::getMSAA() { return m_impl->pendingSamples > 0 ? m_impl->pendingSamples : (int)m_impl->samples; }

void NukeDiligent::setHDR(bool on)
{
	if (!m_impl->device) { m_impl->hdr = on; return; }   // before init: Setup builds at this setting
	m_impl->pendingHDR = on ? 1 : 0;                       // applied at the start of the next frame (see render())
}
bool NukeDiligent::getHDR() { return m_impl->pendingHDR >= 0 ? (m_impl->pendingHDR != 0) : m_impl->hdr; }

void NukeDiligent::setHDROutput(bool on)
{
#ifdef _WIN32
	m_impl->hdrOutput = on;   // honoured at init (swap-chain format + HDR10 colour space)
#else
	(void)on;                 // no D3D11/DXGI off Windows — HDR output stays off
#endif
}
bool NukeDiligent::getHDROutput() { return m_impl->hdr10Active; }

void NukeDiligent::getViewProj(float* view16, float* proj16)
{
	if (view16) memcpy(view16, m_impl->curView.Data(), 16 * sizeof(float));
	if (proj16) memcpy(proj16, m_impl->curProj.Data(), 16 * sizeof(float));
}

void NukeDiligent::renderDrawLists(const NukeUIDrawData& data)
{
	if (!m_impl->uiPSO || data.listCount == 0) return;
	if (data.dispSize[0] <= 0.f || data.dispSize[1] <= 0.f) return;

	int totalVtx = 0, totalIdx = 0;
	for (int i = 0; i < data.listCount; ++i) { totalVtx += data.lists[i].vtxCount; totalIdx += data.lists[i].idxCount; }
	if (totalVtx == 0 || totalIdx == 0) return;

	IRenderDevice*  dev = m_impl->device;
	IDeviceContext* ctx = m_impl->context;

	if (!m_impl->uiVB || m_impl->uiVBSize < totalVtx)
	{
		m_impl->uiVB.Release();
		while (m_impl->uiVBSize < totalVtx) m_impl->uiVBSize = m_impl->uiVBSize ? m_impl->uiVBSize * 2 : 4096;
		BufferDesc bd;
		bd.Name = "UI VB"; bd.BindFlags = BIND_VERTEX_BUFFER; bd.Usage = USAGE_DYNAMIC; bd.CPUAccessFlags = CPU_ACCESS_WRITE;
		bd.Size = (Uint64)m_impl->uiVBSize * sizeof(NukeUIVert);
		dev->CreateBuffer(bd, nullptr, &m_impl->uiVB);
	}
	if (!m_impl->uiIB || m_impl->uiIBSize < totalIdx)
	{
		m_impl->uiIB.Release();
		while (m_impl->uiIBSize < totalIdx) m_impl->uiIBSize = m_impl->uiIBSize ? m_impl->uiIBSize * 2 : 8192;
		BufferDesc bd;
		bd.Name = "UI IB"; bd.BindFlags = BIND_INDEX_BUFFER; bd.Usage = USAGE_DYNAMIC; bd.CPUAccessFlags = CPU_ACCESS_WRITE;
		bd.Size = (Uint64)m_impl->uiIBSize * sizeof(uint16_t);
		dev->CreateBuffer(bd, nullptr, &m_impl->uiIB);
	}

	{
		MapHelper<NukeUIVert> vtx(ctx, m_impl->uiVB, MAP_WRITE, MAP_FLAG_DISCARD);
		MapHelper<uint16_t>   idx(ctx, m_impl->uiIB, MAP_WRITE, MAP_FLAG_DISCARD);
		if (!vtx || !idx) return;
		NukeUIVert* pv = vtx;
		uint16_t*   pi = idx;
		for (int i = 0; i < data.listCount; ++i)
		{
			const NukeUIDrawList& l = data.lists[i];
			std::memcpy(pv, l.vtx, (size_t)l.vtxCount * sizeof(NukeUIVert));
			std::memcpy(pi, l.idx, (size_t)l.idxCount * sizeof(uint16_t));
			pv += l.vtxCount;
			pi += l.idxCount;
		}
	}

	{
		float L = data.dispPos[0], R = data.dispPos[0] + data.dispSize[0];
		float T = data.dispPos[1], B = data.dispPos[1] + data.dispSize[1];
		float4x4 proj{
			2.f / (R - L), 0.f, 0.f, 0.f,
			0.f, 2.f / (T - B), 0.f, 0.f,
			0.f, 0.f, 0.5f, 0.f,
			(R + L) / (L - R), (T + B) / (B - T), 0.5f, 1.f};
		MapHelper<float4x4> cb(ctx, m_impl->uiCB, MAP_WRITE, MAP_FLAG_DISCARD);
		if (cb) *cb = proj;
	}

	// Target: an explicit RT (bindRenderTarget -> runtime UI into the viewport/camera RT) or the
	// backbuffer (editor UI). Bound here, no clear, so the UI composites over whatever's already there.
	ITextureView* uirtv = m_impl->uiRTV ? m_impl->uiRTV : m_impl->swapChain->GetCurrentBackBufferRTV();
	const Uint32 surfW = (m_impl->uiRTV && m_impl->uiTW) ? m_impl->uiTW : m_impl->swapChain->GetDesc().Width;
	const Uint32 surfH = (m_impl->uiRTV && m_impl->uiTH) ? m_impl->uiTH : m_impl->swapChain->GetDesc().Height;
	ctx->SetRenderTargets(1, &uirtv, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

	IBuffer* vbs[] = {m_impl->uiVB};
	ctx->SetVertexBuffers(0, 1, vbs, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, SET_VERTEX_BUFFERS_FLAG_RESET);
	ctx->SetIndexBuffer(m_impl->uiIB, 0, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	ctx->SetPipelineState(m_impl->uiPSO);
	const float bf[4] = {0, 0, 0, 0};
	ctx->SetBlendFactors(bf);
	Viewport vp; vp.TopLeftX = 0; vp.TopLeftY = 0; vp.Width = (float)surfW; vp.Height = (float)surfH; vp.MinDepth = 0; vp.MaxDepth = 1;
	ctx->SetViewports(1, &vp, surfW, surfH);

	Uint32 globalIdx = 0, globalVtx = 0;
	ITextureView* lastView = nullptr;
	for (int i = 0; i < data.listCount; ++i)
	{
		const NukeUIDrawList& l = data.lists[i];
		for (int c = 0; c < l.cmdCount; ++c)
		{
			const NukeUICmd& cmd = l.cmds[c];
			if (cmd.elemCount == 0) continue;
			Rect sc;
			sc.left   = std::max((Int32)cmd.clipRect[0], 0);
			sc.top    = std::max((Int32)cmd.clipRect[1], 0);
			sc.right  = std::min((Int32)cmd.clipRect[2], (Int32)surfW);
			sc.bottom = std::min((Int32)cmd.clipRect[3], (Int32)surfH);
			if (sc.right <= sc.left || sc.bottom <= sc.top) continue;
			ctx->SetScissorRects(1, &sc, surfW, surfH);

			ITextureView* view = reinterpret_cast<ITextureView*>(cmd.texId);
			if (view != lastView)
			{
				lastView = view;
				m_impl->uiTexVar->Set(view);
				ctx->CommitShaderResources(m_impl->uiSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
			}

			DrawIndexedAttribs da{cmd.elemCount, VT_UINT16, DRAW_FLAG_VERIFY_STATES};
			da.FirstIndexLocation = cmd.idxOffset + globalIdx;
			if (m_impl->baseVertexSupported)
				da.BaseVertex = cmd.vtxOffset + globalVtx;
			ctx->DrawIndexed(da);
		}
		globalIdx += l.idxCount;
		globalVtx += l.vtxCount;
	}
}

// ---- Render module export (boost::dll) ----
// Exported as the unmangled symbol "renderModule"; the engine's render-module
// loader looks this up to distinguish render modules from extension plugins.
class NukeDiligentModule : public NUKERenderModule
{
public:
	NukeDiligentModule()
	{
		std::strcpy(title, "Diligent Renderer");
		std::strcpy(description, "NukeEngine renderer backed by Diligent Engine (D3D11/D3D12).");
		std::strcpy(author, "NukeEngine");
		std::strcpy(version, "0.1.0");
		std::strcpy(id, "diligent");
	}
	iRender* CreateRenderer() override { return new NukeDiligent(); }
	void DestroyRenderer(iRender* r) override { delete r; }
};

extern "C" BOOST_SYMBOL_EXPORT NukeDiligentModule renderModule;
NukeDiligentModule renderModule;
