#pragma once
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
#include "EngineFactoryD3D12.h"   // D3D12 backend (ray tracing); chosen at launch via WindowDesc.backend
#include "EngineFactoryVk.h"      // Vulkan backend (task #138): editor default; shaders HLSL->SPIRV via glslang
#include "RenderDevice.h"
#include "DeviceContext.h"
#include "SwapChain.h"
#include "PipelineState.h"
#include "ShaderResourceBinding.h"
#include "Buffer.h"
#include "Texture.h"
#include "Shader.h"
#include "BottomLevelAS.h"   // ray tracing acceleration structures (D3D12)
#include "TopLevelAS.h"
#include "RefCntAutoPtr.hpp"
#include "MapHelper.hpp"
#include "BasicMath.hpp"
#include "GraphicsAccessories.hpp"
// --- Windows-only HDR10 display output (DXGI / D3D11). Other platforms: no D3D11, HDR output is a no-op. ---
#ifdef _WIN32
#include <d3d11.h>           // ID3D11View etc. (needed BEFORE the Diligent D3D11 interface headers below)
#include <d3d12.h>           // D3D12_CPU_DESCRIPTOR_HANDLE etc. (needed BEFORE the Diligent D3D12 interface headers)
#include <dxgi1_6.h>         // IDXGISwapChain3/4 + IDXGIOutput6 (HDR10 colour space + display detection)
#include "SwapChainD3D11.h"  // Diligent ISwapChainD3D11::GetDXGISwapChain (HDR10 swap-chain access)
#include "SwapChainD3D12.h"  // Diligent ISwapChainD3D12::GetDXGISwapChain (HDR10 on the D3D12 backend)
#endif

// Engine headers last (they do `using namespace std;` internally).
#include "NukeDiligent.h"
#include <interface/NUKEEInteface.h>   // NUKEModule — the renderer is an ordinary plugin now

#include <cstring>
#include <vector>
#include <map>
#include <mutex>
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

	// ---- CENTRALIZED GPU-resource lifetime manager --------------------------------------------------
	// THE rule: a GPU object whose raw pointer may still sit in CPU-side data (UI draw lists recorded
	// this frame, texId handles, cached SRVs/RTVs, SRBs, BLAS geometry refs) is NEVER Release()d
	// inline — it goes through Trash(). Trash parks a strong ref for kTrashFrames frames, so any stale
	// pointer still dereferences a LIVE object (and its address cannot be reused by a fresh allocation
	// while parked); Diligent then defers the underlying D3D12 memory to the frame fence. Inline
	// releases of objects like these were the random "device removed" (ACCESS_DENIED / page fault)
	// crashes with detached windows open. Purged once per render(); drained fully in deinit().
	static const uint64_t kTrashFrames = 4;   // > max frames in flight + recorded-but-not-yet-drawn UI window
	uint64_t frameId = 0;                     // advanced once at the top of render()
	std::vector<std::pair<RefCntAutoPtr<IObject>, uint64_t>> gpuTrash;
	std::mutex trashMutex;                    // create/destroy may arrive off the render thread
	void Trash(IObject* o);                   // null-safe: park an object until the GPU can't see it
	void PurgeTrash(bool everything = false); // frame tick (or full drain after IdleGPU)
	// True (and logs the removal reason once) if the D3D12 device has been removed. Guard every Present/
	// Flush with it so a device loss degrades to a skipped frame + a console reason, NOT a Diligent
	// debug-assert crash dialog (the assert fires INSIDE Present, too late to catch otherwise).
	bool  DeviceRemoved();
	void* d3d12DevCache = nullptr;   // cached ID3D12Device* (void* keeps <d3d12.h> out of this header)
	bool                          useD3D12 = false;   // active backend (set in init from WindowDesc.backend)
	bool                          useVulkan = false;  // backend == 2: Vulkan (task #138) — no DXGI anywhere

	// DISK shader-bytecode cache (OUR OWN, Vulkan only): a Vulkan boot re-runs glslang
	// over every HLSL shader (~3s). Key = FNV-1a of the full compile inputs (source,
	// entry, type, macros, flags); value = the compiled SPIR-V (IShader::GetBytecode),
	// one file per shader under config/shadercache_vk/. A hit feeds ByteCode straight to
	// CreateShader — glslang never runs; an edited shader changes the key and recompiles.
	void CreateShaderCached(const Diligent::ShaderCreateInfo& ci, Diligent::IShader** pp);
	void CreateGraphicsPipelineStateCached(const Diligent::GraphicsPipelineStateCreateInfo& ci, Diligent::IPipelineState** pp)
	{
		// Cache-miss shaders compile ASYNC on the worker pool — the PSO needs them ready,
		// so wait here: by this point the whole batch created earlier has been compiling
		// in parallel, and the bind sites never see a not-ready pipeline.
		auto wait = [](Diligent::IShader* s) { if (s) s->GetStatus(true); };
		wait(ci.pVS); wait(ci.pPS); wait(ci.pGS); wait(ci.pHS); wait(ci.pDS);
		device->CreateGraphicsPipelineState(ci, pp);
	}
	// Async-compiled cache misses: the SPIR-V is grabbed and written to disk once the
	// worker finishes (polled per frame — PollShaderSaves).
	std::vector<std::pair<Diligent::RefCntAutoPtr<Diligent::IShader>, std::string>> pendingShaderSaves;
	void PollShaderSaves();
	bool                          vsync    = true;    // main-present sync interval: true = 1 (vsync), false = 0 (uncapped)
	// DirectComposition objects for a TRANSPARENT window (per-pixel alpha to the desktop):
	// the composition swap chain presents into this visual tree. Stored as IUnknown* so the
	// widely-included Impl header stays free of <dcomp.h> (typed use lives in NukeDiligent.cpp).
	IUnknown*                     dcompDevice = nullptr;   // IDCompositionDevice
	IUnknown*                     dcompTarget = nullptr;   // IDCompositionTarget
	IUnknown*                     dcompVisual = nullptr;   // IDCompositionVisual
	bool                          transparent = false;     // transparent window: the empty background
	                                                       // clears to alpha 0 and the final pass outputs
	                                                       // PREMULTIPLIED alpha so the desktop shows through.
	bool                          rtSupported = false; // device reports ray-tracing capability (D3D12 + RT-capable GPU)
	RefCntAutoPtr<IShaderSourceInputStreamFactory> shaderFactory;   // resolves #include + loads RT shaders from the shaders dir

	std::vector<boost::function<void(void)>> onGUI;
	std::vector<boost::function<void(void)>> onRender;
	std::vector<boost::function<void()>>     onClose;

	// --- generic 2D (UI) draw-list renderer ---
	RefCntAutoPtr<IPipelineState>         uiPSO;
	// Per-texture SRB cache (MUTABLE var, set once): zero dynamic descriptors per
	// commit. LRU-purged by frame stamp; destroyTexture2D drops its entry eagerly.
	struct UISRBEntry { RefCntAutoPtr<IShaderResourceBinding> srb; uint64_t lastUse = 0; };
	std::unordered_map<ITextureView*, UISRBEntry> uiSRBCache;
	uint64_t uiFrame = 0;
	IShaderResourceBinding* UISRBFor(ITextureView* view);

	// Frame statistics (editor status bar): scene draws counted this frame, latched at
	// the start of the next one (getFrameStats reads the completed frame's numbers).
	int statDraws = 0, statTris = 0;
	int statDrawsOut = 0, statTrisOut = 0;
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
	void TrashRT(RT& rt);                    // park ALL of an RT's textures (before replacing it)

	// Per-size transient-target cache (scratch / bloom / RT-reflection output). Several DIFFERENT-sized
	// cameras render in one frame (viewport + camera preview + asset-editor previews) — a single shared
	// target that was Release()+recreated on every size change was a mid-frame lifetime race (the exact
	// intermittent device-removal class the G-buffer already got fixed for) AND an allocation storm.
	// Same pattern as GBufferSet: keyed by (w<<32|h), bounded LRU, evictions go through Trash().
	struct SizedTexSet { RefCntAutoPtr<ITexture> a, b; uint64_t lastUsed = 0; };
	std::unordered_map<uint64_t, SizedTexSet> scratchCache, bloomCache, rtOutCache;
	uint64_t sizedClock = 0;                 // shared LRU clock for the sized caches
	void EvictSized(std::unordered_map<uint64_t, SizedTexSet>& cache, uint64_t curKey);

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
	// Custom post-effect chain (one fullscreen pipeline per effect; ping-ponged in HDR before the final tonemap).
	struct PostPipe { RefCntAutoPtr<IPipelineState> pso; RefCntAutoPtr<IShaderResourceBinding> srb; IShaderResourceVariable* srcVar = nullptr; bool isBloom = false;
	                  IShaderResourceVariable* gbufVar = nullptr; IShaderResourceVariable* depthVar = nullptr; bool isSSR = false;
	                  IShaderResourceVariable* objIdVar = nullptr;   // musicvis: generic per-OBJECT id (gbuffer RT2)
	                  IShaderResourceVariable* histVar = nullptr; IShaderResourceVariable* velVar = nullptr; bool isTAA = false;   // temporal AA (history + depth + velocity)
	                  bool isRTRef = false;   // built-in ray-traced reflections (D3D12)
	                  IShaderResourceVariable* tlasVar = nullptr; IShaderResourceVariable* instVar = nullptr;
	                  IShaderResourceVariable* nrmVar = nullptr;  IShaderResourceVariable* rtProbeVar = nullptr;
	                  IShaderResourceVariable* uvVar = nullptr;   IShaderResourceVariable* matTexVar = nullptr; };
	std::unordered_map<uint64_t, PostPipe> postPipes;
	RefCntAutoPtr<IBuffer>                postParamsCB;   // shared PostParams (per-effect params, 256B)
	RefCntAutoPtr<IBuffer>                postFrameCB;    // shared PostFrame (resolution / time)
	RefCntAutoPtr<ITexture>               scratch[2];     // HDR ping-pong targets for the effect chain
	int                                   scratchW = 0, scratchH = 0;
	struct ChainStage { uint64_t pipeline; std::vector<float> params; };
	std::vector<ChainStage>               postChain;      // current camera's effect chain (copied in setPostChain)
	uint64_t CreatePostPipe(const std::string& name, const std::string& ps);
	void     EnsureScratch(int w, int h);
	// Format-safe texture transfer: equal formats -> CopyTexture; different -> fullscreen
	// blit (CopyTextureRegion between e.g. RGBA8 scene color and RGBA16F chain scratch is
	// an INVALID D3D12 call — it poisons the command list and Close() fails E_INVALIDARG).
	void     BlitTexture(ITextureView* srcSRV, ITexture* dstTex);
	std::map<TEXTURE_FORMAT, std::pair<RefCntAutoPtr<IPipelineState>, RefCntAutoPtr<IShaderResourceBinding>>> blitPipes;

	// Built-in bloom (multi-pass): bright-pass -> separable blur (half-res ping-pong) -> composite. Invoked
	// for a chain stage whose post shader is named "bloom".
	RefCntAutoPtr<IPipelineState>         bloomBrightPSO, bloomBlurPSO, bloomCompPSO;
	RefCntAutoPtr<IShaderResourceBinding> bloomBrightSRB, bloomBlurSRB, bloomCompSRB;
	IShaderResourceVariable*              bbSrc = nullptr;   // bright g_Source
	IShaderResourceVariable*              blSrc = nullptr;   // blur g_Source
	IShaderResourceVariable*              bcSrc = nullptr;   // comp g_Source (scene)
	IShaderResourceVariable*              bcBloom = nullptr; // comp g_Bloom
	RefCntAutoPtr<IBuffer>                bloomCB;
	RefCntAutoPtr<ITexture>               bloomTex[2];       // half-res blur ping-pong
	int                                   bloomW = 0, bloomH = 0;
	void EnsureBloom(int w, int h);
	void RunBloom(ITextureView* srcSRV, ITextureView* dstRTV, int w, int h, float threshold, float intensity);

	// --- Reflection probe: scene-captured HDR cubemaps ----------------------------------------------
	struct CubeRT
	{
		RefCntAutoPtr<ITexture>     color;      // RGBA16F cube (6 faces, full mip chain), GenerateMips for rough refl
		RefCntAutoPtr<ITexture>     depth;      // shared D32 per face
		RefCntAutoPtr<ITextureView> faceRTV[6]; // mip-0 RTV per face (CreateView-owned)
		RefCntAutoPtr<ITextureView> dsv;
		ITextureView*               srv = nullptr;   // cube SRV (default view, texture-owned)
		int res = 0, mips = 1;
		bool fmtHdr = true;   // which SceneFmt the cube was built with (rebuild on HDR toggle to match world PSO)
		// MSAA capture intermediates: sky/world PSOs are built at the CURRENT sample count —
		// rendering them straight into the single-sample cube face is a Vulkan render-pass
		// incompatibility (DEVICE_LOST; D3D12 silently tolerated it). Faces render here and
		// resolve into the cube slice per face.
		RefCntAutoPtr<ITexture> msColor, msDepth;
		int msSamples = 1;    // sample count the intermediates were built with (rebuild on change)
	};
	std::unordered_map<uint64_t, CubeRT> cubes;
	void BuildCube(CubeRT& c, int res);   // (re)create the cube GPU resources at the current SceneFmt()
	RefCntAutoPtr<ITexture>              fallbackCube;     // 1x1 cube bound to g_Probe when no probe is active
	ITextureView*                        fallbackCubeSRV = nullptr;
	RefCntAutoPtr<ISampler>              probeSampler;     // linear-clamp; attached to probe/fallback cube SRVs

	// G-buffer prepass (single-sample) for screen-space reflections: normal(oct)+roughness+metalness + depth.
	// Rendered per SSR camera before the colour pass; the "ssr" post effect samples it. Own 1x depth -> no MSAA
	// depth resolve, and the colour/custom shaders stay untouched (a dedicated gbuffer.ps fills it).
	RefCntAutoPtr<ITexture>             gbufColor, gbufDepth, gbufVel, gbufObjId;
	ITextureView*                       gbufRTV = nullptr, *gbufDSV = nullptr, *gbufSRV = nullptr, *gbufDepthSRV = nullptr;
	ITextureView*                       gbufVelRTV = nullptr, *gbufVelSRV = nullptr;   // screen-space motion (TAA)
	ITextureView*                       gbufObjIdRTV = nullptr, *gbufObjIdSRV = nullptr; // generic per-OBJECT id (pivot hash)
	int                                 gbufW = 0, gbufH = 0;
	// G-buffers are cached PER SIZE. The editor renders the main scene and a small
	// selected-camera preview (different sizes) in the SAME frame; a single shared buffer
	// used to be Release()+recreated twice every frame — a per-frame lifetime race that
	// intermittently removed the device (GPU still reading the buffer we just freed). Now
	// each size keeps its own persistent set and the live gbuf* members above merely point
	// at the active one. Bounded LRU so drag-resize (many transient sizes) can't grow
	// unbounded; eviction just drops the refs and lets Diligent's deferred release free the
	// memory after the frame fence (safe — the active set is never evicted).
	struct GBufferSet {
		RefCntAutoPtr<ITexture> color, depth, vel, objId;
		ITextureView *rtv = nullptr, *dsv = nullptr, *srv = nullptr, *depthSRV = nullptr,
		             *velRTV = nullptr, *velSRV = nullptr, *objIdRTV = nullptr, *objIdSRV = nullptr;
		uint64_t lastUsed = 0;
	};
	std::unordered_map<uint64_t, GBufferSet> gbufCache;
	uint64_t                            gbufFrameCtr = 0;   // LRU clock
	uint64_t                            gbufCurKey = 0;     // active set's key (never evicted)
	void EvictGBufferCache();
	bool                                gbufActive = false;   // a valid prepass ran for the current camera
	RefCntAutoPtr<IPipelineState>       gbufPSO;
	RefCntAutoPtr<IShaderResourceBinding> gbufSRB;
	IShaderResourceVariable*            gbufMRVar = nullptr;   // PS g_MetalRough (dynamic)
	IShaderResourceVariable*            gbufNrmVar = nullptr;  // PS g_Normal (dynamic) — normal-mapped gbuffer normal
	RefCntAutoPtr<IBuffer>              ssrCB;                 // SSR matrices (view/proj/invProj/res)
	RefCntAutoPtr<IBuffer>              rtRefCB;               // RT reflections (invViewProj, light, ambient, sky)
	// Temporal AA: per-camera history + previous view/proj (keyed by curTarget), a shared CB, and the current
	// frame's sub-pixel jitter (pixels; applied to the COLOUR projection only, in beginCamera).
	RefCntAutoPtr<IBuffer>              taaCB;
	struct TAAState { RefCntAutoPtr<ITexture> hist; float4x4 prevView, prevProj; bool valid = false; int w = 0, h = 0; };
	std::map<uint64_t, TAAState>        taaStates;
	bool                                curTAA = false;        // is the current camera running TAA?
	float                               curJitterX = 0.0f, curJitterY = 0.0f;   // this frame's jitter (pixels, [-0.5,0.5])
	int                                 taaFrame = 0;
	float4x4                            curProjNoJitter;       // curProj before jitter (for TAA reprojection)
	void RunTAA(PostPipe& pp, ITextureView* srcSRV, ITexture* dstTex, int w, int h, const std::vector<float>& params);

	// --- Ray tracing (D3D12) ---------------------------------------------------------------------------------
	std::unordered_map<Mesh*, RefCntAutoPtr<IBottomLevelAS>> blasCache;   // BLAS per mesh (built once, reused)
	RefCntAutoPtr<ITopLevelAS>         tlas;                  // scene TLAS (rebuilt per frame)
	RefCntAutoPtr<IBuffer>             tlasScratch, tlasInstanceBuf;
	Uint32                             tlasMaxInstances = 0;
	size_t                             lastTlasSig = 0;        // topology signature (count + BLAS set) -> refit when unchanged
	uint32_t                           tlasFrameCtr = 0;       // periodic full rebuild counter (refit hygiene)
	std::vector<TLASBuildInstanceData> rtInstances;           // accumulated between beginRTScene/buildRTScene
	std::vector<std::string>           rtInstanceNames;        // stable storage backing TLASBuildInstanceData::InstanceName
	bool                               rtSceneReady = false;   // a valid TLAS is built for the current frame
	RefCntAutoPtr<ITopLevelAS>         fallbackTLAS;           // empty TLAS bound to g_TLAS when no scene TLAS (rays miss)
	RefCntAutoPtr<IBuffer>             fbTlasScratch, fbTlasInst;
	IBottomLevelAS* GetMeshBLAS(Mesh* mesh);                   // get-or-build the BLAS for a mesh (from its pos buffer)
	void EnsureRTFallback();                                   // build the empty fallback TLAS once

	// RT reflection hit shading: per-instance geometry (normals) + material, so a ray hit can be shaded.
	// Normals of every referenced mesh are concatenated into one raw buffer; each instance stores its byte offset.
	// matByteOffset = this instance's MatCB block in g_MatBytes (auto-gen hit shaders load their params from it).
	// Mirrors HLSL RTInstanceData (rt_common.hlsl) byte-for-byte. 16-byte aligned rows for StructuredBuffer.
	struct RTInstanceData {
		uint32_t nrmOffset, uvOffset, posOffset, matByteOffset;
		uint32_t texIndex, nrmTexIndex, mrTexIndex, aoTexIndex;
		uint32_t emTexIndex, specTexIndex; float specularFactor; uint32_t nrmFlipG;   // 1 = flip green (OpenGL)
		float albedoMetal[4]; float emissiveRough[4];
	};
	std::unordered_map<Mesh*, uint32_t> meshNrmByteOffset;     // mesh -> byte offset of its normals in rtNrmBuf
	std::unordered_map<Mesh*, uint32_t> meshUVByteOffset;      // mesh -> byte offset of its uvs in rtUVBuf
	std::unordered_map<Mesh*, uint32_t> meshPosByteOffset;     // mesh -> byte offset of its positions in rtPosBuf
	std::vector<float>                  allNrmCPU, allUVCPU, allPosCPU;   // concatenated normals / uvs / positions
	bool                                allNrmDirty = false;
	RefCntAutoPtr<IBuffer>              rtNrmBuf;     IBufferView* rtNrmSRV  = nullptr;   // ByteAddressBuffer (all normals)
	RefCntAutoPtr<IBuffer>              rtUVBuf;      IBufferView* rtUVSRV   = nullptr;   // ByteAddressBuffer (all uvs)
	RefCntAutoPtr<IBuffer>              rtPosBuf;     IBufferView* rtPosSRV  = nullptr;   // ByteAddressBuffer (all positions)
	RefCntAutoPtr<IBuffer>              rtInstBuf;    IBufferView* rtInstSRV = nullptr;   // StructuredBuffer<RTInstanceData>
	std::vector<RTInstanceData>         rtInstData;            // parallel to rtInstances, rebuilt per frame
	uint32_t                            rtInstCapacity = 0;
	static const uint32_t               kMaxMatTex = 256;      // bindless material maps (albedo/normal/MR/AO/emissive/spec)
	std::unordered_map<Texture*, uint32_t> matTexSlot;         // engine texture -> slot in the bindless array
	std::vector<ITextureView*>          matTexSRVs;            // unique material map SRVs (<= kMaxMatTex)
	std::vector<Texture*>               matTexPtr;             // engine texture per slot (re-resolve SRV each frame -> animation)

	// --- RT reflection PIPELINE (real DXR: ray-gen + miss + closest-hit + SBT, native recursion) -------------
	RefCntAutoPtr<IPipelineState>         rtPSO;               // ray-tracing PSO (rt_rgen/rt_rmiss/rt_rchit)
	RefCntAutoPtr<IShaderResourceBinding> rtSRB;               // dynamic resources (TLAS, gbuffer, bindless, output)
	RefCntAutoPtr<IShaderBindingTable>    rtSBT;               // shader binding table (ray-gen + miss + hit group)
	RefCntAutoPtr<ITexture>               rtOutTex;            // UAV the ray-gen writes the composited reflection into
	int                                   rtOutW = 0, rtOutH = 0;
	// Global RTX reflection quality (Project Settings -> config/main.json, pushed via setRTReflection).
	float rtCfgIntensity = 1.0f; float rtCfgMaxDist = 100.0f; int rtCfgBounces = 3; float rtCfgRoughCut = 0.6f;
	bool BuildRTPipeline();                                    // build rtPSO/rtSRB/rtSBT (needs shaderFactory + DXC)
	void EnsureRTOutput(int w, int h);                         // (re)create the RGBA16F UAV output at viewport size
	void RunRTReflectPipeline(ITextureView* srcSRV, ITexture* dstTex, int w, int h, const std::vector<float>& params);
	// Auto-generated per-shader RT closest-hits. A material shader with a "<name>.surf.hlsl" gets its own hit
	// group, built by GenChitSource() from the shader's MatCB schema + that surface file (no hand-written .rchit).
	std::unordered_map<std::string, std::string> rtSurfShaders;  // shader name -> its PS source (has the MatCB schema)
	std::unordered_map<std::string, std::string> shaderHitGroup; // shader name -> hit-group name in the RT PSO
	std::string GenChitSource(const std::string& name, const std::string& psSource);  // codegen the closest-hit HLSL
	bool rtPipelineDirty = false;                              // a new surf shader appeared -> rebuild rtPSO
	std::vector<std::string> rtInstShaderGuid;                 // per-instance material shader name (-> hit group), parallel to rtInstances
	std::vector<uint8_t>     allMatCPU;                        // concatenated per-instance MatCB blocks (kMatBlock each)
	static const uint32_t    kMatBlock = 256;                  // per-instance material byte block (matches MatCB capacity)
	RefCntAutoPtr<IBuffer>   rtMatBuf;  IBufferView* rtMatSRV = nullptr;  // g_MatBytes (per-instance MatCB blocks)
	uint32_t                 rtMatCapacity = 0;
	void EnsureGBuffer(int w, int h);
	bool BuildGBufferPipe();
	void SetCameraViewProj(const NukeCameraDesc& cam, int w, int h);   // curView/curProj/curCamPos (shared: camera + gbuffer)
	bool CameraSize(const NukeCameraDesc& cam, int& w, int& h);
	void RunSSR(PostPipe& pp, ITextureView* srcSRV, ITextureView* dstRTV, int w, int h, const std::vector<float>& params);
	bool        probeActive = false;     // a probe cube is bound (and not currently capturing)
	ITextureView* probeCubeSRV = nullptr;
	float       probePos[3] = {0,0,0};
	float       probeIntensity = 1.0f;
	float       probeMaxMip = 0.0f;
	float       probeBoxHalf[3] = {0,0,0};   // parallax box half-extents (0 = no parallax correction)
	bool                                  hdrOutput = false;   // requested HDR10 display output (Player only, before init)
	bool                                  hdr10Active = false; // an HDR10 swap chain is actually live (display is HDR)
	float                                 hdrPaperWhite = 200.0f;   // diffuse-white nits for the HDR10 encode
	float                                 hdrPeak = 1000.0f;        // highlight peak nits
	float                                 toneExposure = 1.0f;      // SDR tonemap exposure multiplier
	float                                 toneWhite = 1.0f;         // SDR tonemap white point (linear value mapped to pure white)
	void CreatePostResources();
	void RunPostPass(ITextureView* hdrSRV, ITextureView* dstRTV, int w, int h, bool toBackbuffer);
	void SetupHDROutput();   // after swap-chain creation: set the HDR10 colour space if the display supports it
	static constexpr TEXTURE_FORMAT HDR_FMT = TEX_FORMAT_RGBA16_FLOAT;
	bool hdr = true;                   // HDR pipeline on (scene = RGBA16F, post tonemaps) / off (RGBA8, world.ps tonemaps)
	int  pendingHDR = -1;              // requested hdr (0/1); applied with pendingSamples at the start of render()
	TEXTURE_FORMAT SceneFmt() const { return hdr ? HDR_FMT : TEX_FORMAT_RGBA8_UNORM; }
	bool wireframe = false;            // scene fill mode: renderObject picks WorldPipe::psoWire when set

	// --- 3D world pipelines (one per shader; all share the layout + CBs + white fallback) ---
	struct WorldPipe
	{
		RefCntAutoPtr<IPipelineState>         pso;        // opaque (blend off, depth write on)
		RefCntAutoPtr<IPipelineState>         psoBlend;   // transparent: alpha blend, depth test on / write off
		RefCntAutoPtr<IPipelineState>         psoAdd;     // additive: add blend, depth write off
		RefCntAutoPtr<IPipelineState>         psoWire;    // wireframe fill (scene draw-mode toggle)
		RefCntAutoPtr<IShaderResourceBinding> srb;
		IShaderResourceVariable*              texVar  = nullptr;  // PS "g_Tex"        (base color, dynamic)
		IShaderResourceVariable*              normVar = nullptr;  // PS "g_Normal"     (normal map, dynamic)
		IShaderResourceVariable*              mrVar   = nullptr;  // PS "g_MetalRough" (dynamic)
		IShaderResourceVariable*              aoVar   = nullptr;  // PS "g_Occlusion"  (dynamic)
		IShaderResourceVariable*              emVar   = nullptr;  // PS "g_Emissive"   (dynamic)
		IShaderResourceVariable*              specVar = nullptr;  // PS "g_Spec"       (specular map, dynamic)
		IShaderResourceVariable*              shadowVar = nullptr;// PS "g_Shadow"      (dynamic)
		IShaderResourceVariable*              cubeVar   = nullptr;// PS "g_ShadowCube" (dynamic)
		IShaderResourceVariable*              probeVar  = nullptr;// PS "g_Probe" (reflection cubemap, dynamic)
		IShaderResourceVariable*              tlasVar   = nullptr;// PS "g_TLAS" (ray-tracing accel struct, RT builds only)
		// INSTANCED variants (7.1): built only when the shader source handles NUKE_INSTANCED
		// (the built-in world shader does; custom shaders opt in). Same blend variants; the
		// instanced SRB has its OWN variable set (a different compiled shader pair).
		RefCntAutoPtr<IPipelineState>         psoInst, psoInstBlend, psoInstAdd, psoInstWire;
		RefCntAutoPtr<IShaderResourceBinding> srbInst;
		IShaderResourceVariable *texVarI = nullptr, *normVarI = nullptr, *mrVarI = nullptr, *aoVarI = nullptr,
		                        *emVarI = nullptr, *specVarI = nullptr, *shadowVarI = nullptr, *cubeVarI = nullptr,
		                        *probeVarI = nullptr, *tlasVarI = nullptr;
		std::string vsSrc, psSrc, dbg;   // kept so the pipeline can be rebuilt (e.g. on an MSAA change)
	};
	std::unordered_map<uint64_t, WorldPipe> worldPipes;   // shader handle -> pipeline
	uint64_t                              defaultWorldHandle = 0;   // builtin "world" pipeline

	// --- GPU instancing (7.1) -----------------------------------------------------------
	// Persistent instance buffers (engine handle -> dynamic VB of NukeInstanceData records).
	struct InstBuf { RefCntAutoPtr<IBuffer> buf; int capacity = 0; int count = 0; };
	std::unordered_map<uint64_t, InstBuf> instBufs;
	uint64_t nextInstBuf = 1;
	// Instanced twins of the shadow / g-buffer pipelines (same CBs, instanced input layout).
	RefCntAutoPtr<IPipelineState>         shadowPSOInst;
	RefCntAutoPtr<IShaderResourceBinding> shadowSRBInst;
	IShaderResourceVariable*              shadowPsTexVarInst = nullptr;
	RefCntAutoPtr<IPipelineState>         gbufPSOInst;
	RefCntAutoPtr<IShaderResourceBinding> gbufSRBInst;
	IShaderResourceVariable*              gbufMRVarInst = nullptr;
	IShaderResourceVariable*              gbufNrmVarInst = nullptr;
	bool warnedNoInstPipe = false;   // one-shot log: material shader without an instanced variant
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
	                     float skyTop[4]; float skyHorizon[4]; float skyGround[4]; float skyParams[4];      // IBL
	                     float probePos[4]; float probeParams[4]; float probeBox[4]; };   // probe: pos.xyz+active, intensity+maxMip, boxHalf.xyz+valid
	void WriteFrameCB(const Diligent::float3& P);   // fill worldFrameCB (lights/shadows/sky/probe) — shared by camera + cube-face passes
	float                                 curCamPos[3] = {0, 0, 0};  // set in beginCamera (PBR view dir)
	uint64_t                              curTarget = 0;             // RT id bound by beginCamera (feedback guard)
	// True between beginCamera binding its targets and the END of endCamera. Sprites (world +
	// canvas-pre) REQUIRE the camera's colour+depth targets — a sprite emitted/flushed outside a
	// camera pass would draw into whatever is bound (e.g. a depth-less UI target: "Sprite PSO
	// D32_FLOAT vs DSV nullptr" spam) — such calls are dropped instead.
	bool                                  cameraPassActive = false;
	std::vector<NukeLight>                lights;      // scene lights (setLights); empty -> default sun

	// --- Shadow maps (directional + spot share a 2D array; one slice per shadow-casting light) -----
	int                                   shadowRes    = 2048;   // global, World-Settings-driven (rebuilds on change)
	int                                   pendingShadowRes = 0;  // requested resolution; applied at render() top
	float                                 shadowDistance = 60.0f; // directional ortho extent / range
	float                                 shadowDepthBias = 0.0015f;
	float                                 shadowNormalBias = 0.0f; // world-units offset along N at sample time
	float                                 shadowSoftness = 1.0f;   // PCF step multiplier
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
	// Debug/gizmo lines (iRender::drawDebugLine): per-frame vertex list (pos3+col4),
	// guarded (game + fixed threads emit), drawn depth-tested at endCamera, cleared at
	// the top of render().
	// Drawn AFTER the post chain (LDR, no depth): TAA has no velocity for lines (they
	// smear/vanish while the camera moves) and the RT-reflection composite overwrites
	// them on reflective surfaces - post-last avoids both. Two PSOs: RT (RGBA8) targets
	// and the swap-chain backbuffer format.
	RefCntAutoPtr<IPipelineState> debugPSO;      // -> RT post targets (RGBA8)
	RefCntAutoPtr<IPipelineState> debugPSOBB;    // -> the backbuffer (swap-chain format)
	RefCntAutoPtr<IShaderResourceBinding> debugSRB;
	RefCntAutoPtr<IShaderResourceBinding> debugSRBBB;
	RefCntAutoPtr<IBuffer> debugCB;
	RefCntAutoPtr<IBuffer> debugVB;
	int debugVBSize = 0;                 // capacity, in vertices
	std::vector<float> debugVerts;       // 7 floats per vertex
	std::mutex debugMutex;
	void CreateDebugResources();
	void DrawDebugLines(bool toBackbuffer);
	// DEPTH-TESTED lines (drawDebugLineDepth): drawn INSIDE the camera pass while the MS scene
	// targets are still bound (endCamera top), so geometry occludes them — quiet gizmos like
	// unselected canvas frames. PSO depends on samples/HDR -> built lazily against the current
	// SceneFmt()/samples and rebuilt when they change. Batch is CONSUMED per camera (cleared
	// after the draw) so one camera's gizmos never leak into another camera's pass.
	RefCntAutoPtr<IPipelineState>         debugDepthPSO;
	RefCntAutoPtr<IShaderResourceBinding> debugDepthSRB;
	int            debugDepthSamples = 0;                        // PSO built for this sample count
	TEXTURE_FORMAT debugDepthFmt     = TEX_FORMAT_UNKNOWN;       // ...and this scene format
	std::vector<float> debugVertsDepth;   // 7 floats per vertex (shares debugMutex + debugVB)
	void DrawDepthDebugLines();

	// Sprites (iRender::drawSprite): unlit textured quads, alpha-blended, drawn IN the camera pass
	// (SceneFmt + MSAA + depth-test, no depth write). PSO depends on samples/HDR -> rebuilt with them.
	RefCntAutoPtr<IPipelineState>         spritePSO;
	RefCntAutoPtr<IShaderResourceBinding> spriteSRB;
	IShaderResourceVariable*              spriteTexVar = nullptr;   // PS g_Sprite (dynamic)
	RefCntAutoPtr<IBuffer>                spriteCB;                 // view*proj
	RefCntAutoPtr<IBuffer>                spriteVB;                 // dynamic (grows), 9 floats/vertex
	int                                   spriteVBSize = 0;         // VB capacity in vertices
	// Batching: drawSprite ACCUMULATES quads and flushes ONE draw per texture run (sprites arrive
	// pre-sorted back-to-front, so consecutive same-texture sprites merge). Flushed on texture
	// change and at endCamera (before the MSAA resolve).
	Texture*                              spriteBatchTex = nullptr;
	std::vector<float>                    spriteBatchVerts;
	void CreateSpriteResources();
	void FlushSprites();

	// LIT sprite runs (drawSpriteRunLit — tilemap layers with a normal map): same batch layout,
	// Lambert lighting from worldFrameCB, per-batch plane TBN in spriteLitCB. SRBs are cached
	// per (diffuse, normal) SRV pair. Flushed on pair change / kind switch / endCamera.
	RefCntAutoPtr<IPipelineState>         spriteLitPSO;
	RefCntAutoPtr<IBuffer>                spriteLitCB;              // float4 T,B,N (N.w = green flip)
	std::map<std::pair<ITextureView*, ITextureView*>, RefCntAutoPtr<IShaderResourceBinding>> spriteLitSRBs;
	Texture*                              spriteLitTex = nullptr;
	Texture*                              spriteLitNormal = nullptr;
	bool                                  spriteLitFlipY = true;
	std::vector<float>                    spriteLitVerts;
	void FlushSpritesLit();

	// Screen-space (Canvas HUD) sprites — verts already in NDC, identity transform. Two queues:
	// PRE = drawn with the scene before post (reuses spritePSO; NDC z=0 => always on top); POST =
	// drawn on the final image after post (own output-format PSO, single-sample, no depth). Both defer
	// their draw to a flush point, so they store per-texture RUNS (texture + vertex count) into one
	// vertex list that the flush replays.
	struct SprRun { Texture* tex; int count; };
	std::vector<float>   spriteScrPreVerts;   std::vector<SprRun> spriteScrPreRuns;
	std::vector<float>   spriteScrPostVerts;  std::vector<SprRun> spriteScrPostRuns;
	RefCntAutoPtr<IPipelineState>         spriteScreenPSO, spriteScreenPSOBB;      // after-post: RT / backbuffer format
	RefCntAutoPtr<IShaderResourceBinding> spriteScreenSRB, spriteScreenSRBBB;
	IShaderResourceVariable*              spriteScreenTexVar = nullptr, *spriteScreenTexVarBB = nullptr;
	void AppendScreenSprite(std::vector<float>& verts, std::vector<SprRun>& runs, Texture* tex,
	                        const float rect[4], const float refSize[2], const float uv[4], const float tint[4],
	                        int scaleMode = 0);   // 0 Fit / 1 Stretch / 2 Expand / 3 FitWidth / 4 FitHeight
	void FlushScreen(std::vector<float>& verts, std::vector<SprRun>& runs, IPipelineState* pso,
	                 IShaderResourceBinding* srb, IShaderResourceVariable* texVar);
	void FlushScreenPre();                    // at endCamera, before the MSAA resolve (into the scene target)
	void FlushScreenPost(bool toBackbuffer);  // after post, on the final output

	// Screen-space decals (iRender::drawDecal): a box volume, surface reconstructed from the gbuf depth,
	// texture projected along the box +Z. Albedo = alpha blend, LightProjector = additive. Unit-cube VB.
	RefCntAutoPtr<IPipelineState>         decalPSO, decalPSOAdd;
	RefCntAutoPtr<IShaderResourceBinding> decalSRB, decalSRBAdd;
	IShaderResourceVariable*              decalTexVar = nullptr, *decalDepthVar = nullptr;
	IShaderResourceVariable*              decalTexVarAdd = nullptr, *decalDepthVarAdd = nullptr;
	RefCntAutoPtr<IBuffer>                decalCB, decalVB;
	void CreateDecalResources();

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
	struct MeshGPU { RefCntAutoPtr<IBuffer> pos, nrm, uv; int numVerts = 0; int version = 0; };
	std::unordered_map<Mesh*, MeshGPU>          meshCache;
	MeshGPU* GetMeshGPU(Mesh* mesh);   // get-or-build the GPU vertex buffers (pos/nrm/uv) for a mesh;
	                                   // re-uploads in place when Mesh::version changed (skinned/procedural)
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
	ITextureView*                         curDSV = nullptr;   // ...and its depth: passes that bind their own
	                                                          // targets (outline) must RESTORE both — later
	                                                          // draws (gizmos/sprites) expect the camera depth
	int                                   curRTW = 0, curRTH = 0;
	ITextureView*                         uiRTV = nullptr;    // explicit 2D target (bindRenderTarget); null = backbuffer
	Uint32                                uiTW = 0, uiTH = 0; // its size (0 = use swapchain)
	void BuildOutlinePipelines();
	void EnsureOutlineMask(int w, int h);

	// UI multi-viewport: one swap chain per detached OS window (keyed by native handle,
	// HWND on Windows). Swap-chain CREATION/RESIZE/DESTRUCTION never happens mid-frame —
	// uiViewportRender/Destroy only QUEUE the request and render() applies them at the top
	// of the next frame, when nothing is recorded (creating or resizing a DXGI swap chain
	// between passes under heavy GPU load intermittently wedged the queue: endless "Timeout
	// elapsed while waiting for the frame waitable object"). Destruction parks the swap
	// chain in the GPU trash instead of a mid-frame IdleGPU.
	std::map<void*, RefCntAutoPtr<ISwapChain>> uiVpSC;
	std::map<void*, std::pair<int, int>>       uiVpPending;   // create/resize requests (handle -> size)
	// Resize DEBOUNCE per window: (last requested size, consecutive frames it held). A live
	// drag re-requests a new size EVERY frame; resizing (Flush+IdleGPU+Resize) 30+ times/sec
	// starves the main swap chain's frame-latency waitable object — permanent "Timeout
	// elapsed waiting for the frame waitable object" stalls. Resize only once the size
	// settles; meanwhile the old buffers present stretched (visually fine mid-drag).
	std::map<void*, std::pair<std::pair<int, int>, int>> uiVpStable;
	std::map<void*, int> uiVpCooldown;   // frames to skip a window after a FAILED chain creation
	std::map<void*, int> uiVpGrace;      // frames to skip draw+present right after a resize (diag)
	// Secondary presents are DEFERRED to after the MAIN Present: presenting (and its
	// internal Flush) mid-frame splits the command stream between the preview RT's write
	// and its SRV sampling — the D3D12 debug layer then kills the device with
	// ACCESS_DENIED ("rendering to a texture with read access") on RT-sampling windows.
	std::vector<void*> vpPresentQueue;
	size_t vpPresentRR = 0;   // round-robin cursor: ONE secondary present per frame (see render())
	uint64_t uiVpFrameNo = 0; // frame counter for the multi-window draw interleave (uiViewportRender)

	// GDI-BLIT host windows (task #137): a detached window = offscreen RT + staging ring +
	// SetDIBitsToDevice. ZERO DXGI objects per window — no secondary swap chains, resizes
	// or presents, so the month-long "secondary present vs heavy frame" ACCESS_DENIED
	// device removal cannot exist BY CONSTRUCTION. Readback is async (ring of 3, mapped
	// with DO_NOT_WAIT after the main present): ~2 frames of latency, invisible for tool
	// windows, and the main frame does no extra flushes at all.
	struct HostBlit
	{
		Diligent::RefCntAutoPtr<Diligent::ITexture> rt;          // offscreen UI render target
		Diligent::RefCntAutoPtr<Diligent::ITexture> staging[3];  // readback ring
		int  w = 0, h = 0;
		int  cur = 0;                  // ring slot written THIS frame
		bool valid[3] = {};            // slot holds a issued copy
		std::vector<uint8_t> scratch;  // BGRX rows for GDI
	};
	std::map<void*, HostBlit> uiHostBlits;
	std::vector<void*> uiHostBlitQueue;   // windows to blit AFTER the main Present
	void BlitHostWindows();               // map ready staging + SetDIBitsToDevice
	// VULKAN native viewports: per-window SWAPCHAIN render (imgui multi-viewport).
	// The Vulkan WSI has none of the DXGI create/resize/present races — this is the
	// normal multi-window path; GDI blit stays the D3D fallback.
	void ViewportRenderSwapchain(void* nativeHandle, int w, int h, const NukeUIDrawData& data);
	void ApplyPendingViewportOps();                            // render() top: create/resize queued swap chains
	// Shared UI draw body (renderDrawLists + secondary viewports draw with it).
	void DrawUILists(ITextureView* rtv, Uint32 surfW, Uint32 surfH, const NukeUIDrawData& data);

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
