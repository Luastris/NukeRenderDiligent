#pragma once
// Include order matters: Windows-pulling headers (GLFW native + Diligent D3D) MUST come
// before the engine headers, which do `using namespace std;` (std::byte vs ::byte clash).

#include <cmath>

#ifdef _WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <shellapi.h>   // ExtractIconEx
#include <dwmapi.h>     // DwmSetWindowAttribute
#ifndef DWMWA_USE_IMMERSIVE_DARK_MODE
#define DWMWA_USE_IMMERSIVE_DARK_MODE 20   // Win10 2004+ (build 19041+)
#endif

#include "EngineFactoryD3D11.h"
#include "EngineFactoryD3D12.h"   // D3D12 backend (ray tracing)
#elif defined(__APPLE__)
#include <GLFW/glfw3.h>           // native handles come through the Cocoa shim, not glfw3native.h
#else
#define GLFW_EXPOSE_NATIVE_X11
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>     // glfwGetX11Display / glfwGetX11Window for the Vulkan surface
// Xlib.h leaks single-word macros that break ordinary C++ (Status/Bool/None are used as
// identifiers by Diligent and the engine alike). The typedefs the two glfwGetX11* calls
// return survive the undefs — only the macro pollution goes.
#undef Status
#undef Bool
#undef True
#undef False
#undef None
#undef Always
#undef Success
#undef Complex
#undef Convex
#undef CursorShape
#include "LinuxNativeWindow.h"    // Diligent-LinuxPlatform: X11 AND Wayland native windows
#include <dlfcn.h>
// Wayland entry points resolve at RUNTIME: linking must not depend on which backends this
// GLFW build carries (the standalone-configure fallback is an X11-only static glfw3).
inline bool NukeGlfwIsWayland()
{
#ifdef GLFW_PLATFORM_WAYLAND
	return glfwGetPlatform() == GLFW_PLATFORM_WAYLAND;
#else
	return false;
#endif
}
inline void* NukeGlfwWaylandDisplay()
{
	static void* (*fn)() = (void* (*)())dlsym(RTLD_DEFAULT, "glfwGetWaylandDisplay");
	return fn ? fn() : nullptr;
}
inline void* NukeGlfwWaylandWindow(GLFWwindow* w)
{
	static void* (*fn)(GLFWwindow*) = (void* (*)(GLFWwindow*))dlsym(RTLD_DEFAULT, "glfwGetWaylandWindow");
	return fn ? fn(w) : nullptr;
}
#endif
#ifdef __APPLE__
#include "MacOSNativeWindow.h"    // Diligent-ApplePlatform: NSView-backed native window
// NukeDiligent_Cocoa.mm: CAMetalLayer attach for the main GLFW window / a secondary NSWindow.
extern "C" void* NukeCocoaMetalView(GLFWwindow* wnd);
extern "C" void* NukeCocoaMetalViewForNSWindow(void* nswindow);
extern "C" void  NukeCocoaSetHiddenFromCapture(GLFWwindow* wnd, bool hide);
#endif
#include "EngineFactoryVk.h"      // Vulkan backend: HLSL->SPIRV via glslang (the only backend off Windows)
#include "RenderDevice.h"
#include "DeviceContext.h"
#include "SwapChain.h"
#include "PipelineState.h"
#include "ShaderResourceBinding.h"
#include "Buffer.h"
#include "Texture.h"
#include "Shader.h"
#include "BottomLevelAS.h"   // ray tracing acceleration structures
#include "TopLevelAS.h"
#include "RefCntAutoPtr.hpp"
#include "MapHelper.hpp"
#include "BasicMath.hpp"
#include "GraphicsAccessories.hpp"
// Windows-only HDR10 display output (DXGI). The d3d11/d3d12/dxgi headers MUST precede
// the Diligent SwapChainD3D* interface headers below.
#ifdef _WIN32
#include <d3d11.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include "SwapChainD3D11.h"
#include "SwapChainD3D12.h"
#endif

// Engine headers last (they do `using namespace std;` internally).
#include "NukeDiligent.h"
#include <interface/NUKEEInteface.h>

#include <cstring>
#include <vector>
#include <map>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <iostream>
#include <cstdlib>   // getenv (diagnostic switches)
#include <deque>
#include <memory>
#include <atomic>
#include <boost/thread.hpp>                      // background pipeline builder
#include <boost/thread/condition_variable.hpp>
#include "API/Model/Log.h"   // Log::Uptime — startup-time accounting (PSO/shader creation stamps)
#include <string>

using namespace Diligent;
// Deliberately NOT `using namespace std` — std::byte clashes with the Windows SDK's ::byte.
using std::cout;
using std::endl;

struct NukeDiligent::Impl
{
	RefCntAutoPtr<IRenderDevice>  device;
	RefCntAutoPtr<IDeviceContext> context;
	RefCntAutoPtr<ISwapChain>     swapChain;

	// Centralized GPU-resource lifetime manager. A GPU object whose raw pointer may still sit in
	// CPU-side data (UI draw lists, texId handles, cached views, SRBs, BLAS refs) is NEVER Release()d
	// inline — Trash() parks a strong ref for kTrashFrames frames so stale pointers stay valid and the
	// address cannot be reused. Purged once per render(); drained fully in deinit().
	static const uint64_t kTrashFrames = 4;   // > max frames in flight + recorded-but-undrawn UI window
	uint64_t frameId = 0;
	std::vector<std::pair<RefCntAutoPtr<IObject>, uint64_t>> gpuTrash;
	std::mutex trashMutex;                    // create/destroy may arrive off the render thread
	void Trash(IObject* o);                   // null-safe: park an object until the GPU can't see it
	void PurgeTrash(bool everything = false); // frame tick (or full drain after IdleGPU)
	// True (logs the reason once) if the D3D12 device has been removed. Guard every Present/Flush
	// with it — the Diligent debug-assert fires INSIDE Present, too late to catch otherwise.
	bool  DeviceRemoved();
	void* d3d12DevCache = nullptr;   // cached ID3D12Device* (void* keeps <d3d12.h> out of the header)
	bool                          useD3D12 = false;   // active backend (set in init from WindowDesc)
	bool                          useVulkan = false;  // backend == 2: Vulkan — no DXGI anywhere

	// Disk shader-bytecode cache (Vulkan only): key = FNV-1a of the full compile inputs, value =
	// compiled SPIR-V under config/shadercache_vk/. A hit feeds ByteCode straight to CreateShader.
	void CreateShaderCached(const Diligent::ShaderCreateInfo& ci, Diligent::IShader** pp);
	// Persistent pipeline cache (VkPipelineCache / D3D12 pipeline library) — the driver's
	// SPIR-V/DXIL -> ISA work is the startup cost (0.4-1 s PER pipeline on Vulkan); with the
	// blob reloaded from disk a warm start creates pipelines in milliseconds.
	RefCntAutoPtr<IPipelineStateCache> psoCache;
	std::atomic<bool> psoCacheDirty{false};   // set from the builder thread too
	double psoCacheSavedAt = 0.0;
	void InitPSOCache();   // load config/psocache_<backend>.bin (after device creation)
	void SavePSOCache(bool force);   // write when dirty (throttled) / at shutdown
	void CreateGraphicsPipelineStateCached(const Diligent::GraphicsPipelineStateCreateInfo& ci, Diligent::IPipelineState** pp)
	{
		// Cache-miss shaders compile ASYNC on the worker pool — the PSO needs them ready.
		const double t0 = nuke::Log::Uptime();
		auto wait = [](Diligent::IShader* s) { if (s) s->GetStatus(true); };
		wait(ci.pVS); wait(ci.pPS); wait(ci.pGS); wait(ci.pHS); wait(ci.pDS);
		const double t1 = nuke::Log::Uptime();
		Diligent::GraphicsPipelineStateCreateInfo c2 = ci;
		c2.pPSOCache = psoCache;
		device->CreateGraphicsPipelineState(c2, pp);
		if (*pp && psoCache) psoCacheDirty = true;
		const double t2 = nuke::Log::Uptime();
		// Diagnostic (NUKE_PSO_TWICE=1): create the identical pipeline again — a warm driver
		// cache makes the repeat ~free, so whatever remains is the backend's own CPU work.
		static const bool twice = std::getenv("NUKE_PSO_TWICE") != nullptr;
		if (twice && *pp)
		{
			Diligent::RefCntAutoPtr<Diligent::IPipelineState> again;
			device->CreateGraphicsPipelineState(c2, &again);
			std::cout << "[NukeDiligent]\tPSO '" << (ci.PSODesc.Name ? ci.PSODesc.Name : "?") << "' REPEAT "
			          << (int)((nuke::Log::Uptime() - t2) * 1000.0) << " ms" << std::endl;
		}
		const double msWait = (t1 - t0) * 1000.0;
		const double msPso  = (t2 - t1) * 1000.0;
		// Startup-time accounting: anything slow ON THE RENDER THREAD is a line in the log
		// (stamped), not a mystery; the builder thread reports per pipe set instead.
		if (msWait + msPso > 30.0 && !IsBuilderThread())
			std::cout << "[NukeDiligent]\tPSO '" << (ci.PSODesc.Name ? ci.PSODesc.Name : "?") << "' " << (int)(msWait + msPso)
			          << " ms (shader wait " << (int)msWait << ", pipeline " << (int)msPso << ")" << std::endl;
	}
	void CreateComputePipelineStateCached(const Diligent::ComputePipelineStateCreateInfo& ci, Diligent::IPipelineState** pp)
	{
		const double t0 = nuke::Log::Uptime();
		if (ci.pCS) ci.pCS->GetStatus(true);
		Diligent::ComputePipelineStateCreateInfo c2 = ci;
		c2.pPSOCache = psoCache;
		device->CreateComputePipelineState(c2, pp);
		if (*pp && psoCache) psoCacheDirty = true;
		const double ms = (nuke::Log::Uptime() - t0) * 1000.0;
		if (ms > 30.0)
			std::cout << "[NukeDiligent]\tPSO '" << (ci.PSODesc.Name ? ci.PSODesc.Name : "?") << "' " << (int)ms << " ms (compute)" << std::endl;
	}
	// Async cache misses: bytecode grabbed and written to disk once the worker finishes.
	// Fed from the builder thread as well — hence the lock.
	std::vector<std::pair<Diligent::RefCntAutoPtr<Diligent::IShader>, std::string>> pendingShaderSaves;
	std::mutex shaderSaveMutex;
	void PollShaderSaves();


	// Pipeline warm-up. Everything that owns pipelines registers a builder here instead of
	// compiling on the draw path: the loop runs the pending ones under a per-frame time budget,
	// so a cold cache costs a few frames of a missing effect rather than one long freeze. A
	// builder returns true when it is done and false to be called again next frame.
	struct WarmEntry { std::string name; bool (*fn)(void*) = nullptr; void* user = nullptr; bool done = false; };
	std::vector<WarmEntry> warmups;
	uint32_t                 warmSamples = 0;
	Diligent::TEXTURE_FORMAT warmFmt = Diligent::TEX_FORMAT_UNKNOWN;
	double                   warmBudgetMs = 3.0;   // per frame, across all builders
	void PumpPipelineWarmup();
	bool                          vsync    = true;    // main-present sync interval (1 = vsync, 0 = uncapped)
#ifdef _WIN32
	// DirectComposition objects for a TRANSPARENT window. IUnknown* keeps <dcomp.h> out of this
	// header (typed use lives in NukeDiligent.cpp).
	IUnknown*                     dcompDevice = nullptr;   // IDCompositionDevice
	IUnknown*                     dcompTarget = nullptr;   // IDCompositionTarget
	IUnknown*                     dcompVisual = nullptr;   // IDCompositionVisual
#endif
	bool                          transparent = false;     // clears to alpha 0; final pass outputs PREMULTIPLIED alpha
	bool                          rtSupported = false;     // device reports ray-tracing capability
	// #include resolver (+ RT shader loading): a MEMORY factory over the sources the engine pushed
	// through setShaderSource. The renderer does NO file IO for shader sources.
	// The #include resolver over shaderSrc. Compiles run on the BUILDER thread and inside the
	// render loop while the game thread keeps pushing sources (module shaders, .nush assets):
	// every access goes through the lock, rebuilds are LAZY (a push only bumps the version) and
	// replaced factories retire into a graveyard until deinit — FXC was caught mid-D3DCompile
	// inside a factory whose last reference a concurrent push had just released (pure virtual
	// call), and the native seam hands modules RAW pointers that must stay valid for the run.
	RefCntAutoPtr<IShaderSourceInputStreamFactory> shaderFactory;
	std::vector<RefCntAutoPtr<IShaderSourceInputStreamFactory>> retiredShaderFactories;
	boost::mutex shaderLock;               // guards shaderSrc + factory + versions
	uint64_t shaderSrcVersion = 1, shaderFactoryVersion = 0;
	RefCntAutoPtr<IShaderSourceInputStreamFactory> ShaderFactory();   // lazy-rebuilt snapshot
	void RebuildShaderFactory();   // shaderLock HELD: rebuild shaderFactory from shaderSrc
	std::atomic<uint64_t> includeEpoch{ 0 };   // FNV over all INCLUDE sources — part of the disk-cache key
	std::vector<Mesh*> rtDynMeshes;   // Mesh::rtDynamic instances added this frame -> per-frame BLAS rebuild
	void RebuildDynamicBLAS();        // rebuild their cached BLASes over the updated vertex buffers

	std::vector<boost::function<void(void)>> onGUI;
	std::vector<boost::function<void(void)>> onRender;
	std::vector<boost::function<void()>>     onClose;

	// --- generic 2D (UI) draw-list renderer ---
	RefCntAutoPtr<IPipelineState>         uiPSO;
	// Per-texture SRB cache (MUTABLE var, set once): zero dynamic descriptors per commit.
	struct UISRBEntry { RefCntAutoPtr<IShaderResourceBinding> srb; uint64_t lastUse = 0; };
	std::unordered_map<ITextureView*, UISRBEntry> uiSRBCache;
	uint64_t uiFrame = 0;
	IShaderResourceBinding* UISRBFor(ITextureView* view);

	// Frame statistics (editor status bar); latched at the start of the next frame.
	int statDraws = 0, statTris = 0;
	int statDrawsOut = 0, statTrisOut = 0;

	// ---- GPU pass timings ------------------------------------------------------------------
	// Duration queries (two timestamps) around each pass. The GPU is frames behind, so results
	// are read from a ring N frames later and never block; each resolved scope is reported to the
	// engine profiler as "gpu.<name>", which is what the status bar and scripts read.
	struct GpuScope
	{
		RefCntAutoPtr<IQuery> query;
		std::string           name;
		bool                  open = false;
	};
	static const int      kGpuRing = 4;
	std::vector<GpuScope> gpuRing[kGpuRing];
	bool                  outlineOpen = false;   // selection-outline mask is accepting meshes
	int                   gpuCur   = 0;
	int                   gpuOpen  = -1;  // index of the pass currently being timed, -1 = none
	size_t                gpuUsed  = 0;   // scopes handed out this frame
	bool                  gpuTimers = false;   // device reports duration-query support
	void GpuFrame();                 // frame boundary: resolve the oldest ring, rotate
	void GpuPass(const char* name);  // start timing a pass (closes the previous one)
	void GpuPassEnd();
	// RAII helper for a pass body that must close at the end of a scope.
	struct GpuMark
	{
		Impl* im;
		GpuMark(Impl* i, const char* n) : im(i) { if (im) im->GpuPass(n); }
		~GpuMark() { if (im) im->GpuPassEnd(); }
	};
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

	// Per-size transient-target cache (scratch / bloom / RT-reflection output): several
	// DIFFERENT-sized cameras render in one frame, so targets must not be recreated mid-frame.
	// Keyed by (w<<32|h), bounded LRU, evictions go through Trash().
	struct SizedTexSet { RefCntAutoPtr<ITexture> a, b; uint64_t lastUsed = 0; };
	std::unordered_map<uint64_t, SizedTexSet> scratchCache, bloomCache, rtOutCache;
	uint64_t sizedClock = 0;                 // shared LRU clock for the sized caches
	void EvictSized(std::unordered_map<uint64_t, SizedTexSet>& cache, uint64_t curKey);

	// --- MSAA --------------------------------------------------------------------------------------
	Uint8 samples = 4;                 // hardware multisample count for all geometry passes (1 = off)
	int   pendingSamples = -1;         // requested sample count; applied at the START of render() (never mid-frame)
	RT    backbufferMS;                // MS color+depth for camera target 0 (Player), resolved to the backbuffer
	// Resolve bookkeeping for the current camera pass (set in beginCamera, used in endCamera).
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
	// Custom post-effect chain: one fullscreen pipeline per effect, ping-ponged in HDR before tonemap.
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
	// Format-safe texture transfer: equal formats -> CopyTexture; different -> fullscreen blit
	// (CopyTextureRegion across formats is an INVALID D3D12 call and poisons the command list).
	void     BlitTexture(ITextureView* srcSRV, ITexture* dstTex);
	std::map<TEXTURE_FORMAT, std::pair<RefCntAutoPtr<IPipelineState>, RefCntAutoPtr<IShaderResourceBinding>>> blitPipes;

	// Built-in bloom: bright-pass -> separable blur (half-res ping-pong) -> composite.
	// Invoked for a chain stage whose post shader is named "bloom".
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
		bool fmtHdr = true;   // SceneFmt the cube was built with (rebuild on HDR toggle to match world PSO)
		// MSAA capture intermediates: sky/world PSOs are built at the current sample count, so drawing
		// straight into the single-sample cube face is a Vulkan render-pass incompatibility (DEVICE_LOST).
		// Faces render here and resolve into the cube slice.
		RefCntAutoPtr<ITexture> msColor, msDepth;
		int msSamples = 1;    // sample count the intermediates were built with (rebuild on change)
	};
	std::unordered_map<uint64_t, CubeRT> cubes;
	void BuildCube(CubeRT& c, int res);   // (re)create the cube GPU resources at the current SceneFmt()
	RefCntAutoPtr<ITexture>              fallbackCube;     // 1x1 cube bound to g_Probe when no probe is active
	ITextureView*                        fallbackCubeSRV = nullptr;
	RefCntAutoPtr<ISampler>              probeSampler;     // linear-clamp; attached to probe/fallback cube SRVs

	// G-buffer prepass (single-sample) for screen-space reflections: normal(oct)+roughness+metalness
	// + depth. Rendered per SSR camera before the colour pass; the "ssr" post effect samples it.
	RefCntAutoPtr<ITexture>             gbufColor, gbufDepth, gbufVel, gbufObjId;
	ITextureView*                       gbufRTV = nullptr, *gbufDSV = nullptr, *gbufSRV = nullptr, *gbufDepthSRV = nullptr;
	ITextureView*                       gbufVelRTV = nullptr, *gbufVelSRV = nullptr;   // screen-space motion (TAA)
	ITextureView*                       gbufObjIdRTV = nullptr, *gbufObjIdSRV = nullptr; // generic per-OBJECT id (pivot hash)
	int                                 gbufW = 0, gbufH = 0;
	// G-buffers are cached PER SIZE (several differently-sized cameras render in one frame; a shared
	// buffer recreated mid-frame is a lifetime race). The gbuf* members above point at the active set.
	// Bounded LRU; eviction drops refs and lets Diligent's deferred release free after the frame fence.
	struct GBufferSet {
		RefCntAutoPtr<ITexture> color, depth, vel, objId;
		ITextureView *rtv = nullptr, *dsv = nullptr, *srv = nullptr, *depthSRV = nullptr,
		             *velRTV = nullptr, *velSRV = nullptr, *objIdRTV = nullptr, *objIdSRV = nullptr;
		uint64_t lastUsed = 0;
	};
	std::unordered_map<uint64_t, GBufferSet> gbufCache;
	uint64_t                            gbufFrameCtr = 0;   // LRU clock
	uint64_t                            gbufCurKey = 0;     // active set's key (never evicted)
	// Overlay-slot texture names: kOvSlots x (albedo, normal, MR, mask2D) + the painted
	// 3D-mask flipbook. Shared by the pipeline var fetch and the draw binds. The G-buffer set
	// skips the albedos (normals/roughness only): kOvSlots x (normal, MR, mask2D) + the mask.
	static const int kOvSlots    = 8;                   // must match Material::kOverlaySlots
	static const int kOvTexCount = kOvSlots * 4 + 3;    // world SRVs: slots + mask3D + detail + detailNrm
	static const int kOvGbufCount = kOvSlots * 3 + 2;   // g-buffer SRVs: slots + mask3D + detailNrm
	static const std::vector<std::string>& OvTexNames();
	static const std::vector<std::string>& OvGbufNames();

	void EvictGBufferCache();
	bool                                gbufActive = false;   // a valid prepass ran for the current camera
	uint64_t                            gbufTarget = ~0ull;   // the camera target that prepass belongs to
	RefCntAutoPtr<IPipelineState>       gbufPSO;
	RefCntAutoPtr<IShaderResourceBinding> gbufSRB;
	IShaderResourceVariable*            gbufMRVar = nullptr;   // PS g_MetalRough (dynamic)
	IShaderResourceVariable*            gbufNrmVar = nullptr;  // PS g_Normal (dynamic) — normal-mapped gbuffer normal
	IShaderResourceVariable*            gbufTexVar = nullptr;  // PS g_Tex (base alpha for cutout)
	IShaderResourceVariable*            gbufWipeVar = nullptr; // PS g_WipeMask (luma-wipe holes)
	// Overlay maps in the G-buffer (kOvSlots x normal/MR/mask2D + the painted 3D mask), one set
	// per SRB variant (normal/instanced/skinned), OvGbufNames() order. gbufOvLast gate the Sets.
	IShaderResourceVariable*            gbufOvVar[kOvGbufCount]     = {};
	IShaderResourceVariable*            gbufOvVarInst[kOvGbufCount] = {};
	IShaderResourceVariable*            gbufOvVarSkin[kOvGbufCount] = {};
	IDeviceObject*                      gbufOvLast[3][kOvGbufCount] = {};
	RefCntAutoPtr<IBuffer>              ssrCB;                 // SSR matrices (view/proj/invProj/res)
	RefCntAutoPtr<IBuffer>              rtRefCB;               // RT reflections (invViewProj, light, ambient, sky)
	// Temporal AA: per-camera history + previous view/proj (keyed by curCamKey), shared CB, and this
	// frame's sub-pixel jitter (applied to the COLOUR projection only, in beginCamera).
	RefCntAutoPtr<IBuffer>              taaCB;
	struct TAAState { RefCntAutoPtr<ITexture> hist; float4x4 prevView, prevProj; bool valid = false; int w = 0, h = 0; uint64_t lastUsed = 0; };
	std::map<uint64_t, TAAState>        taaStates;
	bool                                curTAA = false;        // is the current camera running TAA?
	float                               curJitterX = 0.0f, curJitterY = 0.0f;   // this frame's jitter (pixels, [-0.5,0.5])
	int                                 taaFrame = 0;
	float4x4                            curProjNoJitter;       // curProj before jitter (for TAA reprojection)
	void RunTAA(PostPipe& pp, ITextureView* srcSRV, ITexture* dstTex, int w, int h, const std::vector<float>& params);

	// --- Screen-space ambient occlusion ---------------------------------------------------------
	// Runs off the prepass in beginCamera; screenAOSRV feeds the world PS (g_ScreenAO) for the
	// camera and is dropped in endCamera. Per-target history like TAA.
	int      aoQuality = 0;              // 0 = off, 1 SSAO, 2 HBAO, 3 GTAO, 4 VBAO, 5 RT-AO (World::Settings)
	float    aoRadius = 0.6f, aoIntensity = 1.0f, aoPower = 1.5f;
	PostPipe aoPipe, aoResolvePipe;
	IShaderResourceVariable* aoTlasVar = nullptr;   // RT-AO: g_TLAS on the AO pipe (DXR devices only)
	RefCntAutoPtr<IBuffer> aoCB;
	std::atomic<bool> aoBuilding{false};
	bool     aoFailed = false;
	// raw/den at the AO resolution; resolved (rgb shaped, a accumulated) + hist[2] (r accumulated,
	// g linear depth) at full res — the resolve writes hist[cur] as its second target, reads hist[prev]
	struct AOState { RefCntAutoPtr<ITexture> raw, den, den2, resolved, hist[2]; int w = 0, h = 0, lw = 0, lh = 0, cur = 0; bool valid = false; uint64_t lastUsed = 0; };
	std::map<uint64_t, AOState> aoStates;
	ITextureView* screenAOSRV = nullptr;
	float4x4 gbufView, gbufProj;         // the prepass camera (AO + next frame's motion vectors reproject against THESE)
	int      aoFrame = 0;
	bool BuildAOPipes();
	void RunAO(int w, int h);

	// --- Dynamic diffuse GI (DDGI) ------------------------------------------------------------
	// Volumes pushed per frame; two shared atlases (8x8 irradiance / 16x16 visibility tiles + borders);
	// probe rays by compute + RayQuery, or cube captures folded in by ddgi_cube.cs without RT.
	struct GIVol { NukeGIVolumeDesc desc; int probes = 0, perRow = 0, rows = 0, irrX = 0, irrY = 0, visX = 0, visY = 0; int scroll[3] = {0, 0, 0}; };
	struct GIScroll { long long cell[3] = {0, 0, 0}; int scroll[3] = {0, 0, 0}; bool valid = false; };   // per volume id, across frames
	struct GIReset { int vol, axis, first, count; };
	std::map<uint64_t, GIScroll> giScroll;
	std::vector<GIReset> giResets;           // cells that scrolled into range: tiles zeroed before the next update
	void ApplyGIResets();
	struct GICapture { uint64_t cube = 0; int vol = 0, probe = 0; bool valid = false; };
	std::vector<GIVol> giVols;
	std::vector<GICapture> giCaptures;   // raster fallback: probes captured this frame (budget slots)
	uint64_t giCursor = 0, giLayoutSig = 0, giFrame = 0;
	int giIrrW = 0, giIrrH = 0, giVisW = 0, giVisH = 0;
	float giCaptureMaxD = 0.0f;              // > 0 while a probe cube face renders: world.ps writes distance / this into alpha
	RefCntAutoPtr<ITexture> giIrrAtlas, giVisAtlas;
	ITextureView* giIrrSRV = nullptr; ITextureView* giVisSRV = nullptr;
	RefCntAutoPtr<IBuffer> giCB, giPassCB, giProbeCB, giRayBuf; uint32_t giRayCap = 0;
	RefCntAutoPtr<IPipelineState> giTracePSO, giUpdatePSO, giCubePSO, giProbePSO;
	RefCntAutoPtr<IShaderResourceBinding> giTraceSRB, giUpdateSRB, giCubeSRB, giProbeSRB;
	std::atomic<bool> giBuilding{false}; bool giFailed = false;
	int giProbeSamples = 0; TEXTURE_FORMAT giProbeFmt = TEX_FORMAT_UNKNOWN;
	void WriteGICB();
	bool BuildGIPipes();
	bool EnsureGIPipes();
	void EnsureGIRays(uint32_t count);
	void GIUpdateProbes(int vi, int first, int count, const float rot[4]);
	void DrawGIProbes();

	// --- Screen-space GI (contact bounce from the lit history) ---------------------------------
	int   ssgiQuality = 0; float ssgiRadius = 1.0f, ssgiIntensity = 1.0f;
	PostPipe ssgiPipe, ssgiResolvePipe;
	RefCntAutoPtr<IBuffer> ssgiCB;
	std::atomic<bool> ssgiBuilding{false}; bool ssgiFailed = false;
	struct SSGIState { RefCntAutoPtr<ITexture> lit, raw, den, den2, den3, resolved, hist[2], histZ[2]; int w = 0, h = 0, lw = 0, lh = 0, litW = 0, litH = 0, cur = 0; bool valid = false, litValid = false; uint64_t lastUsed = 0; };
	std::map<uint64_t, SSGIState> ssgiStates;   // per camera key
	ITextureView* screenGISRV = nullptr;         // this camera's bounce for the world PS (null = none)
	int ssgiFrame = 0;
	bool BuildSSGIPipes();
	void RunSSGI(int w, int h);
	void KeepSSGILitHistory(ITextureView* sceneSRV, int w, int h);

	// --- Hi-Z occlusion culling -----------------------------------------------------------------------
	// Two-phase per camera: draws whose id the visibility history calls visible go straight through
	// (phase 1); the rest are deferred. endOpaque builds a MAX depth pyramid from what was drawn,
	// tests EVERY tagged box in a compute pass, replays the deferred draws as indirect draws whose
	// arguments that pass wrote (survivors render this same frame) and copies the verdicts into a
	// staging ring — read back a few frames later into the per-target history.
	struct OcclTag { uint64_t id; float mn[3], mx[3]; };
	struct OcclDeferred
	{
		int       tag = -1;            // index into occlTags
		bool      instanced = false;
		Mesh*     mesh = nullptr; Material* mat = nullptr;
		float     pos[3], quat[4], scale[3];
		uint32_t  firstIndex = 0, indexCount = 0;
		uint64_t  instBuf = 0; int first = 0, count = 0;
		// Indirect-argument seed the test pass copies through (instance count zeroed when culled).
		uint32_t  recCount = 0, recFirst = 0, recInst = 1, recFirstInst = 0; bool recIndexed = true;
		// Per-draw overlay context snapshot (Surface::PushDrawContext): the material is shared, so
		// the values set for THIS draw must be restored before the replay.
		bool      liveSet = false; float liveVal[8], liveChan[8], liveXf[12], liveRes = 0.0f; Texture* liveMask = nullptr;
	};
	struct OcclHist { bool visible = true; uint64_t frame = 0; };
	struct OcclView
	{
		std::unordered_map<uint64_t, OcclHist> hist;
		RefCntAutoPtr<ITexture>                 hiz;        // R32F, full mip chain (MAX reduce)
		RefCntAutoPtr<ITexture>                 hizScratch; // per-level source copy (single-texture state tracking)
		std::vector<RefCntAutoPtr<ITextureView>> hizRTV;    // hiz: one RTV per mip
		std::vector<RefCntAutoPtr<ITextureView>> hizSrcSRV; // scratch: one SRV per mip
		int hizW = 0, hizH = 0, hizMips = 0;
		struct Ring { RefCntAutoPtr<IBuffer> staging; std::vector<uint64_t> ids; int pending = -1; };
		Ring     ring[3]; int ringHead = 0;
		uint64_t lastUsed = 0;
	};
	std::map<uint64_t, OcclView>        occlViews;            // keyed by curCamKey
	bool                                occlEnabled = false, occlFreeze = false;
	bool                                occlPassActive = false; // tags accepted: beginCamera .. endOpaque
	bool                                occlPending = false;    // setOcclusionId armed for the next draw
	bool                                occlDrawTag = false;    // the submit in progress carries the tag
	OcclTag                             occlPendingTag{};
	// One public submit = one tag, whatever its early exits or section count: the scope moves the
	// armed tag onto the submit and drops it on the way out, so nothing leaks onto the next draw.
	struct TagScope
	{
		Impl* im;
		explicit TagScope(Impl* i) : im(i)
		{ im->occlDrawTag = im->occlPending && im->occlReplay < 0; im->occlPending = false; im->occlPendingSlot = -1; }
		~TagScope() { im->occlDrawTag = false; }
	};
	int                                 occlPendingSlot = -1;   // tag slot the pending draw resolved to (-1 = not yet)
	bool                                occlPendingDefer = false;
	int                                 occlReplay = -1;        // deferred record being replayed (draws go indirect)
	std::vector<OcclTag>                occlTags;
	std::vector<OcclDeferred>           occlDeferred;
	RefCntAutoPtr<IBuffer>              occlAabbBuf, occlRecBuf, occlVisBuf, occlArgsBuf, occlCB;
	Uint32                              occlCap = 0;            // element capacity of the buffers above
	RefCntAutoPtr<IPipelineState>       occlCSPSO, hizCopyPSO, hizCopyMSPSO, hizDownPSO;
	RefCntAutoPtr<IShaderResourceBinding> occlCSSRB, hizCopySRB, hizCopyMSSRB, hizDownSRB;
	IShaderResourceVariable*            hizCopyVar = nullptr; IShaderResourceVariable* hizCopyMSVar = nullptr; IShaderResourceVariable* hizDownVar = nullptr;
	uint64_t                            occlFrame = 0;          // frames rendered (history aging)
	int                                 statOcclTracked = 0, statOcclCulled = 0;   // last completed camera
	void CreateOcclResources();
	void OcclBeginCamera();             // reset per-pass lists, consume matured readbacks
	int  OcclEndOpaque();               // pyramid + test + readback copy; 0 = drop deferred, 1 = replay indirect, 2 = replay plain
	bool OcclDecide(uint64_t id);       // history verdict (true = draw now)
	bool OcclArm(int& slot, bool& defer);   // resolve the pending tag for the draw being submitted
	void OcclEnsureBuffers(Uint32 n);
	void OcclBuildHiZ(OcclView& v, ITexture* depth, int w, int h);
	void OcclDebugBoxes();              // frozen view: wire boxes of the culled draws

	// --- Ray tracing (D3D12) ---------------------------------------------------------------------------------
	std::unordered_map<Mesh*, RefCntAutoPtr<IBottomLevelAS>> blasCache;   // BLAS per mesh (built once, reused)
	// v4 indexed meshes: BLAS per IB range (a section, or the whole LOD0). Key packs
	// firstIndex AND indexCount — a whole-LOD0 range and its first section share firstIndex 0.
	std::map<std::pair<Mesh*, uint64_t>, RefCntAutoPtr<IBottomLevelAS>> blasSectionCache;
	IBottomLevelAS* GetMeshBLASRange(Mesh* mesh, uint32_t firstIndex, uint32_t indexCount);

	// --- GPU skinning (stage 3) ---------------------------------------------------------
	struct MeshGPU;   // declared below with the mesh cache
	RefCntAutoPtr<IPipelineState>         skinCSPSO;
	RefCntAutoPtr<IShaderResourceBinding> skinCSSRB;    // one SRB, all-dynamic vars rebound per dispatch
	RefCntAutoPtr<IBuffer>                skinCSParamsCB;
	struct SkinRec
	{
		Mesh* src = nullptr;
		RefCntAutoPtr<IBuffer> palette, morphW;   // per-frame uploads (dynamic, grow-on-demand)
		int boneCap = 0, morphCap = 0;
	};
	std::map<Mesh*, SkinRec> skinRecs;            // skinned INSTANCE mesh -> its skin state
	void EnsureSkinInputs(Mesh* source, MeshGPU& gs);   // lazy static compute inputs on the source
	void RebuildSkinBLAS(Mesh* instance, MeshGPU& gi);  // refit all cached BLAS ranges over the posed verts
	RefCntAutoPtr<ITopLevelAS>         tlas;                  // scene TLAS (rebuilt per frame)
	RefCntAutoPtr<IBuffer>             tlasScratch, tlasInstanceBuf;
	Uint32                             tlasMaxInstances = 0;
	// Foliage RT bend (bend.cs): bends the merged chunk meshes with the shared NukeBend and
	// rebuilds their BLAS every frame, so RT shadows/reflections of vegetation sway.
	RefCntAutoPtr<IPipelineState>         bendCSPSO;
	RefCntAutoPtr<IShaderResourceBinding> bendCSSRB;
	RefCntAutoPtr<IBuffer>                bendCSParamsCB;
	std::vector<Mesh*>                    rtBendMeshes;          // bend meshes seen this frame (cleared in beginRTScene)
	bool                                  blasBentThisFrame = false;
	void BendRTMeshes();
	size_t                             lastTlasSig = 0;        // topology signature (count + BLAS set) -> refit when unchanged
	uint32_t                           tlasFrameCtr = 0;       // periodic full rebuild counter (refit hygiene)
	uint64_t                           lastRTFullSig = 0;      // full scene hash (BLASes+transforms+masks+data) -> SKIP build when identical
	std::vector<TLASBuildInstanceData> rtInstances;           // accumulated between beginRTScene/buildRTScene
	std::vector<std::string>           rtInstanceNames;        // grow-only "i<n>" pool backing TLASBuildInstanceData::InstanceName
	std::map<Material*, uint32_t>      rtMatBlockCache;        // material -> allMatCPU offset of its built block (per accumulation)
	bool                               rtSceneReady = false;   // a valid TLAS is built for the current frame
	RefCntAutoPtr<ITopLevelAS>         fallbackTLAS;           // empty TLAS bound to g_TLAS when no scene TLAS (rays miss)
	RefCntAutoPtr<IBuffer>             fbTlasScratch, fbTlasInst;
	IBottomLevelAS* GetMeshBLAS(Mesh* mesh);                   // get-or-build the BLAS for a mesh (from its pos buffer)
	void EnsureRTFallback();                                   // build the empty fallback TLAS once

	// RT reflection hit shading: per-instance geometry offsets (into the concatenated normal/uv/pos
	// buffers) + material block offset in g_MatBytes. MIRRORED byte-for-byte in rt_common.hlsl
	// (RTInstanceData) AND world.ps.hlsl (RTInstInfo) — change all three together. 16-byte rows.
	struct RTInstanceData {
		uint32_t nrmOffset, uvOffset, posOffset, matByteOffset;
		uint32_t texIndex, nrmTexIndex, mrTexIndex, aoTexIndex;
		uint32_t emTexIndex, specTexIndex; float specularFactor; uint32_t nrmFlipG;   // 1 = flip green (OpenGL)
		float albedoMetal[4]; float emissiveRough[4];
		// Particles: colOffset = byte offset into g_DynCol (0xFFFFFFFF = none),
		// shadowShape = 0 quad / 1 disc / 2 strip.
		uint32_t colOffset; uint32_t shadowShape; float shadowAlpha; uint32_t pad0;
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
	// Per-frame color pool for dynamic sprite meshes (particle gradients/fade), rebuilt from
	// Mesh::rtColorArray of the instances added. RAW buffer, grows as needed.
	std::vector<float>                  rtDynColCPU;
	RefCntAutoPtr<IBuffer>              rtDynColBuf;  IBufferView* rtDynColSRV = nullptr;
	Uint64                              rtDynColCap = 0;
	std::vector<RTInstanceData>         rtInstData;            // parallel to rtInstances, rebuilt per frame
	uint32_t                            rtInstCapacity = 0;
	static const uint32_t               kMaxMatTex = 256;      // bindless material maps (albedo/normal/MR/AO/emissive/spec)
	std::unordered_map<Texture*, uint32_t> matTexSlot;         // engine texture -> slot in the bindless array
	std::vector<ITextureView*>          matTexSRVs;            // unique material map SRVs (<= kMaxMatTex)
	std::vector<Texture*>               matTexPtr;             // engine texture per slot (re-resolve SRV each frame -> animation)

	// --- RT reflection pipeline (DXR: ray-gen + miss + closest-hit + SBT) ---
	RefCntAutoPtr<IPipelineState>         rtPSO;               // ray-tracing PSO (rt_rgen/rt_rmiss/rt_rchit)
	RefCntAutoPtr<IShaderResourceBinding> rtSRB;               // dynamic resources (TLAS, gbuffer, bindless, output)
	RefCntAutoPtr<IShaderBindingTable>    rtSBT;               // shader binding table (ray-gen + miss + hit group)
	RefCntAutoPtr<ITexture>               rtOutTex;            // UAV the ray-gen writes the composited reflection into
	int                                   rtOutW = 0, rtOutH = 0;
	// Global RTX reflection quality (pushed via setRTReflection).
	float rtCfgIntensity = 1.0f; float rtCfgMaxDist = 100.0f; int rtCfgBounces = 3; float rtCfgRoughCut = 0.6f;
	bool BuildRTPipeline();                                    // build rtPSO/rtSRB/rtSBT (needs shaderFactory + DXC)
	void EnsureRTOutput(int w, int h);                         // (re)create the RGBA16F UAV output at viewport size
	void RunRTReflectPipeline(ITextureView* srcSRV, ITexture* dstTex, int w, int h, const std::vector<float>& params);
	// Auto-generated per-shader RT closest-hits: a material shader with a "<name>.surf.hlsl" gets its
	// own hit group, built by GenChitSource() from the shader's MatCB schema + that surface file.
	std::unordered_map<std::string, std::string> rtSurfShaders;  // shader name -> its PS source (has the MatCB schema)
	// Surface bodies registered from CODE (module-embedded shaders have no file on the include
	// path); GenChitSource inlines these instead of the #include.
	std::unordered_map<std::string, std::string> rtSurfSources;  // shader name -> Surface() HLSL body
	std::unordered_map<std::string, std::string> shaderHitGroup; // shader name -> hit-group name in the RT PSO
	std::string GenChitSource(const std::string& name, const std::string& psSource);  // codegen the closest-hit HLSL
	std::atomic<bool> rtPipelineDirty{false};                  // a new surf shader appeared -> rebuild rtPSO (set on the render thread, cleared by the builder)
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
		bool wantsVcol = false;   // 4-element input layout (NUKE_VCOLOR): draws need a color VB
		// Shader-declared g_Layer* SRVs beyond the fixed set: filled from Material::extraTex by name.
		struct ExtraVar { std::string name; IShaderResourceVariable* var = nullptr; IDeviceObject* last = nullptr; };
		std::vector<ExtraVar> extraVars;
		RefCntAutoPtr<IPipelineState>         pso;        // opaque (blend off, depth write on)
		RefCntAutoPtr<IPipelineState>         psoBlend;   // transparent: alpha blend, depth test on / write off
		RefCntAutoPtr<IPipelineState>         psoAdd;     // additive: add blend, depth write off
		RefCntAutoPtr<IPipelineState>         psoWire;    // wireframe fill (scene draw-mode toggle)
		// Vertex-color variant (opt-in: sources handle NUKE_VCTINT): the mesh's color stream
		// tints the base color or masks the overlay slots. Same resource layout -> shares srb.
		RefCntAutoPtr<IPipelineState>         psoVcol, psoVcolBlend, psoVcolAdd;
		// Displacement tessellation variant (opt-in: VS handles NUKE_TESS + PS declares g_Disp).
		// Own SRB (extra HS/DS stages + DOMAIN g_Height); bound by NAME per draw — tess draws are rare.
		RefCntAutoPtr<IPipelineState>         psoTess;
		RefCntAutoPtr<IShaderResourceBinding> srbTess;
		RefCntAutoPtr<IShaderResourceBinding> srb;
		IShaderResourceVariable*              texVar  = nullptr;  // PS "g_Tex"        (base color, dynamic)
		IShaderResourceVariable*              normVar = nullptr;  // PS "g_Normal"     (normal map, dynamic)
		IShaderResourceVariable*              mrVar   = nullptr;  // PS "g_MetalRough" (dynamic)
		IShaderResourceVariable*              aoVar   = nullptr;  // PS "g_Occlusion"  (dynamic)
		IShaderResourceVariable*              emVar   = nullptr;  // PS "g_Emissive"   (dynamic)
		IShaderResourceVariable*              specVar = nullptr;  // PS "g_Spec"       (specular map, dynamic)
		IShaderResourceVariable*              wipeVar = nullptr;  // PS "g_WipeMask"   (luma-wipe mask, dynamic)
		IShaderResourceVariable*              heightVar = nullptr;// PS "g_Height"     (POM/displacement height, dynamic)
		// Overlay slots: kOvSlots x (albedo, normal, MR, mask2D) + the
		// painted 3D-mask flipbook. Same order as OvTexNames(); lastBind[13..] gate them.
		IShaderResourceVariable*              ovVar[kOvTexCount] = {};
		// BRDF pack: anisotropy flow map + the pre-transparent scene snapshot.
		IShaderResourceVariable*              flowVar = nullptr;   // PS "g_Flow"
		IShaderResourceVariable*              refrVar = nullptr;   // PS "g_SceneRefr"
		IShaderResourceVariable*              saoVar = nullptr;    // PS "g_ScreenAO" (screen-space AO visibility)
		IShaderResourceVariable*              giIrrVar = nullptr, *giVisVar = nullptr;   // PS DDGI atlases
		IShaderResourceVariable*              sgiVar = nullptr;    // PS "g_ScreenGI" (screen-space bounce)
		IShaderResourceVariable*              mskVar  = nullptr;   // PS "g_MskStamp" (LiveMask stamp)
		IShaderResourceVariable*              shadowVar = nullptr;// PS "g_Shadow"      (dynamic)
		IShaderResourceVariable*              cubeVar   = nullptr;// PS "g_ShadowCube" (dynamic)
		IShaderResourceVariable*              probeVar  = nullptr;// PS "g_Probe" (reflection cubemap, dynamic)
		IShaderResourceVariable*              tlasVar   = nullptr;// PS "g_TLAS" (ray-tracing accel struct, RT builds only)
		IShaderResourceVariable*              rtInstVar = nullptr;// PS "g_RTInst" (per-instance RT data: shadow footprints)
		// INSTANCED variants: built only when the shader source handles NUKE_INSTANCED. Same blend
		// variants; the instanced SRB has its OWN variable set (a different compiled shader pair).
		RefCntAutoPtr<IPipelineState>         psoInst, psoInstBlend, psoInstAdd, psoInstWire;
		RefCntAutoPtr<IShaderResourceBinding> srbInst;
		IShaderResourceVariable *texVarI = nullptr, *normVarI = nullptr, *mrVarI = nullptr, *aoVarI = nullptr,
		                        *emVarI = nullptr, *specVarI = nullptr, *wipeVarI = nullptr, *heightVarI = nullptr,
		                        *shadowVarI = nullptr, *cubeVarI = nullptr, *probeVarI = nullptr, *tlasVarI = nullptr,
		                        *rtInstVarI = nullptr;
		IShaderResourceVariable *ovVarI[kOvTexCount] = {};
		IShaderResourceVariable *flowVarI = nullptr, *refrVarI = nullptr, *mskVarI = nullptr, *saoVarI = nullptr, *giIrrVarI = nullptr, *giVisVarI = nullptr, *sgiVarI = nullptr;
		// Redundancy gates: object each DYNAMIC variable currently holds — Diligent rewrites the
		// descriptor cache on EVERY Set() of a dynamic var, so only Set() on an actual change.
		// [0..12] = tex,norm,mr,ao,em,spec,shadow,cube,probe,tlas,rtinst,wipe,height;
		// [13..] = overlay-slot maps (OvTexNames() order), then flow, scene-refraction, mask stamp, screen AO, GI irradiance, GI visibility, screen GI.
		IDeviceObject* lastBind[13 + kOvTexCount + 7]  = {};
		IDeviceObject* lastBindI[13 + kOvTexCount + 7] = {};
		std::string vsSrc, psSrc, dbg;   // kept so the pipeline can be rebuilt (e.g. on MSAA change)
		std::string hsSrc, dsSrc;        // CUSTOM tess stages (empty = shared world.hs/world.ds)
		bool tessCustom = false;         // the shader SHIPPED hs/ds (builder copies resolve the shared pair into hsSrc)
		// What this pipeline was built for. Stale or never-built pipes are skipped by the draw
		// and rebuilt by the warm-up; the draw falls back to the default world pipeline.
		Uint8          builtSamples = 0;
		TEXTURE_FORMAT builtFmt = TEX_FORMAT_UNKNOWN;
		bool           buildFailed = false;   // latched: a broken shader is not retried every frame
		bool           building = false;      // a background build is in flight for this pipe
		int            builtStages = 0;       // kStage* bits adopted so far (base first, the rest by priority)
	};
	// Build stages of a world pipe — what a draw needs first comes first, across ALL shaders:
	// base (opaque + instanced opaque + SRBs: objects appear), blend (transparent/additive),
	// extra (vertex-color tint, tessellation), wire (editor wireframe). Priorities interleave
	// the module/G-buffer/RT jobs between them.
	enum : int { kStageBase = 1, kStageBlend = 2, kStageExtra = 4, kStageWire = 8, kStageAll = 15 };
	enum : int { kPrioBase = 0, kPrioGBuffer = 5, kPrioBlend = 10, kPrioModule = 12, kPrioExtra = 20, kPrioRT = 25, kPrioWire = 30 };
	// The boot pipeline: a tiny unlit-textured world shader built synchronously at init (tens of
	// ms). PipeFor hands it out while neither the shader's own pipe nor the default one exists,
	// so the scene is on screen from the first frame and fades into full shading as the real
	// pipelines land in the background.
	WorldPipe bootPipe;
	std::unordered_map<uint64_t, WorldPipe> worldPipes;   // shader handle -> pipeline
	uint64_t                              defaultWorldHandle = 0;   // builtin "world" pipeline

	// --- background pipeline builder --------------------------------------------------------
	// World pipelines (13 variants per shader) are built on a dedicated thread: the Diligent
	// device is thread-safe for object creation, the draw path already skips a pipe that is not
	// ready (PipeFor falls back / returns null), so startup shows frames immediately and
	// objects appear as their pipelines land — never a frozen window. Builds are snapshots
	// (sources copied at request time); the render thread adopts finished results in the
	// warm-up pump and parks the replaced GPU objects through Trash.
	struct PipeBuild
	{
		uint64_t       handle = 0;
		WorldPipe      pipe;                      // sources in, pipelines out
		int            stage = kStageBase;        // which kStage* this build produces
		Uint8          samples = 0;               // what it is being built for
		TEXTURE_FORMAT fmt = TEX_FORMAT_UNKNOWN;
		RefCntAutoPtr<IShaderSourceInputStreamFactory> factory;   // include resolver snapshot
		bool           ok = false;
	};
	struct BuildJob { boost::function<void()> build, adopt; std::string name; };
	// ONE priority queue for pipe stages and jobs: lower prio first, ties by arrival.
	struct BuildItem { int prio = 0; uint64_t seq = 0; std::shared_ptr<PipeBuild> pipe; std::shared_ptr<BuildJob> job; };
	std::vector<BuildItem>                  buildQueue;
	uint64_t                                buildSeq = 0;
	std::vector<std::shared_ptr<PipeBuild>> pipeDone;    // built, awaiting adoption
	boost::mutex              pipeMutex;
	boost::condition_variable pipeCv;
	std::vector<boost::thread> pipeThreads;   // a small pool: pipes of different shaders build concurrently
	bool                      pipeStop = false, pipeThreadStarted = false;
	// Generic background build: `build` runs on the builder thread (device-object creation
	// only — never the context), `adopt` on the render thread once it finished (publish flags,
	// swap pointers). Used by the G-buffer/RT pipelines and by modules through the native hatch.
	std::vector<std::shared_ptr<BuildJob>> jobDone;
	void EnqueueBuild(const boost::function<void()>& build, const boost::function<void()>& adopt,
	                  int prio = kPrioModule, const char* name = "");
	void EnqueueItem(BuildItem&& it);   // sorted insert + wake
	std::atomic<bool> gbufBuilding{false};   // G-buffer pipes in flight: the prepass skips
	std::atomic<bool> rtBuilding{false};     // RT reflection pipeline in flight: the pass blits through
	void BuildBootPipe();               // synchronous stand-in (boot.vs/ps), init + MSAA change
	void StartPipeBuilder();
	void StopPipeBuilder();
	void PipeBuilderLoop();
	static bool& BuilderThreadFlag() { static thread_local bool f = false; return f; }
	static bool  IsBuilderThread() { return BuilderThreadFlag(); }
	void RequestPipeBuild(uint64_t h, int stage);   // render thread: snapshot + enqueue one stage (one in flight per pipe)
	void AdoptBuiltPipes();              // render thread: swap finished builds into worldPipes
	void TrashPipe(WorldPipe& wp);       // park a pipe's GPU objects (may be in flight this frame)
	// Sample count / scene format an auxiliary pipeline set was built for. Same contract as
	// WorldPipe: the draw skips a stale set, the warm-up rebuilds one set per frame.
	struct PipeStamp
	{
		Uint8          s = 0;
		TEXTURE_FORMAT f = TEX_FORMAT_UNKNOWN;
		bool current(Uint8 s2, TEXTURE_FORMAT f2) const { return s == s2 && f == f2; }
		void stamp(Uint8 s2, TEXTURE_FORMAT f2) { s = s2; f = f2; }
	};
	PipeStamp skyStamp, debugStamp, spriteStamp, decalStamp, outlineStamp;
	bool WarmEnginePipelines();   // renderer's own entry in the warm-up pump
	// The pipeline for a shader handle, or the default one while that shader's pipeline is
	// still being built (or was built for another sample count / scene format). Null only when
	// even the default is not usable.
	WorldPipe* PipeFor(uint64_t h)
	{
		const TEXTURE_FORMAT fmt = SceneFmt();
		auto ok = [&](WorldPipe& w) { return w.pso && w.builtSamples == samples && w.builtFmt == fmt; };
		auto it = worldPipes.find(h);
		if (it != worldPipes.end() && ok(it->second)) return &it->second;
		auto d = worldPipes.find(defaultWorldHandle);
		if (d != worldPipes.end() && ok(d->second)) return &d->second;
		if (ok(bootPipe)) return &bootPipe;   // nothing real yet: the unlit boot shading
		return nullptr;
	}

	// Per-draw redundancy gates: the shared dynamic CBs (worldCB/worldMatCB/drawFlagsCB) only re-map
	// when their content changes WITHIN a pass. Every pass begin bumps passSerial and invalidates all
	// gates — dynamic rings recycle per frame and other passes write the same buffers.
	uint32_t        passSerial = 1;                 // bumped by every begin*Pass / beginCamera / beginCubeFace
	const Material* matCBFor   = nullptr;           // material whose bytes sit in worldMatCB + drawFlagsCB
	uint32_t        matCBPass  = 0;                 // ...valid while == passSerial
	float           matCBTessF = 0.0f;              // quantized tess factor those bytes carry in g_Disp.w
	// Tess SRB name-bind gate: the bound set is per MATERIAL, not per draw (terrain = dozens
	// of draws sharing one palette; per-draw GetVariableByName sweeps were a CPU sink).
	const Material* tessBindMat  = nullptr;
	uint32_t        tessBindPass = ~0u;
	IShaderResourceBinding* sceneCommitSrb = nullptr;   // last SRB committed by RenderObjectRange...
	uint32_t                sceneCommitPass = 0;        // ...valid while == passSerial and no new Set()
	RefCntAutoPtr<IBuffer>  dummyColVB;                 // zero vcolors for vcol pipes on colorless meshes
	uint32_t                dummyColVerts = 0;          // ...grown on demand to the largest such mesh
	uint32_t        instWorldCBPass = 0;            // pass in which worldCB holds the instanced VP (identity world)
	struct { const void* mesh = nullptr; uint64_t buf = 0; void* pso = nullptr; uint32_t pass = 0; } lastInstBind;

	// --- GPU instancing: persistent instance buffers (handle -> dynamic VB of NukeInstanceData) ---
	struct InstBuf { RefCntAutoPtr<IBuffer> buf; int capacity = 0; int count = 0; };
	std::unordered_map<uint64_t, InstBuf> instBufs;
	uint64_t nextInstBuf = 1;
	// Instanced twins of the shadow / g-buffer pipelines (same CBs, instanced input layout).
	RefCntAutoPtr<IPipelineState>         shadowPSOInst;
	RefCntAutoPtr<IShaderResourceBinding> shadowSRBInst;
	IShaderResourceVariable*              shadowPsTexVarInst = nullptr;
	IShaderResourceVariable*              shadowPsWipeVarInst = nullptr;   // shadow PS "g_WipeMask" (inst)
	RefCntAutoPtr<IPipelineState>         gbufPSOInst;
	RefCntAutoPtr<IShaderResourceBinding> gbufSRBInst;
	IShaderResourceVariable*              gbufMRVarInst = nullptr;
	IShaderResourceVariable*              gbufNrmVarInst = nullptr;
	IShaderResourceVariable*              gbufTexVarInst = nullptr;
	IShaderResourceVariable*              gbufWipeVarInst = nullptr;
	// Skinned twin (NUKE_SKINNED): stream 3 = previous-frame skinned positions -> true MVs.
	RefCntAutoPtr<IPipelineState>         gbufPSOSkin;
	RefCntAutoPtr<IShaderResourceBinding> gbufSRBSkin;
	IShaderResourceVariable*              gbufMRVarSkin = nullptr;
	IShaderResourceVariable*              gbufNrmVarSkin = nullptr;
	IShaderResourceVariable*              gbufTexVarSkin = nullptr;
	IShaderResourceVariable*              gbufWipeVarSkin = nullptr;
	bool warnedNoInstPipe = false;   // one-shot log: material shader without an instanced variant
	uint64_t                              nextShaderHandle   = 1;   // handles handed to the engine
	RefCntAutoPtr<IBuffer>                worldCB;     // VS: WVP + World   (shared)
	RefCntAutoPtr<IBuffer>                worldMatCB;  // PS: color + params + custom shader props (shared)
	// MatCB capacity: std block 1024 + custom shader props. The terrain splat shader now packs
	// 17 float4s past the std block (1296 bytes total) — an overflowing prop is SILENTLY
	// dropped and the shader goes black, so keep headroom.
	static const uint32_t                 kMatCBBytes = 2048;
	float                                 tessFillFactor = 0.0f;   // per-draw tess factor patched into g_Disp.w
	Diligent::RefCntAutoPtr<Diligent::IBuffer> drawFlagsCB;    // per-draw flags (x = receiveShadows)
	RefCntAutoPtr<ITexture>               whiteTex;    // 1x1 fallback when a material has no texture
	RefCntAutoPtr<ITexture>               flatNormTex; // 1x1 (0.5,0.5,1) flat normal fallback
	RefCntAutoPtr<IBuffer>                worldFrameCB;// PS b1: camera pos + ambient + light array (shared)
	// PBR lighting buffer layout — MUST match FrameCB in world.ps.hlsl. Each float4 = 16 bytes.
	static const int                      kMaxLights = 256;
	struct GPULight { float posType[4]; float dirRange[4]; float colorIntensity[4]; float spot[4]; };
	struct FrameCBData { float camPos[4]; float ambient[4]; float lightCount[4]; GPULight lights[kMaxLights];
	                     float shadowVP[16 * 4]; float shadowParams[4];        // 4 = SHADOW_SLOTS
	                     float skyTop[4]; float skyHorizon[4]; float skyGround[4]; float skyParams[4];      // IBL
	                     float probePos[4]; float probeParams[4]; float probeBox[4];    // probe: pos.xyz+active, intensity+maxMip, boxHalf.xyz+valid
	                     float wind[4]; float wind2[4];      // dir.xyz+gusted strength; turbAmount, 1/turbScale, time, gustFreq
	                     float misc[4]; };                    // x = GI probe capture (alpha = distance / y), y = max distance
	float windDirStrength[4] = { 1, 0, 0, 0 };   // setWind (pushed per frame)
	float windParams[4]      = { 0, 0, 0, 0 };
	// Foliage bend: the VS-side BendCB — wind + up to 8 "pushers" that part the blades. Written
	// once per frame from setWind; bound as a STATIC var on every INSTANCED pipeline whose vertex
	// shader declares cbuffer BendCB.
	RefCntAutoPtr<IBuffer> bendCB;
	float bendPushers[8][4] = {};
	int   bendPusherCount = 0;
	float bendVolumes[16][12] = {};   // (pos,r)(dir,strength)(mode,falloff,seed,0) per volume
	int   bendVolumeCount = 0;
	void  UpdateBendCB();
	void WriteFrameCB(const Diligent::float3& P);   // fill worldFrameCB (lights/shadows/sky/probe)
	float                                 curCamPos[3] = {0, 0, 0};  // set in beginCamera (PBR view dir)
	float                                 curCamFwd[3] = {0, 0, 1};  // camera forward (ripple window aim)
	bool                                  curCamEditor = false;      // this pass = editor viewport camera
	uint64_t                              curTarget = 0;             // RT id bound by beginCamera (feedback guard)
	uint64_t                              curCamKey = 0;             // per-camera state key (target + camera id): TAA / AO / occlusion views
	static uint64_t CamKey(const NukeCameraDesc& c) { return c.target ^ (c.cameraId * 0x9E3779B97F4A7C15ull); }
	void PruneCameraStates();   // once per frame: drop TAA / AO / occlusion states nobody rendered with lately
	// True between beginCamera binding its targets and the end of endCamera. Sprites REQUIRE the
	// camera's colour+depth targets, so sprite calls outside a camera pass are dropped.
	bool                                  cameraPassActive = false;
	// RT water attenuation, fed by the native water hooks (NukeDiligent_Native.cpp), consumed by
	// the ray shaders' CB fill (RT.cpp).
	float rtWaterOcc[4] = { 0, 0, 0.25f, 0 };   // level, on (this frame), 1/opacityDepth, 0
	float rtWaterCol[3] = { 0.02f, 0.10f, 0.09f };
	float rtWaterAbs[3] = { 0.45f, 0.09f, 0.06f };
	// Generic ortho bottom-depth capture (begin/end/fetchWaterBottom, NukeDiligent_Native.cpp).
	RefCntAutoPtr<ITexture> capDepth, capStaging;
	int   capPending = -1;
	bool  capActive = false;
	float capLevel = 0.0f, capEyeY = 0.0f, capNear = 0.0f;
	std::vector<NukeLight>                lights;      // scene lights (setLights); empty -> default sun

	// --- Shadow maps (directional + spot share a 2D array; one slice per shadow-casting light) -----
	int                                   shadowRes    = 2048;   // World-Settings-driven (rebuilds on change)
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
	IShaderResourceVariable*              shadowPsWipeVar = nullptr;  // shadow PS "g_WipeMask" (luma-wipe holes)
	RefCntAutoPtr<IBuffer>                shadowVSCB;    // VS: g_LightWVP (per shadow draw)
	RefCntAutoPtr<IBuffer>                shadowPSCB;    // PS: g_Alpha    (per shadow draw)
	RefCntAutoPtr<ISampler>               shadowCmpSampler;   // PCF comparison sampler (set on shadowSRV)
	NukeSky                               sky;   // procedural sky / environment
	// Debug/gizmo lines (iRender::drawDebugLine): per-frame vertex list (pos3+col4), mutex-guarded
	// (game + fixed threads emit), cleared at the top of render(). Drawn AFTER the post chain (LDR,
	// no depth) — TAA has no velocity for lines and the RT-reflection composite would overwrite them.
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
	// DEPTH-TESTED lines (drawDebugLineDepth): drawn INSIDE the camera pass while the MS scene targets
	// are still bound, so geometry occludes them. PSO depends on samples/HDR -> built lazily against
	// the current SceneFmt()/samples. Batch is CONSUMED per camera so gizmos don't leak between passes.
	RefCntAutoPtr<IPipelineState>         debugDepthPSO;
	RefCntAutoPtr<IShaderResourceBinding> debugDepthSRB;
	int            debugDepthSamples = 0;                        // PSO built for this sample count
	TEXTURE_FORMAT debugDepthFmt     = TEX_FORMAT_UNKNOWN;       // ...and this scene format
	// Editor infinite grid: analytic shader plane, drawn depth-tested before the gizmo lines.
	RefCntAutoPtr<IPipelineState>         gridPSO;     // fallback: depth-tested + ULP bias (no prepass)
	RefCntAutoPtr<IShaderResourceBinding> gridSRB;
	RefCntAutoPtr<IPipelineState>         gridPSOND;   // depth-aware: no depth test, PS occludes from the prepass
	RefCntAutoPtr<IShaderResourceBinding> gridSRBND;
	IShaderResourceVariable*              gridDepthVar   = nullptr;
	IShaderResourceVariable*              gridDepthVarND = nullptr;
	RefCntAutoPtr<IBuffer>                gridCB;
	int            gridSamples = 0;
	TEXTURE_FORMAT gridFmt     = TEX_FORMAT_UNKNOWN;
	float          gridStep    = 0.0f;      // per-frame handover from the editor (0 = hidden)

	// Custom cursors: hardware = cached GLFW cursors, software = cached textures drawn by
	// DrawCursorPass after the UI.
	std::map<uint64_t, GLFWcursor*> hwCursors;
	struct SwCursor
	{
		RefCntAutoPtr<ITexture>               tex;
		RefCntAutoPtr<IShaderResourceBinding> srb;
		int w = 0, h = 0, hotX = 0, hotY = 0;
	};
	std::map<uint64_t, SwCursor> swCursors;
	GLFWwindow* cursorWindow = nullptr;
	uint64_t curCursorId  = 0;      // active id (0 = none/OS arrow)
	int      curCursorMode = 0;     // 0 arrow / 1 hardware / 2 software
	RefCntAutoPtr<IPipelineState> cursorPSO;
	RefCntAutoPtr<IBuffer>        cursorCB;
	TEXTURE_FORMAT cursorFmt = TEX_FORMAT_UNKNOWN;
	void DrawCursorPass();          // software cursor: draw over the finished backbuffer
	// Fullscreen video overlay: drawn letterboxed after the UI (reuses the cursor PSO).
	// claimScreenOverlay flips overlayClaimed and the editor draws it in its viewport instead.
	Texture* overlayTex = nullptr;
	bool overlayClaimed = false;
	RefCntAutoPtr<IShaderResourceBinding> overlaySRB;
	ITextureView* overlayLastSRV = nullptr;
	bool EnsureCursorPSO();
	void DrawOverlayPass();

	// Mesh-cost debug view (setDebugView 1): world draws render flat-colored by triangle load
	// + dark wireframe, through their own PSOs (plain/instanced x solid/wire).
	int debugView = 0;
	RefCntAutoPtr<IPipelineState> costPSO, costPSOInst, costWirePSO, costWirePSOInst, costProxyPSO;
	RefCntAutoPtr<IShaderResourceBinding> costSRB, costSRBInst, costWireSRB, costWireSRBInst, costProxySRB;
	RefCntAutoPtr<IBuffer> costCB, costCubeVB;
	PipeStamp costStamp;
	bool EnsureCostPSOs();
	void DrawCostRange(MeshGPU& g, const float4x4& world, uint32_t firstIndex, uint32_t indexCount, float wireAlpha);
	void DrawCostInstanced(MeshGPU& g, Mesh* mesh, IBuffer* instBuf, int first, int count);
	void DrawCostProxyBox(const float pos[3], const float quat[4], const float size[3], double tris);

	void DrawEditorGridPass();              // endCamera, before DrawDepthDebugLines
	std::vector<float> debugVertsDepth;   // 7 floats per vertex (shares debugMutex + debugVB)
	void DrawDepthDebugLines();

	// Sprites (iRender::drawSprite): unlit textured quads, alpha-blended, drawn IN the camera pass
	// (SceneFmt + MSAA + depth-test, no depth write). PSO depends on samples/HDR and rebuilds with them.
	RefCntAutoPtr<IPipelineState>         spritePSO;
	RefCntAutoPtr<IShaderResourceBinding> spriteSRB;
	IShaderResourceVariable*              spriteTexVar = nullptr;   // PS g_Sprite (dynamic)
	RefCntAutoPtr<IBuffer>                spriteCB;                 // view*proj
	RefCntAutoPtr<IBuffer>                spriteVB;                 // dynamic (grows), 9 floats/vertex
	int                                   spriteVBSize = 0;         // VB capacity in vertices
	// Batching: drawSprite accumulates quads and flushes ONE draw per texture run (sprites arrive
	// pre-sorted back-to-front). Flushed on texture change and at endCamera, before the MSAA resolve.
	Texture*                              spriteBatchTex = nullptr;
	bool                                  spriteBatchOpen = false;   // batch live (tex may legally be null -> white 1x1)
	float                                 spriteSoftDist = 0.f;      // soft-particle fade distance for the CURRENT run (0 = off)
	IShaderResourceVariable*              spriteDepthVar = nullptr;  // PS "g_SceneDepth" (prepass depth; white when absent)
	float                                 curNear = 0.1f, curFar = 1000.f;   // camera planes (soft-particle linearization)
	std::vector<float>                    spriteBatchVerts;
	void CreateSpriteResources();
	void FlushSprites();

	// LIT sprite runs (drawSpriteRunLit — tilemap layers with a normal map): same batch layout,
	// Lambert lighting from worldFrameCB, per-batch plane TBN in spriteLitCB. SRBs cached per
	// (diffuse, normal) SRV pair; flushed on pair change / kind switch / endCamera.
	RefCntAutoPtr<IPipelineState>         spriteLitPSO;
	RefCntAutoPtr<IBuffer>                spriteLitCB;              // float4 T,B,N (N.w = green flip)
	std::map<std::pair<ITextureView*, ITextureView*>, RefCntAutoPtr<IShaderResourceBinding>> spriteLitSRBs;
	Texture*                              spriteLitTex = nullptr;
	Texture*                              spriteLitNormal = nullptr;
	bool                                  spriteLitFlipY = true;
	std::vector<float>                    spriteLitVerts;
	void FlushSpritesLit();

	// Screen-space (Canvas HUD) sprites — verts already in NDC, identity transform. Two queues:
	// PRE = drawn with the scene before post (reuses spritePSO); POST = drawn on the final image
	// after post (own output-format PSO, single-sample, no depth). Each stores per-texture runs.
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

	// Screen-space decals (iRender::drawDecal): box volume, surface reconstructed from the gbuf depth,
	// texture projected along the box +Z. Albedo = alpha blend, LightProjector = additive.
	RefCntAutoPtr<IPipelineState>         decalPSO, decalPSOAdd, decalPSOMod;
	RefCntAutoPtr<IShaderResourceBinding> decalSRB, decalSRBAdd, decalSRBMod;
	IShaderResourceVariable*              decalTexVar = nullptr, *decalDepthVar = nullptr;
	IShaderResourceVariable*              decalTexVarAdd = nullptr, *decalDepthVarAdd = nullptr;
	IShaderResourceVariable*              decalTexVarMod = nullptr, *decalDepthVarMod = nullptr;
	RefCntAutoPtr<IBuffer>                decalCB, decalVB;
	// Target-filtered decals: the mesh re-draw variant (depth-tested, position stream only).
	RefCntAutoPtr<IPipelineState>         decalMeshPSO, decalMeshPSOAdd, decalMeshPSOMod;
	RefCntAutoPtr<IShaderResourceBinding> decalMeshSRB, decalMeshSRBAdd, decalMeshSRBMod;
	IShaderResourceVariable*              decalMeshTexVar = nullptr, *decalMeshTexVarAdd = nullptr, *decalMeshTexVarMod = nullptr;
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
	uint64_t MakeWorldPSO(const std::string& vsSrc, const std::string& psSrc, const char* dbg,
	                      const std::string& hsSrc = std::string(), const std::string& dsSrc = std::string());
	// Builds every variant of one world pipeline into `wp` (any thread: touches only the device
	// and `wp`). `factory` = the include resolver to use (a snapshot on the builder thread).
	bool     BuildWorldPipe(WorldPipe& wp, const std::string& vsSrc, const std::string& psSrc, const char* dbg,
	                        IShaderSourceInputStreamFactory* factory = nullptr, int stages = 15 /*kStageAll*/);
	void     RebuildForMSAA();   // rebuild all sample-count-dependent pipelines + targets after `samples` changes
	// ---- pooled mesh streams -----------------------------------------------------------
	// Churn-free residency for pooled meshes (Mesh::pooled — terrain nodes): streams live as
	// RANGES inside shared arena buffers, so (re)serving/freeing a node allocates ranges and
	// never creates or destroys GPU objects. Element-granular first-fit free lists, coalescing.
	struct PoolArena
	{
		RefCntAutoPtr<IBuffer> pos, nrm, uv, col, idx;   // uv stays zero-filled (pooled meshes have none)
		uint32_t vertCap = 0, idxCap = 0;
		std::map<uint32_t, uint32_t> vFree, iFree;       // offset -> count (elements)
	};
	std::vector<std::unique_ptr<PoolArena>> meshPool;
	static bool PoolAlloc(std::map<uint32_t, uint32_t>& fm, uint32_t count, uint32_t& off);
	static void PoolFree(std::map<uint32_t, uint32_t>& fm, uint32_t off, uint32_t count);
	PoolArena* PoolAllocMesh(uint32_t verts, uint32_t inds, uint32_t& vOff, uint32_t& iOff);
	// Shared grow-only BLAS build scratch: one-shot builds (static meshes) reuse it instead of
	// creating and destroying a committed resource per build (heap churn on terrain streaming).
	RefCntAutoPtr<IBuffer> blasSharedScratch;
	Uint64 blasSharedScratchSize = 0;
	IBuffer* BlasScratchFor(Uint64 size);

	struct MeshGPU { RefCntAutoPtr<IBuffer> pos, nrm, uv; int numVerts = 0; int version = 0;
	                 RefCntAutoPtr<IBuffer> col;   // optional vertex-color stream (Mesh::colorArray) — slot 3, ATTRIB3
	                 RefCntAutoPtr<IBuffer> idx; int numIndices = 0;   // v4 indexed meshes (null = soup)
	                 // Pooled residency: streams are ranges in `arena` (pos/... above stay null).
	                 // Consumers MUST go through the accessors below — a pooled mesh's data does
	                 // NOT start at byte 0 of its buffers.
	                 PoolArena* arena = nullptr;
	                 uint32_t vOff = 0, iOff = 0;          // element offsets inside the arena
	                 IBuffer* PosBuf() const { return arena ? arena->pos.RawPtr() : pos.RawPtr(); }
	                 IBuffer* NrmBuf() const { return arena ? arena->nrm.RawPtr() : nrm.RawPtr(); }
	                 IBuffer* UVBuf()  const { return arena ? arena->uv.RawPtr()  : uv.RawPtr(); }
	                 IBuffer* ColBuf() const { return arena ? arena->col.RawPtr() : col.RawPtr(); }
	                 IBuffer* IdxBuf() const { return arena ? arena->idx.RawPtr() : idx.RawPtr(); }
	                 Uint64 PosOfs() const { return arena ? (Uint64)vOff * 12 : 0; }
	                 Uint64 NrmOfs() const { return arena ? (Uint64)vOff * 12 : 0; }
	                 Uint64 UVOfs()  const { return arena ? (Uint64)vOff * 8  : 0; }
	                 Uint64 ColOfs() const { return arena ? (Uint64)vOff * 16 : 0; }
	                 Uint64 IdxOfs() const { return arena ? (Uint64)iOff * 4  : 0; }
	                 // RT wind bend: NukeBend compute inputs + the BENT position buffer the BLAS builds over.
	                 RefCntAutoPtr<IBuffer> bendSrc, bendData, bendPivot, posBent, blasScratch;
	                 // GPU skinning: skinned INSTANCE = UAV-writable pos/nrm (pos doubles as the
	                 // draw VB + BLAS input) + previous-frame positions (TAA velocity);
	                 // SOURCE mesh = static compute inputs, built lazily on first setSkinPalette.
	                 bool skinned = false;
	                 RefCntAutoPtr<IBuffer> skinPosPrev;
	                 RefCntAutoPtr<IBuffer> skinSrcPos, skinSrcNrm, skinIdxBuf, skinWgtBuf, skinMorph;
	                 int skinMorphCount = 0; };
	std::unordered_map<Mesh*, MeshGPU>          meshCache;
	MeshGPU* GetMeshGPU(Mesh* mesh);   // get-or-build a mesh's GPU vertex buffers; re-uploads in
	                                   // place when Mesh::version changed (skinned/procedural)
	// Active LOD by approximate screen coverage (bounding-sphere diameter / camera distance),
	// against MeshLOD::screenSize thresholds. Uses the last beginCamera position.
	int  SelectLod(Mesh* mesh, const float pos[3], const float scale[3]);
	// The contiguous index (or soup vertex) range covering ALL sections of `lod`.
	void LodRange(Mesh* mesh, int lod, uint32_t& first, uint32_t& count);
	float lodCamPos[3] = { 0, 0, 0 };   // latched in beginCamera (shadow/probe passes reuse it)
	std::unordered_map<Texture*, RefCntAutoPtr<ITexture>> texCache;   // engine Texture -> GPU texture
	std::unordered_map<Texture*, int> dynTexVersion;   // dynamic textures: last uploaded dynamicVersion

	// ---- texture streaming: mip residency pool -------------------------------------------
	// A WORLD-DRAWN BC texture with a real mip chain streams: only mips [residentBase..last]
	// live on the GPU (the texture object is created at that size — normalized UVs make the
	// swap invisible to shaders). Distance feedback comes from the draw submit (StreamTouch);
	// the per-frame pump re-targets residency under the VRAM budget and rebuilds a bounded
	// number of textures per frame. UI/sprite textures never get touched -> never stream.
	struct StreamTex
	{
		int   residentBase = 0;      // current first-resident mip (0 = full res)
		float minDist = 1e30f;       // nearest use this frame (reset by the pump)
		float lastDist = 1e30f;      // nearest use of the LAST touched frame
		uint64_t lastTouch = 0;      // frameId of the last draw feedback
		int   wantLowerFrames = 0;   // consecutive frames the target sat BELOW resident detail
	};
	std::unordered_map<Texture*, StreamTex> streamTex;
	long long streamBudget = 0;      // bytes; 0 = streaming off
	long long streamResident = 0;    // resident bytes across streamed textures
	long long streamFullBytes = 0;   // full-chain bytes across streamed textures (saved = full - resident)
	void StreamTouch(Texture* t, float dist);        // per-draw feedback (world passes)
	void StreamPump();                               // per-frame residency step
	bool StreamEligible(Texture* t) const;           // BC + mip chain + not animated/RT
	static int StreamTailBase(Texture* t);           // base where max dim <= 64 (always resident)
	static int AlignedBase(Texture* t, int base);    // BC: a resource's top level must be 4-aligned (NPOT chains)
	static long long StreamBytes(Texture* t, int base);   // GPU bytes of mips [base..last]
	int  StreamDesiredBase(Texture* t, float dist) const; // distance -> first-resident mip
	RefCntAutoPtr<ITexture> CreateEngineTex(Texture* t, int baseMip);   // factored BC/RGBA8 upload
	std::unordered_map<Texture*, std::vector<RefCntAutoPtr<ITexture>>> animTex;   // GIF: one Texture2D per frame

	// ---- Fast loading 4: DirectStorage provider (NukeDiligent_Storage.cpp, D3D12 only) -------
	// Pak-resident textures (Texture::pakSource) stream into VRAM through DirectStorage: the draw
	// that needs one queues it and draws without it until it lands (the adopter publishes it to
	// texCache); the content scan prefetches every registered texture at low priority.
	struct StorTexJob; struct StorMemJob; struct StorQueue; struct DStor;
	DStor* dstor = nullptr;
	std::unordered_set<Texture*> storPendingTex;   // requested, not landed
	std::unordered_set<Texture*> storFailed;       // pak path failed: CPU decode on the next draw
	uint64_t storLanded = 0;
	void StorageInit();                                    // after device creation
	void StorageShutdown();                                // before the device dies
	void StoragePump();                                    // render thread, once per frame
	bool StorageRequestTex(Texture* t, int base, bool low);   // queue mips [base..] into a fresh GPU texture
	void StorageVerify(Texture* t, ITexture* tex, int base);  // NUKE_DSTORAGE_VERIFY readback check
	float4x4 curView, curProj;   // set in beginCamera, used in renderObject

	// Selection outline (editor): pass 1 renders the selected mesh into a mask RT; pass 2 is a
	// fullscreen edge-detect drawing a constant-pixel-thickness border around the mask.
	RefCntAutoPtr<IPipelineState>         outlineMaskPSO, outlineEdgePSO;
	RefCntAutoPtr<IShaderResourceBinding> outlineMaskSRB, outlineEdgeSRB;
	IShaderResourceVariable*              outlineEdgeMaskVar = nullptr;   // edge PS "g_Mask" (dynamic)
	RefCntAutoPtr<ITexture>               outlineMaskTex;
	ITextureView*                         outlineMaskRTV = nullptr;
	ITextureView*                         outlineMaskSRV = nullptr;
	int                                   outlineMaskW = 0, outlineMaskH = 0;
	RefCntAutoPtr<IBuffer>                outlineEdgeCB;      // texel size + thickness
	// background refraction: the opaque scene resolved/copied at beginTransparent.
	RefCntAutoPtr<ITexture>               refrTex;
	ITextureView*                         refrSRV = nullptr;   // null = no snapshot this camera
	ITextureView*                         curRTV = nullptr;   // current camera color target (outline rebind)
	ITextureView*                         curDSV = nullptr;   // ...and its depth; passes that bind their own
	                                                          // targets must RESTORE both before returning
	int                                   curRTW = 0, curRTH = 0;
	ITextureView*                         uiRTV = nullptr;    // explicit 2D target (bindRenderTarget); null = backbuffer
	Uint32                                uiTW = 0, uiTH = 0; // its size (0 = use swapchain)
	void BuildOutlinePipelines();
	void EnsureOutlineMask(int w, int h);

	// UI multi-viewport: one swap chain per detached OS window (keyed by native handle). Swap-chain
	// CREATE/RESIZE/DESTROY must NEVER happen mid-frame — doing so between passes wedges the DXGI
	// queue — so requests are queued and applied by render() at the top of the next frame.
	// Destruction parks the swap chain in the GPU trash instead of a mid-frame IdleGPU.
	std::map<void*, RefCntAutoPtr<ISwapChain>> uiVpSC;
	std::map<void*, std::pair<int, int>>       uiVpPending;   // create/resize requests (handle -> size)
	// Resize debounce per window: (last requested size, consecutive frames it held). Resizing every
	// frame during a live drag starves the main swap chain's frame-latency waitable object, so only
	// resize once the size settles; meanwhile the old buffers present stretched.
	std::map<void*, std::pair<std::pair<int, int>, int>> uiVpStable;
	std::map<void*, int> uiVpCooldown;   // frames to skip a window after a FAILED chain creation
	// A resize the driver REFUSED (the chain kept its size — e.g. Vulkan clamping to the
	// surface's current extent). Retrying it every frame is an infinite resize loop, so the
	// target is remembered and skipped until a different one is asked for.
	std::map<void*, std::pair<int, int>> uiVpRefused;
	std::map<void*, int> uiVpGrace;      // frames to skip draw+present right after a resize (diag)
	// Secondary presents are DEFERRED to after the MAIN Present: presenting mid-frame splits the
	// command stream between a preview RT's write and its SRV sampling, which the D3D12 debug layer
	// kills with ACCESS_DENIED.
	std::vector<void*> vpPresentQueue;
	size_t vpPresentRR = 0;   // round-robin cursor: ONE secondary present per frame
	uint64_t uiVpFrameNo = 0; // frame counter for the multi-window draw interleave (uiViewportRender)

	// GDI-blit host windows: a detached window = offscreen RT + staging ring + SetDIBitsToDevice.
	// Zero DXGI objects per window, so no secondary swap chains, resizes or presents. Readback is
	// async (ring of 3, mapped DO_NOT_WAIT after the main present): ~2 frames of latency, no extra flush.
	struct HostBlit
	{
		Diligent::RefCntAutoPtr<Diligent::ITexture> rt;          // offscreen UI render target
		Diligent::RefCntAutoPtr<Diligent::ITexture> staging[3];  // readback ring
		int  w = 0, h = 0;
		int  cur = 0;                  // ring slot written THIS frame
		bool valid[3] = {};            // slot holds an issued copy
		std::vector<uint8_t> scratch;  // BGRX rows for GDI
	};
	std::map<void*, HostBlit> uiHostBlits;
	std::vector<void*> uiHostBlitQueue;   // windows to blit AFTER the main Present
	void BlitHostWindows();               // map ready staging + SetDIBitsToDevice
	// Vulkan native viewports: per-window swapchain render (imgui multi-viewport). The Vulkan WSI
	// has none of the DXGI create/resize/present races; GDI blit stays the D3D fallback.
	void ViewportRenderSwapchain(void* nativeHandle, int w, int h, const NukeUIDrawData& data);
	void ApplyPendingViewportOps();                            // render() top: create/resize queued swap chains
	// Shared UI draw body (renderDrawLists + secondary viewports draw with it).
	void DrawUILists(ITextureView* rtv, Uint32 surfW, Uint32 surfH, const NukeUIDrawData& data);

	// Water lives in NukeWater.dll, driven through the native escape hatch (NukeDiligentNative.h).
	// The renderer keeps only generic capabilities: the ortho bottom capture, the shader/PSO caches,
	// the scratch targets and the hook points in beginCamera/endCamera.

	// Shader sources pushed by the engine (the renderer does NO file IO). name -> HLSL.
	std::unordered_map<std::string, std::string> shaderSrc;
	std::string shaderSource(const char* name)
	{
		// The builder thread reads while the game thread pushes — same lock as the factory.
		boost::mutex::scoped_lock l(shaderLock);
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
