#include "NukeDiligent_FrameGenShared.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cfloat>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <API/Model/Time.h>   // the frame delta the generators pace by
#include <cstdlib>
#ifdef _WIN32
#include "RenderDeviceD3D12.h"
#include "DeviceContextD3D12.h"
#include "TextureD3D12.h"
#include "SwapChainD3D12.h"      // AttachDXGISwapChain / DetachDXGISwapChain (NUKE PATCH: patches/DiligentCore-fg-swapchain.patch)
#include "CommandQueueD3D12.h"
#include <dxgi1_6.h>
#endif
#if NUKE_FFX_FG
#include <api/include/ffx_api.h>                    // AMD FidelityFX API 2.3.0: FSR 3.1 frame generation through amd_fidelityfx_framegeneration_dx12.dll
#include <api/include/ffx_api_types.h>
#include <api/include/dx12/ffx_api_dx12.h>
#include <ffx_framegeneration.h>
#include <dx12/ffx_api_framegeneration_dx12.h>
#endif
#if NUKE_XESS_FG
#include <xess_fg/xefg_swapchain.h>                 // Intel XeSS-FG through libxess_fg.dll, paced by XeLL (libxell.dll)
#include <xess_fg/xefg_swapchain_d3d12.h>
#include <xell/xell.h>
#include <xell/xell_d3d12.h>
#endif

// 4.2 frame generation: a vendor proxy swap chain sits under the Diligent one and presents a
// generated frame between every two rendered ones (or more where the hardware allows). The
// rendered frame's depth, motion and camera come from the upscale stage's prepass; the image
// before the UI (HUD-less) is copied after the world passes. Picked on the chain's upscale stage
// (g_FrameGen, g_FrameGenFrames). Three sources, all runtimes loaded by name from modules/:
//   - AMD FSR 3.1 FG: D3D12 through amd_fidelityfx_framegeneration_dx12.dll (SDK 2.3.0), Vulkan
//     through amd_fidelityfx_vk.dll (SDK 1.1.4, NukeDiligent_FrameGenVk.cpp); any GPU;
//   - NVIDIA DLSS-G through Streamline (sl.interposer.dll + sl.dlss_g / sl.reflex / sl.pcl +
//     nvngx_dlssg.dll): D3D12 (a proxied DXGI swap chain) and Vulkan (the interposer's swap-chain
//     entry points); RTX 40+, multi-frame on RTX 50;
//   - Intel XeSS-FG (libxess_fg.dll + libxell.dll): D3D12 only in the SDK; Arc, and any SM 6.4 GPU.

using namespace Diligent;
using std::cout; using std::endl;
using namespace nukediligent;

namespace nukediligent {

const char* FrameGenName(int kind) { return kind == FG_FSR ? "FSR 3.1 FG" : kind == FG_DLSS ? "DLSS-G" : kind == FG_XESS ? "XeSS-FG" : "none"; }

#if defined(_WIN32) && NUKE_STREAMLINE
static SlFns g_sl; static bool g_slTried = false;
SlFns& Sl()
{
	if (!g_slTried)
	{
		g_slTried = true;
		g_sl.dll = LoadVendorDll(L"sl.interposer.dll");
		if (g_sl.dll)
		{
#define NUKE_SL(fn, sym) g_sl.fn = (decltype(g_sl.fn))GetProcAddress(g_sl.dll, sym)
			NUKE_SL(Init, "slInit"); NUKE_SL(Shutdown, "slShutdown"); NUKE_SL(SetD3DDevice, "slSetD3DDevice");
			NUKE_SL(UpgradeInterface, "slUpgradeInterface"); NUKE_SL(IsFeatureSupported, "slIsFeatureSupported");
			NUKE_SL(SetFeatureLoaded, "slSetFeatureLoaded"); NUKE_SL(GetFeatureFunction, "slGetFeatureFunction");
			NUKE_SL(SetTagForFrame, "slSetTagForFrame"); NUKE_SL(SetConstants, "slSetConstants");
			NUKE_SL(GetNewFrameToken, "slGetNewFrameToken"); NUKE_SL(FreeResources, "slFreeResources");
#undef NUKE_SL
		}
	}
	return g_sl;
}
static void SlLog(sl::LogType type, const char* msg)
{
	if (!msg || type == sl::LogType::eInfo) return;
	std::string s(msg); while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
	cout << "[Streamline]\t" << s << endl;
}
sl::float4x4 SlMat(const float4x4& m)
{
	sl::float4x4 r;
	for (int i = 0; i < 4; ++i) r.setRow(i, sl::float4(m.m[i][0], m.m[i][1], m.m[i][2], m.m[i][3]));
	return r;
}
const sl::ViewportHandle kSlViewport(0u);
void SlMark(sl::PCLMarker m, void* token) { if (Sl().PCLSetMarker && token) Sl().PCLSetMarker(m, *(const sl::FrameToken*)token); }

bool SlInitOnce(NukeDiligent::Impl& d, bool vulkan, void* d3dDevice)
{
	if (d.fg.slInit) return d.fg.slReady;
	d.fg.slInit = true;
	if (!Sl().ok()) { cout << "[NukeDiligent]\tStreamline: sl.interposer.dll not next to NukeRenderDiligent.dll" << endl; return false; }
	static std::wstring dir = ModuleDir();
	static const wchar_t* paths[] = { dir.c_str() };
	static sl::Feature feats[3]; static uint32_t nFeats = 0;
	{   // NUKE_SL_FEATURES=dlssg,reflex,pcl narrows the plugin set (diagnostics); default = all three
		const char* fs = std::getenv("NUKE_SL_FEATURES");
		const std::string f = fs ? fs : "dlssg,reflex,pcl";
		nFeats = 0;
		if (f.find("dlssg") != std::string::npos) feats[nFeats++] = sl::kFeatureDLSS_G;
		if (f.find("reflex") != std::string::npos) feats[nFeats++] = sl::kFeatureReflex;
		if (f.find("pcl") != std::string::npos) feats[nFeats++] = sl::kFeaturePCL;
	}
	const char* verbose = std::getenv("NUKE_SL_VERBOSE");   // NUKE_SL_VERBOSE=1: Streamline's full log into modules/sl.log
	sl::Preferences pref{};
	pref.showConsole = false;
	pref.logLevel = (verbose && *verbose == '1') ? sl::LogLevel::eVerbose : sl::LogLevel::eDefault;
	pref.pathsToPlugins = paths; pref.numPathsToPlugins = 1;
	if (verbose && *verbose == '1') pref.pathToLogsAndData = dir.c_str();
	pref.logMessageCallback = SlLog;
	// Manual hooking (Diligent owns the device / queue; only the swap chain goes through the
	// proxies), frame-based tags; no over-the-air plugin downloads at run time.
	pref.flags = sl::PreferenceFlags::eUseManualHooking | sl::PreferenceFlags::eUseFrameBasedResourceTagging | sl::PreferenceFlags::eDisableCLStateTracking;
	pref.featuresToLoad = feats; pref.numFeaturesToLoad = nFeats;
	pref.applicationId = 231313132;
	pref.engine = sl::EngineType::eCustom;
	pref.engineVersion = "NukeEngine";
	pref.projectId = "750fed3a-efba-42ba-801b-22d4cbad9148";
	pref.renderAPI = vulkan ? sl::RenderAPI::eVulkan : sl::RenderAPI::eD3D12;
	sl::Result r = Sl().Init(pref, sl::kSDKVersion);
	if (r != sl::Result::eOk) { cout << "[NukeDiligent]\tStreamline init failed: " << sl::getResultAsStr(r) << endl; return false; }
	if (vulkan) return true;   // the feature functions need the device: SlFeatures() after it exists
	if (d3dDevice)
	{
		Sl().SetD3DDevice(d3dDevice);
		LUID luid = ((ID3D12Device*)d3dDevice)->GetAdapterLuid();
		sl::AdapterInfo ai{};
		ai.deviceLUID = (uint8_t*)&luid; ai.deviceLUIDSizeInBytes = sizeof(LUID);
		r = Sl().IsFeatureSupported(sl::kFeatureDLSS_G, ai);
		if (r != sl::Result::eOk) { cout << "[NukeDiligent]\tDLSS-G not supported here: " << sl::getResultAsStr(r) << endl; return false; }
	}
	return SlFeatures(d);
}

// The DLSS-G / Reflex / PCL feature functions (the plugins answer only once a device exists).
bool SlFeatures(NukeDiligent::Impl& d)
{
	if (d.fg.slReady) return true;
	void* f = nullptr;
	Sl().GetFeatureFunction(sl::kFeatureDLSS_G, "slDLSSGSetOptions", f); Sl().DLSSGSetOptions = (PFun_slDLSSGSetOptions*)f; f = nullptr;
	Sl().GetFeatureFunction(sl::kFeatureDLSS_G, "slDLSSGGetState", f);   Sl().DLSSGGetState = (PFun_slDLSSGGetState*)f; f = nullptr;
	Sl().GetFeatureFunction(sl::kFeatureReflex, "slReflexSetOptions", f); Sl().ReflexSetOptions = (PFun_slReflexSetOptions*)f; f = nullptr;
	Sl().GetFeatureFunction(sl::kFeatureReflex, "slReflexSleep", f);      Sl().ReflexSleep = (PFun_slReflexSleep*)f; f = nullptr;
	Sl().GetFeatureFunction(sl::kFeaturePCL, "slPCLSetMarker", f);        Sl().PCLSetMarker = (PFun_slPCLSetMarker*)f; f = nullptr;
	if (Sl().ReflexSetOptions)   // DLSS-G requires Reflex; the markers ride the PCL plugin
	{
		sl::ReflexOptions ro{};
		ro.mode = sl::ReflexMode::eLowLatency;
		Sl().ReflexSetOptions(ro);
	}
	d.fg.slReady = Sl().DLSSGSetOptions && Sl().DLSSGGetState && Sl().PCLSetMarker && Sl().ReflexSleep;
	if (!d.fg.slReady) cout << "[NukeDiligent]\tDLSS-G: feature functions missing (sl.dlss_g / sl.reflex / sl.pcl plugins?)" << endl;
	return d.fg.slReady;
}
#endif

}  // namespace nukediligent

namespace {

#ifdef _WIN32
#if NUKE_FFX_FG
struct FfxFgFns
{
	HMODULE dll = nullptr;
	PfnFfxCreateContext  CreateContext = nullptr;
	PfnFfxDestroyContext DestroyContext = nullptr;
	PfnFfxConfigure      Configure = nullptr;
	PfnFfxQuery          Query = nullptr;
	PfnFfxDispatch       Dispatch = nullptr;
	bool ok() const { return dll && CreateContext && DestroyContext && Configure && Query && Dispatch; }
};
FfxFgFns g_ffxFg; bool g_ffxFgTried = false;
const FfxFgFns& FfxFg()
{
	if (!g_ffxFgTried)
	{
		g_ffxFgTried = true;
		g_ffxFg.dll = LoadVendorDll(L"amd_fidelityfx_framegeneration_dx12.dll");
		if (g_ffxFg.dll)
		{
			g_ffxFg.CreateContext  = (PfnFfxCreateContext)GetProcAddress(g_ffxFg.dll, "ffxCreateContext");
			g_ffxFg.DestroyContext = (PfnFfxDestroyContext)GetProcAddress(g_ffxFg.dll, "ffxDestroyContext");
			g_ffxFg.Configure      = (PfnFfxConfigure)GetProcAddress(g_ffxFg.dll, "ffxConfigure");
			g_ffxFg.Query          = (PfnFfxQuery)GetProcAddress(g_ffxFg.dll, "ffxQuery");
			g_ffxFg.Dispatch       = (PfnFfxDispatch)GetProcAddress(g_ffxFg.dll, "ffxDispatch");
		}
	}
	return g_ffxFg;
}
// The swap chain calls back for every generated frame: the FG effect context runs the interpolation.
ffxReturnCode_t FfxFgDispatchCb(ffxDispatchDescFrameGeneration* params, void* user)
{
	return FfxFg().Dispatch((ffxContext*)user, &params->header);
}
uint32_t FfxSurfaceFormat(TEXTURE_FORMAT f)
{
	switch (f)
	{
	case TEX_FORMAT_RGBA8_UNORM:      return FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM;
	case TEX_FORMAT_RGBA8_UNORM_SRGB: return FFX_API_SURFACE_FORMAT_R8G8B8A8_SRGB;
	case TEX_FORMAT_BGRA8_UNORM:      return FFX_API_SURFACE_FORMAT_B8G8R8A8_UNORM;
	case TEX_FORMAT_BGRA8_UNORM_SRGB: return FFX_API_SURFACE_FORMAT_B8G8R8A8_SRGB;
	case TEX_FORMAT_RGB10A2_UNORM:    return FFX_API_SURFACE_FORMAT_R10G10B10A2_UNORM;
	default:                          return FFX_API_SURFACE_FORMAT_UNKNOWN;
	}
}
#endif

#if NUKE_XESS_FG
struct XeFgFns
{
	HMODULE dll = nullptr, xell = nullptr;
	decltype(&xefgSwapChainD3D12CreateContext)          D3D12CreateContext = nullptr;
	decltype(&xefgSwapChainD3D12InitFromSwapChainDesc)  D3D12InitFromSwapChainDesc = nullptr;
	decltype(&xefgSwapChainD3D12GetSwapChainPtr)        D3D12GetSwapChainPtr = nullptr;
	decltype(&xefgSwapChainD3D12TagFrameResource)       D3D12TagFrameResource = nullptr;
	decltype(&xefgSwapChainTagFrameConstants)           TagFrameConstants = nullptr;
	decltype(&xefgSwapChainSetEnabled)                  SetEnabled = nullptr;
	decltype(&xefgSwapChainSetPresentId)                SetPresentId = nullptr;
	decltype(&xefgSwapChainGetLastPresentStatus)        GetLastPresentStatus = nullptr;
	decltype(&xefgSwapChainSetLoggingCallback)          SetLoggingCallback = nullptr;
	decltype(&xefgSwapChainDestroy)                     Destroy = nullptr;
	decltype(&xefgSwapChainSetLatencyReduction)         SetLatencyReduction = nullptr;
	decltype(&xefgSwapChainGetProperties)               GetProperties = nullptr;
	decltype(&xefgSwapChainSetNumInterpolatedFrames)    SetNumInterpolatedFrames = nullptr;
	decltype(&xefgSwapChainGetVersion)                  GetVersion = nullptr;
	decltype(&xellD3D12CreateContext)                   XellD3D12CreateContext = nullptr;
	decltype(&xellDestroyContext)                       XellDestroyContext = nullptr;
	decltype(&xellSetSleepMode)                         XellSetSleepMode = nullptr;
	decltype(&xellSleep)                                XellSleep = nullptr;
	decltype(&xellAddMarkerData)                        XellAddMarkerData = nullptr;
	decltype(&xellSetLoggingCallback)                   XellSetLoggingCallback = nullptr;
	bool ok() const
	{
		return dll && xell && D3D12CreateContext && D3D12InitFromSwapChainDesc && D3D12GetSwapChainPtr && D3D12TagFrameResource
		    && TagFrameConstants && SetEnabled && SetPresentId && GetLastPresentStatus && Destroy && SetLatencyReduction && GetProperties
		    && SetNumInterpolatedFrames && XellD3D12CreateContext && XellDestroyContext && XellSetSleepMode && XellSleep && XellAddMarkerData;
	}
};
XeFgFns g_xefg; bool g_xefgTried = false;
const XeFgFns& XeFg()
{
	if (!g_xefgTried)
	{
		g_xefgTried = true;
		g_xefg.xell = LoadVendorDll(L"libxell.dll");   // XeSS-FG's own dependency: it must resolve from modules/ first
		g_xefg.dll  = LoadVendorDll(L"libxess_fg.dll");
		if (g_xefg.dll)
		{
#define NUKE_XEFG(fn, sym) g_xefg.fn = (decltype(g_xefg.fn))GetProcAddress(g_xefg.dll, sym)
			NUKE_XEFG(D3D12CreateContext, "xefgSwapChainD3D12CreateContext");
			NUKE_XEFG(D3D12InitFromSwapChainDesc, "xefgSwapChainD3D12InitFromSwapChainDesc");
			NUKE_XEFG(D3D12GetSwapChainPtr, "xefgSwapChainD3D12GetSwapChainPtr");
			NUKE_XEFG(D3D12TagFrameResource, "xefgSwapChainD3D12TagFrameResource");
			NUKE_XEFG(TagFrameConstants, "xefgSwapChainTagFrameConstants");
			NUKE_XEFG(SetEnabled, "xefgSwapChainSetEnabled");
			NUKE_XEFG(SetPresentId, "xefgSwapChainSetPresentId");
			NUKE_XEFG(GetLastPresentStatus, "xefgSwapChainGetLastPresentStatus");
			NUKE_XEFG(SetLoggingCallback, "xefgSwapChainSetLoggingCallback");
			NUKE_XEFG(Destroy, "xefgSwapChainDestroy");
			NUKE_XEFG(SetLatencyReduction, "xefgSwapChainSetLatencyReduction");
			NUKE_XEFG(GetProperties, "xefgSwapChainGetProperties");
			NUKE_XEFG(SetNumInterpolatedFrames, "xefgSwapChainSetNumInterpolatedFrames");
			NUKE_XEFG(GetVersion, "xefgSwapChainGetVersion");
#undef NUKE_XEFG
		}
		if (g_xefg.xell)
		{
#define NUKE_XELL(fn, sym) g_xefg.fn = (decltype(g_xefg.fn))GetProcAddress(g_xefg.xell, sym)
			NUKE_XELL(XellD3D12CreateContext, "xellD3D12CreateContext");
			NUKE_XELL(XellDestroyContext, "xellDestroyContext");
			NUKE_XELL(XellSetSleepMode, "xellSetSleepMode");
			NUKE_XELL(XellSleep, "xellSleep");
			NUKE_XELL(XellAddMarkerData, "xellAddMarkerData");
			NUKE_XELL(XellSetLoggingCallback, "xellSetLoggingCallback");
#undef NUKE_XELL
		}
	}
	return g_xefg;
}
void XeFgLog(const char* msg, xefg_swapchain_logging_level_t lvl, void*) { if (msg && lvl >= XEFG_SWAPCHAIN_LOGGING_LEVEL_WARNING) cout << "[XeSS-FG]\t" << msg << endl; }
void XellLog(const char* msg, xell_logging_level_t lvl) { if (msg && lvl >= XELL_LOGGING_LEVEL_WARNING) cout << "[XeLL]\t" << msg << endl; }
#endif
#endif   // _WIN32

double NowSeconds()
{
	using namespace std::chrono;
	return duration<double>(steady_clock::now().time_since_epoch()).count();
}

}  // namespace

// The upscale stage's g_FrameGen / g_FrameGenFrames: the stage's mode when generation is asked
// for, -1 otherwise. `frames` = generated frames per rendered one (the hardware caps it later).
int NukeDiligent::Impl::FrameGenWanted(int& frames) const
{
	frames = 0;
	for (const auto& cs : postChain)
	{
		auto pit = postPipes.find(cs.pipeline);
		if (pit == postPipes.end() || !pit->second.isUpscale) continue;
		const bool on = cs.params.size() > 3 && cs.params[3] >= 0.5f;
		if (!on) return -1;
		frames = cs.params.size() > 4 ? (int)(cs.params[4] + 0.5f) : 1;
		if (frames < 1) frames = 1;
		return cs.params.size() > 0 ? (int)(cs.params[0] + 0.5f) : UP_AUTO;
	}
	return -1;
}

// The generator that goes with the stage's mode, skipping the ones that refused this session:
// the mode's own vendor first, then the adapter's, then the cross-vendor ones (FSR FG runs on
// any GPU, XeSS-FG on any SM 6.4 one under D3D12). On Vulkan: FSR FG (AMD's Vulkan runtime) and
// DLSS-G (Streamline's Vulkan proxies, live only on an NVIDIA adapter); no XeSS-FG there.
int NukeDiligent::Impl::ResolveFrameGenKind(int mode) const
{
	int order[3] = { FG_FSR, FG_XESS, FG_DLSS };
	const ADAPTER_VENDOR vendor = device ? device->GetAdapterInfo().Vendor : ADAPTER_VENDOR_UNKNOWN;
	int first = mode == UP_DLSS ? FG_DLSS : mode == UP_FSR ? FG_FSR : mode == UP_XESS ? FG_XESS
	          : vendor == ADAPTER_VENDOR_NVIDIA ? FG_DLSS : vendor == ADAPTER_VENDOR_INTEL ? FG_XESS : FG_FSR;
	int n = 0; int list[3];
	list[n++] = first;
	for (int k : order) if (k != first) list[n++] = k;
	for (int i = 0; i < n; ++i)
	{
		const int k = list[i];
		if (fg.failed & (1 << k)) continue;
		if (useVulkan && k == FG_XESS) continue;
		if (useVulkan && k == FG_DLSS && !fgVk.slProxied) continue;
		return k;
	}
	return FG_NONE;
}

// The generators this build / backend / adapter can attach, the ones that refused this session
// excluded; bits in the engine's UpscaleMode numbering (2 DLSS, 3 FSR, 4 XeSS).
unsigned NukeDiligent::Impl::FrameGenOffered() const
{
	unsigned bits = 0;
#ifdef _WIN32
	if (transparent || !(useD3D12 || useVulkan)) return 0;
	const ADAPTER_VENDOR vendor = device ? device->GetAdapterInfo().Vendor : ADAPTER_VENDOR_UNKNOWN;
#if NUKE_FFX_FG
	if (useD3D12 && !(fg.failed & (1 << FG_FSR)) && FfxFg().ok()) bits |= 1u << (UP_FSR + 1);
#endif
#if NUKE_FFX_VK
	if (useVulkan && !(fg.failed & (1 << FG_FSR)) && nukediligent::FfxVkAvailable()) bits |= 1u << (UP_FSR + 1);
#endif
#if NUKE_XESS_FG
	if (useD3D12 && !(fg.failed & (1 << FG_XESS)) && XeFg().ok()) bits |= 1u << (UP_XESS + 1);
#endif
#if NUKE_STREAMLINE
	if (vendor == ADAPTER_VENDOR_NVIDIA && !(fg.failed & (1 << FG_DLSS)) && (useVulkan ? fgVk.slProxied : Sl().ok())) bits |= 1u << (UP_DLSS + 1);
#endif
	(void)vendor;
#endif
	return bits;
}

// Once per frame before the back buffer is fetched: the stage's wish against what is attached.
void NukeDiligent::Impl::FrameGenApply()
{
	int frames = 0;
	const int mode = FrameGenWanted(frames);
	const bool camera = frameId - fg.lastCapFrame <= 2;   // a back-buffer camera rendered last frame
	const bool want = mode >= 0 && (useD3D12 || useVulkan) && !transparent && (camera || (fg.attached && frameId - fg.lastCapFrame < 300));
	if (!want)
	{
		if (fg.attached) FrameGenDetach(mode < 0 ? "switched off" : "no back-buffer camera");
		fg.mode = -1;
		return;
	}
	if (fg.attached && fg.mode == mode && fg.frames == frames) return;
	const int kind = ResolveFrameGenKind(mode);
	if (fg.attached && kind != fg.kind) FrameGenDetach("mode changed");
	fg.mode = mode; fg.frames = frames;
	if (kind == FG_NONE)
	{
		if (!fg.saidNone) { fg.saidNone = true; cout << "[NukeDiligent]\tframe generation: no generator available on this GPU / backend / build (see above)" << endl; }
		return;
	}
	if (!fg.attached)
	{
		if (!FrameGenAttach(kind, frames))
		{
			fg.failed |= 1 << kind;
			const int next = ResolveFrameGenKind(mode);
			if (next != FG_NONE && next != kind) FrameGenAttach(next, frames);
			if (!fg.attached && next != FG_NONE) fg.failed |= 1 << next;
		}
		return;
	}
	// Same generator, a different frame count: retune in place where the vendor allows it.
#ifdef _WIN32
#if NUKE_XESS_FG
	if (fg.kind == FG_XESS && fg.xefg)
	{
		xefg_swapchain_properties_t props{};
		uint32_t cap = 1;
		if (XeFg().GetProperties((xefg_swapchain_handle_t)fg.xefg, &props) == XEFG_SWAPCHAIN_RESULT_SUCCESS && props.maxSupportedInterpolations > 0) cap = props.maxSupportedInterpolations;
		fg.framesLive = (int)std::min<uint32_t>((uint32_t)frames, cap);
		XeFg().SetNumInterpolatedFrames((xefg_swapchain_handle_t)fg.xefg, (uint32_t)fg.framesLive);
		cout << "[NukeDiligent]\tframe generation: XeSS-FG x" << (fg.framesLive + 1) << endl;
	}
#endif
#if NUKE_STREAMLINE
	if (fg.kind == FG_DLSS && fg.slReady && Sl().DLSSGSetOptions && Sl().DLSSGGetState)
	{
		sl::DLSSGState st{};
		uint32_t cap = 1;
		if (Sl().DLSSGGetState(kSlViewport, st, nullptr) == sl::Result::eOk && st.numFramesToGenerateMax > 0) cap = st.numFramesToGenerateMax;
		fg.framesLive = (int)std::min<uint32_t>((uint32_t)frames, cap);
		sl::DLSSGOptions o{};
		o.mode = fg.live ? sl::DLSSGMode::eOn : sl::DLSSGMode::eOff;
		o.numFramesToGenerate = (uint32_t)fg.framesLive;
		o.flags = sl::DLSSGFlags::eRetainResourcesWhenOff;
		Sl().DLSSGSetOptions(kSlViewport, o);
		cout << "[NukeDiligent]\tframe generation: DLSS-G x" << (fg.framesLive + 1) << endl;
	}
#endif
#endif
	if (fg.kind == FG_FSR && frames != fg.framesLive) cout << "[NukeDiligent]\tframe generation: FSR 3.1 FG generates one frame per rendered frame (x2); " << frames << " asked" << endl;
}

// Replace the swap chain under the Diligent one with the vendor's proxy for the same window.
bool NukeDiligent::Impl::FrameGenAttach(int kind, int frames)
{
#ifdef _WIN32
	if (!swapChain || !device || !context) return false;
	auto attached = [&](int W, int H)
	{
		fg.kind = kind; fg.attached = true; fg.attachFrame = frameId;
		fg.live = kind != FG_DLSS;   // FSR / XeSS start enabled; DLSS-G's options go with its first frame
		fg.frameIndex = 0; fg.reset = true; fg.hadLast = false; fg.captured = false; fg.lastLog = frameId;
		fg.presentBase = 0; fg.ratio = 0.0; fg.ratioBase = 0; fg.ratioFrame = 0;
		cout << "[NukeDiligent]\tframe generation: " << FrameGenName(kind) << " x" << (fg.framesLive + 1) << " attached (" << W << "x" << H << ", " << (useVulkan ? "Vulkan" : "D3D12") << ")" << endl;
	};
	if (useVulkan)
	{
		fg.framesLive = 1;
		if (!FrameGenVkAttach(kind, frames)) return false;
		attached((int)swapChain->GetDesc().Width, (int)swapChain->GetDesc().Height);
		return true;
	}
	if (!useD3D12) return false;
	RefCntAutoPtr<ISwapChainD3D12> sc12(swapChain, IID_SwapChainD3D12);
	RefCntAutoPtr<IRenderDeviceD3D12> d12(device, IID_RenderDeviceD3D12);
	if (!sc12 || !d12) return false;
	IDXGISwapChain* cur = sc12->GetDXGISwapChain();
	if (!cur) return false;
	IDXGISwapChain1* cur1 = nullptr;
	if (FAILED(cur->QueryInterface(__uuidof(IDXGISwapChain1), (void**)&cur1)) || !cur1) return false;
	DXGI_SWAP_CHAIN_DESC1 d1{}; DXGI_SWAP_CHAIN_FULLSCREEN_DESC fs{}; HWND hwnd = nullptr; IDXGIFactory2* factory = nullptr;
	cur1->GetDesc1(&d1); cur1->GetFullscreenDesc(&fs); cur1->GetHwnd(&hwnd);
	cur1->GetParent(__uuidof(IDXGIFactory2), (void**)&factory);
	cur1->Release();
	if (!hwnd || !factory || !fs.Windowed)
	{
		cout << "[NukeDiligent]\tframe generation: " << (hwnd ? "exclusive fullscreen" : "no window") << " - the generators need a windowed HWND swap chain" << endl;
		if (factory) factory->Release();
		return false;
	}
	d1.Flags &= ~DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;   // the vendor pacer owns the timing
	ID3D12Device* dev = d12->GetD3D12Device();
	ID3D12CommandQueue* queue = nullptr;
	{
		ICommandQueue* q = context->LockCommandQueue();
		RefCntAutoPtr<ICommandQueueD3D12> q12(q, IID_CommandQueueD3D12);
		if (q12) queue = q12->GetD3D12CommandQueue();
		context->UnlockCommandQueue();   // the detach below idles the GPU through this lock
	}
	if (!dev || !queue) { factory->Release(); return false; }
	const TEXTURE_FORMAT bbFmt = swapChain->GetDesc().ColorBufferFormat;
	const int W = (int)d1.Width, H = (int)d1.Height;

	context->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE);
	context->Flush();
	device->IdleGPU();
	sc12->DetachDXGISwapChain();   // DXGI allows one swap chain per window: the old one is gone first
	IDXGISwapChain* proxy = nullptr;
	std::string detail;
	fg.framesLive = 1;

#if NUKE_FFX_FG
	if (kind == FG_FSR)
	{
		if (!FfxFg().ok()) detail = "amd_fidelityfx_framegeneration_dx12.dll not next to NukeRenderDiligent.dll";
		else
		{
			ffxCreateBackendDX12Desc be{};
			be.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;
			be.device = dev;
			ffxCreateContextDescFrameGenerationVersion ver{};
			ver.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION_VERSION;
			ver.version = FFX_FRAMEGENERATION_VERSION;
			ver.header.pNext = &be.header;
			ffxCreateContextDescFrameGenerationHudless hl{};
			hl.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION_HUDLESS;
			hl.hudlessBackBufferFormat = FfxSurfaceFormat(bbFmt);
			hl.header.pNext = &ver.header;
			ffxCreateContextDescFrameGeneration cd{};
			cd.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
			cd.header.pNext = &hl.header;
			cd.flags = hdr10Active ? FFX_FRAMEGENERATION_ENABLE_HIGH_DYNAMIC_RANGE : 0u;
			cd.displaySize = { (uint32_t)W, (uint32_t)H };
			cd.maxRenderSize = { (uint32_t)W, (uint32_t)H };
			cd.backBufferFormat = FfxSurfaceFormat(bbFmt);
			ffxReturnCode_t rc = FfxFg().CreateContext(&fg.ffx, &cd.header, nullptr);
			if (rc != FFX_API_RETURN_OK || !fg.ffx) { detail = "FG context creation failed: " + std::to_string(rc); fg.ffx = nullptr; }
			else
			{
				IDXGISwapChain4* sc4 = nullptr;
				ffxCreateContextDescFrameGenerationSwapChainVersionDX12 sv{};
				sv.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_VERSION_DX12;
				sv.version = FFX_FRAMEGENERATION_SWAPCHAIN_DX12_VERSION;
				ffxCreateContextDescFrameGenerationSwapChainForHwndDX12 sc{};
				sc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_FOR_HWND_DX12;
				sc.header.pNext = &sv.header;
				sc.swapchain = &sc4; sc.hwnd = hwnd; sc.desc = &d1; sc.fullscreenDesc = &fs; sc.dxgiFactory = factory; sc.gameQueue = queue;
				rc = FfxFg().CreateContext(&fg.ffxSc, &sc.header, nullptr);
				if (rc != FFX_API_RETURN_OK || !sc4) { detail = "FG swap chain creation failed: " + std::to_string(rc); fg.ffxSc = nullptr; FfxFg().DestroyContext(&fg.ffx, nullptr); fg.ffx = nullptr; }
				else proxy = sc4;   // the creation reference is ours
			}
		}
	}
#endif
#if NUKE_XESS_FG
	if (kind == FG_XESS)
	{
		if (!XeFg().ok()) detail = "libxess_fg.dll / libxell.dll not next to NukeRenderDiligent.dll";
		else
		{
			xefg_swapchain_handle_t h = nullptr;
			xefg_swapchain_result_t r = XeFg().D3D12CreateContext(dev, &h);
			if (r != XEFG_SWAPCHAIN_RESULT_SUCCESS || !h) detail = "XeSS-FG context refused: " + std::to_string((int)r) + " (needs SM 6.4)";
			else
			{
				if (XeFg().SetLoggingCallback) XeFg().SetLoggingCallback(h, XEFG_SWAPCHAIN_LOGGING_LEVEL_WARNING, XeFgLog, nullptr);
				xell_context_handle_t xl = nullptr;
				if (XeFg().XellD3D12CreateContext(dev, &xl) != XELL_RESULT_SUCCESS || !xl) { detail = "XeLL context refused"; XeFg().Destroy(h); }
				else
				{
					if (XeFg().XellSetLoggingCallback) XeFg().XellSetLoggingCallback(xl, XELL_LOGGING_LEVEL_WARNING, XellLog);
					xell_sleep_params_t sp{};
					sp.bLowLatencyMode = 1;
					XeFg().XellSetSleepMode(xl, &sp);
					XeFg().SetLatencyReduction(h, xl);
					xefg_swapchain_d3d12_init_params_t ip{};
					ip.maxInterpolatedFrames = XEFG_SWAPCHAIN_USE_MAX_SUPPORTED_INTERPOLATED_FRAMES;
					ip.uiMode = XEFG_SWAPCHAIN_UI_MODE_AUTO;
					r = XeFg().D3D12InitFromSwapChainDesc(h, hwnd, &d1, &fs, queue, factory, &ip);
					IDXGISwapChain4* sc4 = nullptr;
					if (r != XEFG_SWAPCHAIN_RESULT_SUCCESS) detail = "XeSS-FG swap chain init failed: " + std::to_string((int)r);
					else if (XeFg().D3D12GetSwapChainPtr(h, __uuidof(IDXGISwapChain4), (void**)&sc4) != XEFG_SWAPCHAIN_RESULT_SUCCESS || !sc4) detail = "XeSS-FG gave no swap chain";
					if (!sc4) { XeFg().Destroy(h); XeFg().XellDestroyContext(xl); }
					else
					{
						xefg_swapchain_properties_t props{};
						uint32_t cap = 1;
						if (XeFg().GetProperties(h, &props) == XEFG_SWAPCHAIN_RESULT_SUCCESS && props.maxSupportedInterpolations > 0) cap = props.maxSupportedInterpolations;
						fg.framesLive = (int)std::min<uint32_t>((uint32_t)frames, cap);
						XeFg().SetNumInterpolatedFrames(h, (uint32_t)fg.framesLive);
						XeFg().SetEnabled(h, 1);
						fg.xefg = h; fg.xell = xl; proxy = sc4;
					}
				}
			}
		}
	}
#endif
#if NUKE_STREAMLINE
	if (kind == FG_DLSS)
	{
		if (!SlInitOnce(*this, false, dev)) detail = "Streamline / DLSS-G unavailable (see above)";
		else
		{
			// The swap chain must come from the Streamline factory proxy: that is where DLSS-G
			// hooks Present, GetBuffer and the back-buffer index.
			IDXGIFactory2* pf = factory; pf->AddRef();
			sl::Result r = Sl().UpgradeInterface((void**)&pf);
			IDXGISwapChain1* sc1 = nullptr;
			if (r != sl::Result::eOk) { detail = std::string("factory proxy refused: ") + sl::getResultAsStr(r); pf->Release(); }
			else
			{
				const HRESULT hr = pf->CreateSwapChainForHwnd(queue, hwnd, &d1, &fs, nullptr, &sc1);
				pf->Release();
				if (FAILED(hr) || !sc1) { detail = "proxied swap chain creation failed: 0x" + std::to_string((unsigned)hr); sc1 = nullptr; }
			}
			if (sc1)
			{
				sl::DLSSGState st{};
				uint32_t cap = 1;
				if (Sl().DLSSGGetState(kSlViewport, st, nullptr) == sl::Result::eOk && st.numFramesToGenerateMax > 0) cap = st.numFramesToGenerateMax;
				fg.framesLive = (int)std::min<uint32_t>((uint32_t)frames, cap);
				proxy = sc1;   // generation switches on with the first frame that has inputs (FrameGenBeforePresent)
			}
		}
	}
#endif
	factory->Release();
	if (!proxy)
	{
		if (detail.empty()) detail = "not built into this renderer";
		cout << "[NukeDiligent]\tframe generation: " << FrameGenName(kind) << " unavailable - " << detail << endl;
		sc12->AttachDXGISwapChain(nullptr);   // Diligent's own chain again
		if (hdrOutput) SetupHDROutput();
		return false;
	}
	sc12->AttachDXGISwapChain(proxy);
	if (hdrOutput) SetupHDROutput();   // the colour space goes to the new DXGI object
	fg.proxy = proxy;
	attached(W, H);
	return true;
#else
	(void)kind; (void)frames;
	return false;
#endif
}

// Back to Diligent's own swap chain; the vendor contexts go with the proxy.
void NukeDiligent::Impl::FrameGenDetach(const char* why)
{
	if (!fg.attached) return;
#ifdef _WIN32
	context->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE);
	context->Flush();
	device->IdleGPU();
#if NUKE_STREAMLINE
	if (fg.kind == FG_DLSS && fg.slReady && Sl().DLSSGSetOptions)
	{
		sl::DLSSGOptions o{};
		o.mode = sl::DLSSGMode::eOff;   // presents run synchronously again before the chain goes
		Sl().DLSSGSetOptions(kSlViewport, o);
	}
#endif
	if (useVulkan) FrameGenVkDetach();
	else
	{
		RefCntAutoPtr<ISwapChainD3D12> sc12(swapChain, IID_SwapChainD3D12);
#if NUKE_XESS_FG
		if (fg.kind == FG_XESS && fg.xefg) XeFg().SetEnabled((xefg_swapchain_handle_t)fg.xefg, 0);
#endif
		if (sc12) sc12->DetachDXGISwapChain();
		if (fg.proxy) { ((IDXGISwapChain*)fg.proxy)->Release(); fg.proxy = nullptr; }
#if NUKE_FFX_FG
		if (fg.ffxSc) { FfxFg().DestroyContext(&fg.ffxSc, nullptr); fg.ffxSc = nullptr; }
		if (fg.ffx)   { FfxFg().DestroyContext(&fg.ffx, nullptr);   fg.ffx = nullptr; }
#endif
#if NUKE_XESS_FG
		if (fg.xefg) { XeFg().Destroy((xefg_swapchain_handle_t)fg.xefg); fg.xefg = nullptr; }
		if (fg.xell) { XeFg().XellDestroyContext((xell_context_handle_t)fg.xell); fg.xell = nullptr; }
#endif
		if (sc12) sc12->AttachDXGISwapChain(nullptr);
		if (hdrOutput) SetupHDROutput();
	}
#if NUKE_STREAMLINE
	if (fg.kind == FG_DLSS && Sl().FreeResources) Sl().FreeResources(sl::kFeatureDLSS_G, kSlViewport);
	fg.slToken = nullptr;
#endif
	Trash(fg.hudless); fg.hudless.Release();
	fg.depth.Release(); fg.vel.Release();
	cout << "[NukeDiligent]\tframe generation: " << FrameGenName(fg.kind) << " released (" << why << ")" << endl;
#else
	(void)why;
#endif
	fg.attached = false; fg.live = false; fg.kind = FG_NONE; fg.captured = false; fg.ratio = 0.0;
}

// Frame start: the vendor's frame id, its latency sleep and the simulation / submit markers. The
// engine ticks inside the render callbacks, so the simulation and submission phases overlap here.
void NukeDiligent::Impl::FrameGenFrameStart()
{
	if (!fg.attached) return;
#ifdef _WIN32
#if NUKE_STREAMLINE
	if (fg.kind == FG_DLSS)
	{
		sl::FrameToken* tok = nullptr;
		const uint32_t idx = (uint32_t)fg.frameIndex;
		if (Sl().GetNewFrameToken(tok, &idx) == sl::Result::eOk) fg.slToken = tok; else fg.slToken = nullptr;
		if (fg.slToken && Sl().ReflexSleep) Sl().ReflexSleep(*tok);
		SlMark(sl::PCLMarker::eSimulationStart, fg.slToken);
		SlMark(sl::PCLMarker::eRenderSubmitStart, fg.slToken);
	}
#endif
#if NUKE_XESS_FG
	if (fg.kind == FG_XESS && fg.xell)
	{
		xell_context_handle_t xl = (xell_context_handle_t)fg.xell;
		XeFg().XellSleep(xl, (uint32_t)fg.frameIndex);
		XeFg().XellAddMarkerData(xl, (uint32_t)fg.frameIndex, XELL_SIMULATION_START);
		XeFg().XellAddMarkerData(xl, (uint32_t)fg.frameIndex, XELL_RENDERSUBMIT_START);
	}
#endif
#endif
}

// endCamera of the back-buffer pass: what the generators reconstruct the in-between frame from.
void NukeDiligent::Impl::FrameGenCapture()
{
	fg.lastCapFrame = frameId;   // the attach waits for this even without a generator
	if (!fg.attached || !gbufDepth || !gbufVel) return;
	// A later back-buffer camera this frame wins; "previous" is always the last PRESENTED frame's.
	fg.prevView = fg.lastView; fg.prevProj = fg.lastProj;
	fg.depth = gbufDepth; fg.vel = gbufVel;
	fg.rw = curRTW; fg.rh = curRTH;
	fg.jx = curJitterX; fg.jy = curJitterY;
	fg.view = curView; fg.proj = curProjNoJitter;
	fg.nearZ = curNear; fg.farZ = curFar; fg.fov = curFovY;
	memcpy(fg.pos, curCamPos, sizeof(fg.pos));
	fg.reset = !fg.hadLast;
	fg.captured = true;
}

// After the world passes, before the UI: the HUD-less image the generators separate the UI with.
void NukeDiligent::Impl::FrameGenHudless(ITextureView* backRTV)
{
	if (!fg.attached || !fg.captured || !backRTV) return;
	ITexture* back = backRTV->GetTexture();
	if (!back) return;
	const TextureDesc& bd = back->GetDesc();
	if (!fg.hudless || fg.hudless->GetDesc().Width != bd.Width || fg.hudless->GetDesc().Height != bd.Height || fg.hudless->GetDesc().Format != bd.Format)
	{
		Trash(fg.hudless); fg.hudless.Release();
		TextureDesc td; td.Name = "FrameGen HUD-less"; td.Type = RESOURCE_DIM_TEX_2D; td.Width = bd.Width; td.Height = bd.Height;
		td.Format = bd.Format; td.BindFlags = BIND_SHADER_RESOURCE; td.Usage = USAGE_DEFAULT;
		device->CreateTexture(td, nullptr, &fg.hudless);
		if (!fg.hudless) return;
	}
	context->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE);
	CopyTextureAttribs cp(back, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, fg.hudless, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	context->CopyTexture(cp);
#ifdef _WIN32
#if NUKE_STREAMLINE
	if (fg.kind == FG_DLSS) SlMark(sl::PCLMarker::eSimulationEnd, fg.slToken);
#endif
#if NUKE_XESS_FG
	if (fg.kind == FG_XESS && fg.xell) XeFg().XellAddMarkerData((xell_context_handle_t)fg.xell, (uint32_t)fg.frameIndex, XELL_SIMULATION_END);
#endif
#endif
}

// This frame's sl::Constants for DLSS-G (either backend); the matrices are the captured camera's.
bool NukeDiligent::Impl::FrameGenSlConstants(int W, int H)
{
#if defined(_WIN32) && NUKE_STREAMLINE
	if (!fg.slToken || !fg.slReady) return false;
	const sl::FrameToken& tok = *(const sl::FrameToken*)fg.slToken;
	const float right[3] = { fg.view.m[0][0], fg.view.m[1][0], fg.view.m[2][0] };
	const float up[3]    = { fg.view.m[0][1], fg.view.m[1][1], fg.view.m[2][1] };
	const float fwd[3]   = { fg.view.m[0][2], fg.view.m[1][2], fg.view.m[2][2] };
	const float4x4 vp = fg.view * fg.proj;
	const float4x4 prevVP = fg.reset ? vp : (fg.prevView * fg.prevProj);
	const float4x4 clipToPrev = vp.Inverse() * prevVP;
	sl::Constants c{};
	c.cameraViewToClip = SlMat(fg.proj);
	c.clipToCameraView = SlMat(fg.proj.Inverse());
	c.clipToLensClip = SlMat(float4x4::Identity());
	c.clipToPrevClip = SlMat(clipToPrev);
	c.prevClipToClip = SlMat(clipToPrev.Inverse());
	c.jitterOffset = { fg.jx, fg.jy };
	c.mvecScale = { -1.0f, -1.0f };   // UV (current - previous) -> the previous - current the reprojection wants
	c.cameraPinholeOffset = { 0.0f, 0.0f };
	c.cameraPos = { fg.pos[0], fg.pos[1], fg.pos[2] };
	c.cameraUp = { up[0], up[1], up[2] };
	c.cameraRight = { right[0], right[1], right[2] };
	c.cameraFwd = { fwd[0], fwd[1], fwd[2] };
	c.cameraNear = fg.nearZ; c.cameraFar = fg.farZ;
	c.cameraFOV = fg.fov;
	c.cameraAspectRatio = H > 0 ? (float)W / (float)H : 1.0f;
	c.motionVectorsInvalidValue = FLT_MIN;
	c.depthInverted = sl::Boolean::eFalse;
	c.cameraMotionIncluded = sl::Boolean::eTrue;
	c.motionVectors3D = sl::Boolean::eFalse;
	c.reset = fg.reset ? sl::Boolean::eTrue : sl::Boolean::eFalse;
	c.orthographicProjection = sl::Boolean::eFalse;
	c.motionVectorsDilated = sl::Boolean::eFalse;
	c.motionVectorsJittered = sl::Boolean::eFalse;
	const sl::Result r = Sl().SetConstants(c, tok, kSlViewport);
	if (r != sl::Result::eOk && fg.lastLog != frameId) { fg.lastLog = frameId; cout << "[NukeDiligent]\tDLSS-G constants refused: " << sl::getResultAsStr(r) << endl; }
	return r == sl::Result::eOk;
#else
	(void)W; (void)H;
	return false;
#endif
}

// Right before Present: this frame's inputs to the vendor, and the generation on / off for the
// frame (a frame without a back-buffer camera - the editor's, a menu - presents as rendered).
void NukeDiligent::Impl::FrameGenBeforePresent()
{
	if (!fg.attached) return;
#ifdef _WIN32
	const bool ready = fg.captured && fg.depth && fg.vel && fg.hudless;
	const float dt = Time::getSingleton() ? (float)std::min(std::max(Time::getSingleton()->gameDelta, 0.0), 0.25) : 0.0f;
	const int W = (int)swapChain->GetDesc().Width, H = (int)swapChain->GetDesc().Height;
	if (useVulkan)
	{
		FrameGenVkBeforePresent(ready, dt, W, H);
#if NUKE_STREAMLINE
		if (fg.kind == FG_DLSS) { SlMark(sl::PCLMarker::eRenderSubmitEnd, fg.slToken); SlMark(sl::PCLMarker::ePresentStart, fg.slToken); }
#endif
		return;
	}
	RefCntAutoPtr<IDeviceContextD3D12> ctx12(context, IID_DeviceContextD3D12);
	auto res12 = [](ITexture* t) -> ID3D12Resource* { RefCntAutoPtr<ITextureD3D12> t12(t, IID_TextureD3D12); return t12 ? t12->GetD3D12Texture() : nullptr; };
	if (ready && ctx12)
	{
		// Every input in the shader-read state - through Diligent, so its tracking agrees.
		context->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE);
		StateTransitionDesc bars[3] = {
			{fg.depth,   RESOURCE_STATE_UNKNOWN, RESOURCE_STATE_SHADER_RESOURCE, STATE_TRANSITION_FLAG_UPDATE_STATE},
			{fg.vel,     RESOURCE_STATE_UNKNOWN, RESOURCE_STATE_SHADER_RESOURCE, STATE_TRANSITION_FLAG_UPDATE_STATE},
			{fg.hudless, RESOURCE_STATE_UNKNOWN, RESOURCE_STATE_SHADER_RESOURCE, STATE_TRANSITION_FLAG_UPDATE_STATE} };
		context->TransitionResourceStates(3, bars);
	}
	const bool wantLive = ready;
	// The camera basis the generators reproject with (rows of the view matrix).
	const float right[3] = { fg.view.m[0][0], fg.view.m[1][0], fg.view.m[2][0] };
	const float up[3]    = { fg.view.m[0][1], fg.view.m[1][1], fg.view.m[2][1] };
	const float fwd[3]   = { fg.view.m[0][2], fg.view.m[1][2], fg.view.m[2][2] };
	(void)right; (void)up; (void)fwd; (void)res12;
#if NUKE_FFX_FG
	if (fg.kind == FG_FSR && fg.ffx && ctx12)
	{
		if (ready)
		{
			ffxDispatchDescFrameGenerationPrepareV2 pd{};
			pd.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE_V2;
			pd.frameID = fg.frameIndex;
			pd.flags = 0;
			pd.commandList = ctx12->GetD3D12CommandList();
			pd.renderSize = { (uint32_t)fg.rw, (uint32_t)fg.rh };
			pd.jitterOffset = { fg.jx, fg.jy };
			pd.motionVectorScale = { -(float)fg.rw, -(float)fg.rh };   // UV (current - previous) -> pixels (previous - current)
			pd.frameTimeDelta = dt * 1000.0f;
			pd.reset = fg.reset;
			pd.cameraNear = fg.nearZ; pd.cameraFar = fg.farZ;
			pd.cameraFovAngleVertical = fg.fov;
			pd.viewSpaceToMetersFactor = 1.0f;
			pd.depth = ffxApiGetResourceDX12(res12(fg.depth), FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
			pd.motionVectors = ffxApiGetResourceDX12(res12(fg.vel), FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
			memcpy(pd.cameraPosition, fg.pos, sizeof(pd.cameraPosition));
			memcpy(pd.cameraUp, up, sizeof(pd.cameraUp));
			memcpy(pd.cameraRight, right, sizeof(pd.cameraRight));
			memcpy(pd.cameraForward, fwd, sizeof(pd.cameraForward));
			const ffxReturnCode_t rc = FfxFg().Dispatch(&fg.ffx, &pd.header);
			if (rc != FFX_API_RETURN_OK && fg.lastLog != frameId) { fg.lastLog = frameId; cout << "[NukeDiligent]\tFSR FG prepare failed: " << rc << endl; }
		}
		ffxConfigureDescFrameGeneration cfg{};
		cfg.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
		cfg.swapChain = fg.proxy;
		cfg.presentCallback = nullptr;   // the swap chain's own UI composition (back buffer minus HUD-less)
		cfg.frameGenerationCallback = FfxFgDispatchCb;
		cfg.frameGenerationCallbackUserContext = &fg.ffx;
		cfg.frameGenerationEnabled = wantLive;
		cfg.allowAsyncWorkloads = false;
		if (ready) cfg.HUDLessColor = ffxApiGetResourceDX12(res12(fg.hudless), FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
		cfg.flags = 0;
		cfg.onlyPresentGenerated = false;
		cfg.generationRect = { 0, 0, W, H };
		cfg.frameID = fg.frameIndex;
		const ffxReturnCode_t rc = FfxFg().Configure(&fg.ffx, &cfg.header);
		if (rc != FFX_API_RETURN_OK && fg.lastLog != frameId) { fg.lastLog = frameId; cout << "[NukeDiligent]\tFSR FG configure failed: " << rc << endl; }
		if (ready) { ctx12->Flush(); lastInstBind.pso = nullptr; }   // the vendor recorded its own heaps / state: a fresh command list
		fg.live = wantLive;
	}
#endif
#if NUKE_XESS_FG
	if (fg.kind == FG_XESS && fg.xefg && ctx12)
	{
		xefg_swapchain_handle_t h = (xefg_swapchain_handle_t)fg.xefg;
		if (wantLive != fg.live) { XeFg().SetEnabled(h, wantLive ? 1u : 0u); fg.live = wantLive; }
		if (ready)
		{
			// XeSS wants the exact NON_PIXEL_SHADER_RESOURCE state it is told.
			ctx12->TransitionTextureState(fg.depth, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
			ctx12->TransitionTextureState(fg.vel, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
			ctx12->TransitionTextureState(fg.hudless, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
			ID3D12GraphicsCommandList* cl = ctx12->GetD3D12CommandList();
			const uint32_t id = (uint32_t)fg.frameIndex;
			xefg_swapchain_d3d12_resource_data_t rd{};
			rd.validity = XEFG_SWAPCHAIN_RV_UNTIL_NEXT_PRESENT;
			rd.incomingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
			rd.type = XEFG_SWAPCHAIN_RES_DEPTH;         rd.resourceSize = { (uint32_t)fg.rw, (uint32_t)fg.rh }; rd.pResource = res12(fg.depth);   XeFg().D3D12TagFrameResource(h, cl, id, &rd);
			rd.type = XEFG_SWAPCHAIN_RES_MOTION_VECTOR; rd.resourceSize = { (uint32_t)fg.rw, (uint32_t)fg.rh }; rd.pResource = res12(fg.vel);     XeFg().D3D12TagFrameResource(h, cl, id, &rd);
			rd.type = XEFG_SWAPCHAIN_RES_HUDLESS_COLOR; rd.resourceSize = { (uint32_t)W, (uint32_t)H };         rd.pResource = res12(fg.hudless); XeFg().D3D12TagFrameResource(h, cl, id, &rd);
			xefg_swapchain_frame_constant_data_t c{};
			memcpy(c.viewMatrix, fg.view.Data(), sizeof(c.viewMatrix));
			memcpy(c.projectionMatrix, fg.proj.Data(), sizeof(c.projectionMatrix));
			c.jitterOffsetX = fg.jx; c.jitterOffsetY = fg.jy;
			c.motionVectorScaleX = -(float)fg.rw; c.motionVectorScaleY = -(float)fg.rh;   // UV (current - previous) -> pixels (current to previous)
			c.resetHistory = fg.reset ? 1u : 0u;
			c.frameRenderTime = dt * 1000.0f;
			XeFg().TagFrameConstants(h, id, &c);
			XeFg().SetPresentId(h, id);
			ctx12->Flush(); lastInstBind.pso = nullptr;
		}
		if (fg.xell)
		{
			XeFg().XellAddMarkerData((xell_context_handle_t)fg.xell, (uint32_t)fg.frameIndex, XELL_RENDERSUBMIT_END);
			XeFg().XellAddMarkerData((xell_context_handle_t)fg.xell, (uint32_t)fg.frameIndex, XELL_PRESENT_START);
		}
	}
#endif
#if NUKE_STREAMLINE
	if (fg.kind == FG_DLSS && fg.slReady)
	{
		if (wantLive != fg.live && Sl().DLSSGSetOptions && frameId != fg.attachFrame)   // not on the attach frame: the detach's "off" was this frame's call
		{
			sl::DLSSGOptions o{};
			o.mode = wantLive ? sl::DLSSGMode::eOn : sl::DLSSGMode::eOff;
			o.numFramesToGenerate = (uint32_t)std::max(fg.framesLive, 1);
			o.flags = sl::DLSSGFlags::eRetainResourcesWhenOff;
			Sl().DLSSGSetOptions(kSlViewport, o);
			fg.live = wantLive;
		}
		if (ready && fg.slToken && ctx12)
		{
			const sl::FrameToken& tok = *(const sl::FrameToken*)fg.slToken;
			const uint32_t readState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
			sl::Resource depth(sl::ResourceType::eTex2d, res12(fg.depth), readState);
			sl::Resource mvec(sl::ResourceType::eTex2d, res12(fg.vel), readState);
			sl::Resource hudless(sl::ResourceType::eTex2d, res12(fg.hudless), readState);
			sl::Extent rExt{ 0u, 0u, (uint32_t)fg.rw, (uint32_t)fg.rh };
			sl::Extent oExt{ 0u, 0u, (uint32_t)W, (uint32_t)H };
			sl::ResourceTag tags[4] = {
				sl::ResourceTag(&depth,   sl::kBufferTypeDepth,         sl::ResourceLifecycle::eValidUntilPresent, &rExt),
				sl::ResourceTag(&mvec,    sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eValidUntilPresent, &rExt),
				sl::ResourceTag(&hudless, sl::kBufferTypeHUDLessColor,  sl::ResourceLifecycle::eValidUntilPresent, &oExt),
				sl::ResourceTag(nullptr,  sl::kBufferTypeBackbuffer,    sl::ResourceLifecycle::eValidUntilPresent, &oExt) };   // extent only: SL knows the buffer
			Sl().SetTagForFrame(tok, kSlViewport, tags, 4, ctx12->GetD3D12CommandList());
			FrameGenSlConstants(W, H);
		}
		SlMark(sl::PCLMarker::eRenderSubmitEnd, fg.slToken);
		SlMark(sl::PCLMarker::ePresentStart, fg.slToken);
	}
#endif
#endif
}

// After Present: the markers, the frame id, the vendor's word on what it presented (now and then),
// the presents-per-rendered-frame the title shows.
void NukeDiligent::Impl::FrameGenAfterPresent()
{
	// The renderer's own rendered-FPS meter (the host's title counter is the engine's, not ours).
	{
		const double now = NowSeconds();
		if (fg.fpsLast > 0.0) { fg.fpsAcc += now - fg.fpsLast; ++fg.fpsFrames; }
		fg.fpsLast = now;
		if (fg.fpsAcc >= 0.5) { fg.fpsRendered = fg.fpsFrames / fg.fpsAcc; fg.fpsAcc = 0.0; fg.fpsFrames = 0; }
	}
	if (!fg.attached) return;
#ifdef _WIN32
#if NUKE_STREAMLINE
	if (fg.kind == FG_DLSS)
	{
		SlMark(sl::PCLMarker::ePresentEnd, fg.slToken);
		if (fg.slReady && Sl().DLSSGGetState && (fg.frameIndex % 300) == 299)
		{
			sl::DLSSGState st{};
			if (Sl().DLSSGGetState(kSlViewport, st, nullptr) == sl::Result::eOk)
			{
				if (st.status != sl::DLSSGStatus::eOk) cout << "[NukeDiligent]\tDLSS-G status 0x" << std::hex << (unsigned)st.status << std::dec << " (1 = resolution too low, 2 = Reflex not detected, 4 = HDR format, 8 = constants invalid, 16 = back-buffer index not queried)" << endl;
				else cout << "[NukeDiligent]\tDLSS-G: " << st.numFramesActuallyPresented << " frames presented per rendered frame (max generated " << st.numFramesToGenerateMax << ")" << endl;
			}
		}
	}
#endif
#if NUKE_XESS_FG
	if (fg.kind == FG_XESS && fg.xefg)
	{
		if (fg.xell) XeFg().XellAddMarkerData((xell_context_handle_t)fg.xell, (uint32_t)fg.frameIndex, XELL_PRESENT_END);
		if ((fg.frameIndex % 300) == 299)
		{
			xefg_swapchain_present_status_t ps{};
			if (XeFg().GetLastPresentStatus((xefg_swapchain_handle_t)fg.xefg, &ps) == XEFG_SWAPCHAIN_RESULT_SUCCESS)
				cout << "[NukeDiligent]\tXeSS-FG: " << ps.framesPresented << " frames presented last present, generation " << (ps.isFrameGenEnabled ? "on" : "off") << ", result " << (int)ps.frameGenResult << endl;
		}
	}
#endif
	// The vendor-neutral measurement: presents against rendered frames (D3D12: DXGI's present
	// count on the proxy, generated frames included; Vulkan: the vendor's counter).
	if ((fg.frameIndex % 30) == 29)
	{
		if (useVulkan)
		{
			const double r = FrameGenVkPresentRatio();
			if (r > 0.0) fg.ratio = r;
		}
		else if (fg.proxy)
		{
			UINT n = 0;
			if (SUCCEEDED(((IDXGISwapChain*)fg.proxy)->GetLastPresentCount(&n)))
			{
				if (fg.ratioBase && n > fg.ratioBase && fg.frameIndex > fg.ratioFrame) fg.ratio = (double)(n - fg.ratioBase) / (double)(fg.frameIndex - fg.ratioFrame);
				fg.ratioBase = n; fg.ratioFrame = fg.frameIndex;
			}
		}
		if ((fg.frameIndex % 300) == 299 && fg.ratio > 0.0)
			cout << "[NukeDiligent]\tframe generation: " << FrameGenName(fg.kind) << " " << std::fixed << std::setprecision(2) << fg.ratio << " presents per rendered frame" << endl;
	}
#endif
	++fg.frameIndex;
	if (fg.captured) { fg.lastView = fg.view; fg.lastProj = fg.proj; }
	fg.hadLast = fg.captured;   // a frame without a capture breaks the reprojection chain
	fg.captured = false;
}

// The host's title gets the generator's state and the presented FPS (the host counts rendered frames).
void NukeDiligent::Impl::FrameGenTitle(std::string& title)
{
	if (!fg.attached) return;
	const double ratio = fg.ratio > 0.0 ? fg.ratio : (double)(fg.framesLive + 1);
	char b[96];
	snprintf(b, sizeof(b), " | %s x%d -> %d FPS", FrameGenName(fg.kind), fg.framesLive + 1, (int)(fg.fpsRendered * ratio + 0.5));
	title += b;
}

void NukeDiligent::Impl::ShutdownFrameGen()
{
	FrameGenDetach("shutdown");
#if defined(_WIN32) && NUKE_STREAMLINE
	if (fg.slInit && Sl().Shutdown) { Sl().Shutdown(); fg.slInit = false; fg.slReady = false; }
#endif
}
