#include "NukeDiligentImpl.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <API/Model/Time.h>   // the frame delta the temporal reconstruction eases by
#include "NukeDiligent_FrameGenShared.h"   // the Vulkan FSR entry points (NukeDiligent_FrameGenVk.cpp)
#if NUKE_DLSS_NGX
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>      // the Vulkan types the Vk device interface and the NGX header want first
#include "RenderDeviceD3D12.h"
#include "RenderDeviceVk.h"
#include <nvsdk_ngx.h>          // the NGX diagnostics (why DLSS was not offered)
#include <nvsdk_ngx_vk.h>
#endif
#if defined(_WIN32) && (NUKE_FFX || NUKE_XESS)
#include "RenderDeviceD3D12.h"
#include "DeviceContextD3D12.h"
#include "TextureD3D12.h"
#endif
#if NUKE_FFX
#include <api/include/ffx_api.h>            // AMD FidelityFX API: FSR 3.1 / FSR 4 through amd_fidelityfx_upscaler_dx12.dll
#include <api/include/ffx_api_types.h>
#include <api/include/dx12/ffx_api_dx12.h>
#include <ffx_upscale.h>
#endif
#if NUKE_XESS
#include <xess/xess.h>                  // Intel XeSS through libxess.dll
#include <xess/xess_d3d12.h>
#endif

// 4.2 super resolution: the scene renders at an internal size below the target and the chain's
// "upscale" stage reconstructs the output. Three sources of upscalers, all vendor runtimes as
// separate DLLs found next to this renderer DLL (absent = the variant is not offered):
//   - Diligent's SuperResolution factory: DLSS through NGX (nvngx_dlss.dll + an RTX driver; D3D12
//     and Vulkan) and the software FSR 1 (spatial, every GPU and backend);
//   - AMD FidelityFX API: FSR 3.1, and FSR 4 where the driver and GPU carry it, through
//     amd_fidelityfx_upscaler_dx12.dll (D3D12 only - the 2.x SDK ships no Vulkan binary);
//   - Intel XeSS through libxess.dll (D3D12; runs on any DP4a GPU).
// Temporal variants replace TAA and run in HDR at the stage's slot; the spatial one runs on the
// tonemapped image into the output.

using namespace Diligent;
using std::cout; using std::endl;

// Mode numbers of upscale.post.hlsl g_Mode: UP_* (NukeDiligent_FrameGenShared.h).
enum { KIND_DILIGENT = 0, KIND_FFX = 1, KIND_XESS = 2 };

#ifdef _WIN32
namespace nukediligent {
// The directory this renderer DLL lives in (modules/): the vendor runtimes sit next to it.
// Shared with NukeDiligent_FrameGen.cpp.
std::wstring ModuleDir()
{
	HMODULE self = nullptr;
	GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCWSTR)&ModuleDir, &self);
	wchar_t buf[MAX_PATH * 2] = {};
	GetModuleFileNameW(self, buf, MAX_PATH * 2);
	std::wstring p(buf);
	const size_t k = p.find_last_of(L"\\/");
	return k == std::wstring::npos ? L"." : p.substr(0, k);
}
HMODULE LoadVendorDll(const wchar_t* name)
{
	const std::wstring path = ModuleDir() + L"\\" + name;
	return LoadLibraryExW(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
}
}  // namespace nukediligent
using nukediligent::ModuleDir; using nukediligent::LoadVendorDll;
#endif

namespace {

#if NUKE_FFX
struct FfxFns
{
	HMODULE dll = nullptr;
	PfnFfxCreateContext  CreateContext = nullptr;
	PfnFfxDestroyContext DestroyContext = nullptr;
	PfnFfxConfigure      Configure = nullptr;
	PfnFfxQuery          Query = nullptr;
	PfnFfxDispatch       Dispatch = nullptr;
	bool ok() const { return CreateContext && DestroyContext && Query && Dispatch; }
};
FfxFns g_ffx;
bool g_ffxTried = false;
void FfxMessage(uint32_t type, const wchar_t* msg)
{
	if (!msg) return;
	std::wstring w(msg); std::string s(w.begin(), w.end());
	cout << "[FFX]\t" << (type == FFX_API_MESSAGE_TYPE_ERROR ? "error: " : "") << s << endl;
}
const FfxFns& Ffx()
{
	if (!g_ffxTried)
	{
		g_ffxTried = true;
		g_ffx.dll = LoadVendorDll(L"amd_fidelityfx_upscaler_dx12.dll");
		if (g_ffx.dll)
		{
			g_ffx.CreateContext  = (PfnFfxCreateContext)GetProcAddress(g_ffx.dll, "ffxCreateContext");
			g_ffx.DestroyContext = (PfnFfxDestroyContext)GetProcAddress(g_ffx.dll, "ffxDestroyContext");
			g_ffx.Configure      = (PfnFfxConfigure)GetProcAddress(g_ffx.dll, "ffxConfigure");
			g_ffx.Query          = (PfnFfxQuery)GetProcAddress(g_ffx.dll, "ffxQuery");
			g_ffx.Dispatch       = (PfnFfxDispatch)GetProcAddress(g_ffx.dll, "ffxDispatch");
		}
	}
	return g_ffx;
}
uint32_t FfxQualityOf(int q)
{
	switch (q)
	{
		case 0: return FFX_UPSCALE_QUALITY_MODE_NATIVEAA;
		case 1: return FFX_UPSCALE_QUALITY_MODE_QUALITY;
		case 2: return FFX_UPSCALE_QUALITY_MODE_BALANCED;
		case 3: return FFX_UPSCALE_QUALITY_MODE_PERFORMANCE;
		default: return FFX_UPSCALE_QUALITY_MODE_ULTRA_PERFORMANCE;
	}
}
#endif

#if NUKE_XESS
struct XessFns
{
	HMODULE dll = nullptr;
	decltype(&xessD3D12CreateContext)         D3D12CreateContext = nullptr;
	decltype(&xessD3D12BuildPipelines)        D3D12BuildPipelines = nullptr;
	decltype(&xessD3D12Init)                  D3D12Init = nullptr;
	decltype(&xessD3D12Execute)               D3D12Execute = nullptr;
	decltype(&xessDestroyContext)             DestroyContext = nullptr;
	decltype(&xessGetOptimalInputResolution)  GetOptimalInputResolution = nullptr;
	decltype(&xessGetPipelineBuildStatus)     GetPipelineBuildStatus = nullptr;
	decltype(&xessSetVelocityScale)           SetVelocityScale = nullptr;
	decltype(&xessSetLoggingCallback)         SetLoggingCallback = nullptr;
	decltype(&xessGetVersion)                 GetVersion = nullptr;
	bool ok() const { return D3D12CreateContext && D3D12Init && D3D12Execute && DestroyContext && GetOptimalInputResolution && SetVelocityScale; }
};
XessFns g_xess;
bool g_xessTried = false;
void XessLog(const char* msg, xess_logging_level_t level)
{
	if (msg) cout << "[XeSS]\t" << (level >= XESS_LOGGING_LEVEL_ERROR ? "error: " : "") << msg << endl;
}
const XessFns& Xess()
{
	if (!g_xessTried)
	{
		g_xessTried = true;
		g_xess.dll = LoadVendorDll(L"libxess.dll");
		if (g_xess.dll)
		{
#define NUKE_XESS_FN(field, name) g_xess.field = (decltype(g_xess.field))GetProcAddress(g_xess.dll, #name)
			NUKE_XESS_FN(D3D12CreateContext, xessD3D12CreateContext);
			NUKE_XESS_FN(D3D12BuildPipelines, xessD3D12BuildPipelines);
			NUKE_XESS_FN(D3D12Init, xessD3D12Init);
			NUKE_XESS_FN(D3D12Execute, xessD3D12Execute);
			NUKE_XESS_FN(DestroyContext, xessDestroyContext);
			NUKE_XESS_FN(GetOptimalInputResolution, xessGetOptimalInputResolution);
			NUKE_XESS_FN(GetPipelineBuildStatus, xessGetPipelineBuildStatus);
			NUKE_XESS_FN(SetVelocityScale, xessSetVelocityScale);
			NUKE_XESS_FN(SetLoggingCallback, xessSetLoggingCallback);
			NUKE_XESS_FN(GetVersion, xessGetVersion);
#undef NUKE_XESS_FN
		}
	}
	return g_xess;
}
xess_quality_settings_t XessQualityOf(int q)
{
	switch (q)
	{
		case 0: return XESS_QUALITY_SETTING_AA;
		case 1: return XESS_QUALITY_SETTING_QUALITY;
		case 2: return XESS_QUALITY_SETTING_BALANCED;
		case 3: return XESS_QUALITY_SETTING_PERFORMANCE;
		default: return XESS_QUALITY_SETTING_ULTRA_PERFORMANCE;
	}
}
#endif

// Halton (2,3) sub-pixel jitter in [-0.5, 0.5]: the pattern the vendors without their own
// generator (XeSS) recommend, with the phase count the scale asks for.
float Halton(int i, int b) { float f = 1.0f, r = 0.0f; while (i > 0) { f /= (float)b; r += f * (float)(i % b); i /= b; } return r; }

#if NUKE_DLSS_NGX
void NVSDK_CONV NgxLog(const char* msg, NVSDK_NGX_Logging_Level, NVSDK_NGX_Feature)
{
	if (msg && *msg) cout << "[NGX]\t" << msg << (msg[strlen(msg) - 1] == '\n' ? "" : "\n");
}

// DLSS was not offered on an NVIDIA adapter: ask NGX itself why (its own log + the capability
// parameters), so a missing nvngx_dlss.dll, an old driver or a refused feature is named, not guessed.
void LogNGXReason(IRenderDevice* device)
{
	NVSDK_NGX_FeatureCommonInfo ci{};
	ci.LoggingInfo.LoggingCallback = NgxLog;
	ci.LoggingInfo.MinimumLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_ON;
	ci.LoggingInfo.DisableOtherLoggingSinks = true;
	const char* pid = "750fed3a-efba-42ba-801b-22d4cbad9148";   // the SuperResolution module's project id
	NVSDK_NGX_Parameter* prm = nullptr;
	NVSDK_NGX_Result r = NVSDK_NGX_Result_Fail;
	bool vk = false;
	if (device->GetDeviceInfo().Type == RENDER_DEVICE_TYPE_D3D12)
	{
		RefCntAutoPtr<IRenderDeviceD3D12> d12(device, IID_RenderDeviceD3D12);
		if (!d12) return;
		r = NVSDK_NGX_D3D12_Init_with_ProjectID(pid, NVSDK_NGX_ENGINE_TYPE_CUSTOM, "0", L".", d12->GetD3D12Device(), &ci);
		if (NVSDK_NGX_SUCCEED(r)) r = NVSDK_NGX_D3D12_GetCapabilityParameters(&prm);
	}
	else if (device->GetDeviceInfo().Type == RENDER_DEVICE_TYPE_VULKAN)
	{
		RefCntAutoPtr<IRenderDeviceVk> dvk(device, IID_RenderDeviceVk);
		if (!dvk) return;
		vk = true;
		r = NVSDK_NGX_VULKAN_Init_with_ProjectID(pid, NVSDK_NGX_ENGINE_TYPE_CUSTOM, "0", L".", dvk->GetVkInstance(), dvk->GetVkPhysicalDevice(), dvk->GetVkDevice(), nullptr, nullptr, &ci);
		if (NVSDK_NGX_SUCCEED(r)) r = NVSDK_NGX_VULKAN_GetCapabilityParameters(&prm);
	}
	else return;
	if (prm)
	{
		int avail = 0, upd = 0, mj = 0, mn = 0, init = 0;
		NVSDK_NGX_Parameter_GetI(prm, NVSDK_NGX_Parameter_SuperSampling_Available, &avail);
		NVSDK_NGX_Parameter_GetI(prm, NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver, &upd);
		NVSDK_NGX_Parameter_GetI(prm, NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMajor, &mj);
		NVSDK_NGX_Parameter_GetI(prm, NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMinor, &mn);
		NVSDK_NGX_Parameter_GetI(prm, NVSDK_NGX_Parameter_SuperSampling_FeatureInitResult, &init);
		cout << "[NukeDiligent]\tDLSS not offered: available=" << avail << " needsDriver=" << upd << " minDriver=" << mj << "." << mn
		     << " featureInit=0x" << std::hex << (unsigned)init << std::dec << " (nvngx_dlss.dll must sit next to NukeRenderDiligent.dll)" << endl;
		if (vk) NVSDK_NGX_VULKAN_DestroyParameters(prm); else NVSDK_NGX_D3D12_DestroyParameters(prm);
	}
	else cout << "[NukeDiligent]\tDLSS not offered: NGX init / capability query failed: 0x" << std::hex << (unsigned)r << std::dec << endl;
	if (vk) { RefCntAutoPtr<IRenderDeviceVk> dvk(device, IID_RenderDeviceVk); NVSDK_NGX_VULKAN_Shutdown1(dvk->GetVkDevice()); }
	else    { RefCntAutoPtr<IRenderDeviceD3D12> d12(device, IID_RenderDeviceD3D12); NVSDK_NGX_D3D12_Shutdown1(d12->GetD3D12Device()); }
}
#endif

}  // namespace

void NukeDiligent::Impl::CreateUpscaleFactory()
{
	upscaleStates.clear();
	upVariants.clear();
	srFactory.Release(); srVariants.clear();
	if (!device) return;
	LoadAndCreateSuperResolutionFactory(device, &srFactory);   // static link: the providers probe the device now
	if (srFactory)
	{
		Uint32 n = 0;
		srFactory->EnumerateVariants(n, nullptr);
		srVariants.resize(n);
		if (n) srFactory->EnumerateVariants(n, srVariants.data());
		for (int i = 0; i < (int)srVariants.size(); ++i)
		{
			UpVariant v; v.name = srVariants[i].Name; v.temporal = srVariants[i].Type == SUPER_RESOLUTION_TYPE_TEMPORAL; v.kind = KIND_DILIGENT; v.index = i;
			upVariants.push_back(v);
		}
	}
	const bool d3d12 = device->GetDeviceInfo().Type == RENDER_DEVICE_TYPE_D3D12;
	(void)d3d12;
#if NUKE_FFX
	if (d3d12 && Ffx().ok())
	{
		UpVariant v; v.name = "FFX: FSR"; v.temporal = true; v.kind = KIND_FFX; v.index = -1;
		upVariants.push_back(v);
	}
#endif
#ifdef _WIN32
	if (useVulkan && nukediligent::FfxVkAvailable())   // FSR 3.1.4 through amd_fidelityfx_vk.dll (SDK 1.1.4)
	{
		UpVariant v; v.name = "FFX: FSR 3.1 (Vulkan)"; v.temporal = true; v.kind = KIND_FFX; v.index = -1;
		upVariants.push_back(v);
	}
	if (useVulkan && nukediligent::XessVkAvailable())   // XeSS through libxess.dll's Vulkan entry points
	{
		UpVariant v; v.name = nukediligent::XessVkVariantName(); v.temporal = true; v.kind = KIND_XESS; v.index = -1;
		upVariants.push_back(v);
	}
#endif
#if NUKE_XESS
	if (d3d12 && Xess().ok())
	{
		xess_version_t ver{};
		if (Xess().GetVersion) Xess().GetVersion(&ver);
		UpVariant v; v.name = "XeSS " + std::to_string(ver.major) + "." + std::to_string(ver.minor) + "." + std::to_string(ver.patch); v.temporal = true; v.kind = KIND_XESS; v.index = -1;
		upVariants.push_back(v);
	}
#endif
	bool dlss = false;
	for (const auto& v : upVariants)
	{
		cout << "[NukeDiligent]\tupscaler: " << v.name << (v.temporal ? " (temporal)" : " (spatial)") << endl;
		if (v.name.find("DLSS") != std::string::npos) dlss = true;
	}
	if (upVariants.empty()) cout << "[NukeDiligent]\tupscalers: none" << endl;
#if NUKE_DLSS_NGX
	if (!dlss && device->GetAdapterInfo().Vendor == ADAPTER_VENDOR_NVIDIA) LogNGXReason(device);
#endif
}

// The chain's upscale stage: mode (g_Mode) or -1 when the chain has none; quality + sharpness out.
int NukeDiligent::Impl::UpscaleWanted(int& quality, float& sharp) const
{
	for (const auto& cs : postChain)
	{
		auto pit = postPipes.find(cs.pipeline);
		if (pit == postPipes.end() || !pit->second.isUpscale) continue;
		const int mode = cs.params.size() > 0 ? (int)(cs.params[0] + 0.5f) : UP_AUTO;
		quality = cs.params.size() > 1 ? (int)(cs.params[1] + 0.5f) : 1;
		sharp   = cs.params.size() > 2 ? cs.params[2] : 0.0f;
		if (quality < 0) quality = 0;
		if (quality > 4) quality = 4;
		if (sharp < 0.0f) sharp = 0.0f;
		if (sharp > 1.0f) sharp = 1.0f;
		return mode;
	}
	return -1;
}

// mode -> the variant that implements it here, or the best available for Auto. A mode this GPU
// cannot do falls to the next honest tier: the adapter's own vendor upscaler, then the
// cross-vendor temporal ones, then FSR 1.
int NukeDiligent::Impl::UpscaleVariantOf(int mode) const
{
	auto find = [&](int kind, const char* name, bool temporal) -> int
	{
		for (int i = 0; i < (int)upVariants.size(); ++i)
			if (upVariants[i].kind == kind && upVariants[i].temporal == temporal && (!name || upVariants[i].name.find(name) != std::string::npos)) return i;
		return -1;
	};
	switch (mode)
	{
		case UP_DLSS: return find(KIND_DILIGENT, "DLSS", true);
		case UP_FSR:  return find(KIND_FFX, nullptr, true);
		case UP_XESS: return find(KIND_XESS, nullptr, true);
		case UP_FSR1: return find(KIND_DILIGENT, "FSR", false);
		default: return -1;
	}
}

int NukeDiligent::Impl::ResolveUpscaleVariant(int mode) const
{
	const int dlss = UpscaleVariantOf(UP_DLSS), fsr = UpscaleVariantOf(UP_FSR), xess = UpscaleVariantOf(UP_XESS), fsr1 = UpscaleVariantOf(UP_FSR1);
	int v = UpscaleVariantOf(mode);
	if (v < 0 && mode != UP_FSR1)   // Auto, or the asked vendor is not here: this adapter's own first, then the cross-vendor ones
	{
		const ADAPTER_VENDOR vendor = device ? device->GetAdapterInfo().Vendor : ADAPTER_VENDOR_UNKNOWN;
		if (vendor == ADAPTER_VENDOR_NVIDIA && dlss >= 0) v = dlss;
		else if (vendor == ADAPTER_VENDOR_AMD && fsr >= 0) v = fsr;
		else if (vendor == ADAPTER_VENDOR_INTEL && xess >= 0) v = xess;
		else if (dlss >= 0) v = dlss;
		else if (fsr >= 0) v = fsr;
		else if (xess >= 0) v = xess;
	}
	if (v < 0) v = fsr1;
	return v;
}

// Resolve this pass's upscaling (the chain stage + the device's variants) and its render size.
// Native (quality 0) renders the output size: DLAA for a temporal variant, a no-op for a spatial one.
bool NukeDiligent::Impl::UpscaleInternalSize(int ow, int oh, int& iw, int& ih)
{
	iw = ow; ih = oh;
	curUpVariant = -1; curUpTemporal = false; curUpQuality = 1; curUpSharp = 0.0f;
	if (ow <= 0 || oh <= 0 || upVariants.empty()) return false;
	int quality = 1; float sharp = 0.0f;
	const int mode = UpscaleWanted(quality, sharp);
	if (mode < 0) return false;
	const int v = ResolveUpscaleVariant(mode);
	if (v < 0) return false;
	const UpVariant& var = upVariants[v];
	if (quality == 0 && !var.temporal) return false;   // spatial at native = nothing to do
	curUpVariant = v; curUpTemporal = var.temporal; curUpQuality = quality; curUpSharp = sharp;
	if (quality == 0) return true;
	// 1..4 = Quality, Balanced, Performance, Ultra Performance: the vendor's own ratio when it answers.
	static const float kRatio[5] = { 1.0f, 1.5f, 1.7f, 2.0f, 3.0f };
	int rw = (int)((float)ow / kRatio[quality] + 0.5f), rh = (int)((float)oh / kRatio[quality] + 0.5f);
	if (var.kind == KIND_DILIGENT && srFactory)
	{
		SuperResolutionSourceSettingsAttribs a;
		a.VariantId = srVariants[var.index].VariantId;
		a.OutputWidth = (Uint32)ow; a.OutputHeight = (Uint32)oh;
		a.OutputFormat = HDR_FMT;
		a.Flags = var.temporal ? SUPER_RESOLUTION_FLAG_AUTO_EXPOSURE : SUPER_RESOLUTION_FLAG_NONE;
		static const SUPER_RESOLUTION_OPTIMIZATION_TYPE kOpt[4] = {
			SUPER_RESOLUTION_OPTIMIZATION_TYPE_MAX_QUALITY, SUPER_RESOLUTION_OPTIMIZATION_TYPE_BALANCED,
			SUPER_RESOLUTION_OPTIMIZATION_TYPE_HIGH_PERFORMANCE, SUPER_RESOLUTION_OPTIMIZATION_TYPE_MAX_PERFORMANCE };
		a.OptimizationType = kOpt[quality - 1];
		SuperResolutionSourceSettings s;
		srFactory->GetSourceSettings(a, s);
		if (s.OptimalInputWidth > 0 && s.OptimalInputHeight > 0) { rw = (int)s.OptimalInputWidth; rh = (int)s.OptimalInputHeight; }
	}
#ifdef _WIN32
	else if (var.kind == KIND_FFX && useVulkan)
	{
		nukediligent::FfxVkQueryRenderSize(quality, ow, oh, rw, rh);
	}
#endif
#if NUKE_FFX
	else if (var.kind == KIND_FFX)
	{
		uint32_t w = 0, h = 0;
		ffxQueryDescUpscaleGetRenderResolutionFromQualityMode q{};
		q.header.type = FFX_API_QUERY_DESC_TYPE_UPSCALE_GETRENDERRESOLUTIONFROMQUALITYMODE;
		q.displayWidth = (uint32_t)ow; q.displayHeight = (uint32_t)oh; q.qualityMode = FfxQualityOf(quality);
		q.pOutRenderWidth = &w; q.pOutRenderHeight = &h;
		if (Ffx().Query(nullptr, &q.header) == FFX_API_RETURN_OK && w > 0 && h > 0) { rw = (int)w; rh = (int)h; }
	}
#endif
#ifdef _WIN32
	else if (var.kind == KIND_XESS && useVulkan)
	{
		nukediligent::XessVkQueryRenderSize(quality, ow, oh, rw, rh);
	}
#endif
#if NUKE_XESS
	else if (var.kind == KIND_XESS)
	{
		if (!xessProbe)
		{
			RefCntAutoPtr<IRenderDeviceD3D12> d12(device, IID_RenderDeviceD3D12);
			xess_context_handle_t h = nullptr;
			if (d12 && Xess().D3D12CreateContext(d12->GetD3D12Device(), &h) == XESS_RESULT_SUCCESS) xessProbe = h;
		}
		if (xessProbe)
		{
			xess_2d_t out{ (uint32_t)ow, (uint32_t)oh }, opt{}, mn{}, mx{};
			if (Xess().GetOptimalInputResolution((xess_context_handle_t)xessProbe, &out, XessQualityOf(quality), &opt, &mn, &mx) == XESS_RESULT_SUCCESS && opt.x > 0 && opt.y > 0)
			{ rw = (int)opt.x; rh = (int)opt.y; }
		}
	}
#endif
	iw = std::max(1, std::min(rw, ow)); ih = std::max(1, std::min(rh, oh));
	return true;
}

// The variant's jitter for this frame (pixels): the upscaler's own pattern when one is live for
// this camera, else Halton (2,3) over the phase count the scale asks for (8 x scale^2).
void NukeDiligent::Impl::UpscaleJitter(float& jx, float& jy)
{
	if (!curUpTemporal) return;
	auto it = upscaleStates.find(curCamKey);
	if (it == upscaleStates.end()) return;
	UpscaleState& us = it->second;
	if (us.kind == KIND_DILIGENT && us.sr) { us.sr->GetJitterOffset((Uint32)taaFrame, jx, jy); return; }
#ifdef _WIN32
	if (us.kind == KIND_FFX && us.ffx && useVulkan)
	{
		if (nukediligent::FfxVkJitter(us, taaFrame, jx, jy)) return;
	}
#endif
#if NUKE_FFX
	if (us.kind == KIND_FFX && us.ffx && !useVulkan)
	{
		float x = 0.0f, y = 0.0f;
		ffxQueryDescUpscaleGetJitterOffset q{};
		q.header.type = FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTEROFFSET;
		q.index = taaFrame; q.phaseCount = std::max(us.phaseCount, 1); q.pOutX = &x; q.pOutY = &y;
		if (Ffx().Query(&us.ffx, &q.header) == FFX_API_RETURN_OK) { jx = x; jy = y; return; }
	}
#endif
	const int phase = std::max(us.phaseCount, 1);
	const int idx = (taaFrame % phase) + 1;
	jx = Halton(idx, 2) - 0.5f; jy = Halton(idx, 3) - 0.5f;
}

void NukeDiligent::Impl::DestroyUpscaler(UpscaleState& us)
{
	us.sr.Release();
#ifdef _WIN32
	if (us.ffx && useVulkan) nukediligent::FfxVkUpscaleDestroy(us);
	if (us.xess && useVulkan) nukediligent::XessVkDestroy(us);
#endif
#if NUKE_FFX
	if (us.ffx) { Ffx().DestroyContext(&us.ffx, nullptr); us.ffx = nullptr; }
#endif
#if NUKE_XESS
	if (us.xess) { Xess().DestroyContext((xess_context_handle_t)us.xess); us.xess = nullptr; }
#endif
	Trash(us.out); us.out.Release(); Trash(us.ldr); us.ldr.Release(); Trash(us.in); us.in.Release(); Trash(us.react); us.react.Release();
	us.kind = -1; us.ready = false; us.xessBuilt = false;
}

NukeDiligent::Impl::UpscaleState& NukeDiligent::Impl::EnsureUpscaler(int iw, int ih, int ow, int oh, TEXTURE_FORMAT fmt)
{
	UpscaleState& us = upscaleStates[curCamKey];
	us.lastUsed = frameId;
	const bool same = us.kind >= 0 && us.variant == curUpVariant && us.quality == curUpQuality && us.inW == iw && us.inH == ih && us.outW == ow && us.outH == oh && us.fmt == fmt;
	if (same) return us;
	DestroyUpscaler(us);
	us.variant = curUpVariant; us.quality = curUpQuality; us.inW = iw; us.inH = ih; us.outW = ow; us.outH = oh; us.fmt = fmt; us.fresh = true;
	us.phaseCount = (int)std::ceil(8.0 * ((double)ow / (double)std::max(iw, 1)) * ((double)ow / (double)std::max(iw, 1)));
	if (curUpVariant < 0 || curUpVariant >= (int)upVariants.size()) return us;
	const UpVariant& var = upVariants[curUpVariant];
	us.kind = var.kind;
	if (var.kind == KIND_DILIGENT && srFactory)
	{
		const SuperResolutionInfo& v = srVariants[var.index];
		SuperResolutionDesc d;
		d.Name = "Nuke upscaler";
		d.VariantId = v.VariantId;
		d.InputWidth = (Uint32)iw; d.InputHeight = (Uint32)ih; d.OutputWidth = (Uint32)ow; d.OutputHeight = (Uint32)oh;
		d.ColorFormat = fmt; d.OutputFormat = fmt;
		const bool sharpCap = curUpTemporal ? (v.TemporalCapFlags & SUPER_RESOLUTION_TEMPORAL_CAP_FLAG_SHARPNESS) != 0
		                                    : (v.SpatialCapFlags & SUPER_RESOLUTION_SPATIAL_CAP_FLAG_SHARPNESS) != 0;
		d.Flags = SUPER_RESOLUTION_FLAG_NONE;
		if (curUpSharp > 0.0f && sharpCap) d.Flags = d.Flags | SUPER_RESOLUTION_FLAG_ENABLE_SHARPENING;
		if (curUpTemporal)
		{
			d.Flags = d.Flags | SUPER_RESOLUTION_FLAG_AUTO_EXPOSURE;
			d.DepthFormat = TEX_FORMAT_R32_FLOAT;    // the prepass D32's SRV format
			d.MotionFormat = TEX_FORMAT_RG16_FLOAT;  // the prepass velocity
			if (v.TemporalCapFlags & SUPER_RESOLUTION_TEMPORAL_CAP_FLAG_REACTIVE_MASK) d.ReactiveMaskFormat = TEX_FORMAT_R8_UNORM;   // upscale_reactive.ps
		}
		srFactory->CreateSuperResolution(d, &us.sr);
		us.ready = us.sr != nullptr;
	}
#ifdef _WIN32
	else if (var.kind == KIND_FFX && useVulkan)
	{
		us.ready = nukediligent::FfxVkUpscaleCreate(*this, us, iw, ih, ow, oh);
	}
#endif
#if NUKE_FFX
	else if (var.kind == KIND_FFX)
	{
		RefCntAutoPtr<IRenderDeviceD3D12> d12(device, IID_RenderDeviceD3D12);
		ffxCreateBackendDX12Desc be{};
		be.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;
		be.device = d12 ? d12->GetD3D12Device() : nullptr;
		ffxCreateContextDescUpscaleVersion ver{};
		ver.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION;
		ver.version = FFX_UPSCALER_VERSION;
		ver.header.pNext = &be.header;
		ffxCreateContextDescUpscale cd{};
		cd.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
		cd.header.pNext = &ver.header;
		cd.flags = FFX_UPSCALE_ENABLE_AUTO_EXPOSURE | (hdr ? FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE : FFX_UPSCALE_ENABLE_NON_LINEAR_COLORSPACE);
		cd.maxRenderSize = { (uint32_t)iw, (uint32_t)ih };
		cd.maxUpscaleSize = { (uint32_t)ow, (uint32_t)oh };
		cd.fpMessage = FfxMessage;
		if (be.device && Ffx().CreateContext(&us.ffx, &cd.header, nullptr) == FFX_API_RETURN_OK && us.ffx)
		{
			int32_t phase = 0;
			ffxQueryDescUpscaleGetJitterPhaseCount q{};
			q.header.type = FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT;
			q.renderWidth = (uint32_t)iw; q.displayWidth = (uint32_t)ow; q.pOutPhaseCount = &phase;
			if (Ffx().Query(&us.ffx, &q.header) == FFX_API_RETURN_OK && phase > 0) us.phaseCount = phase;
			us.ready = true;
		}
		else us.ffx = nullptr;
	}
#endif
#ifdef _WIN32
	else if (var.kind == KIND_XESS && useVulkan)
	{
		nukediligent::XessVkUpscaleCreate(*this, us, ow, oh);
	}
#endif
#if NUKE_XESS
	else if (var.kind == KIND_XESS)
	{
		RefCntAutoPtr<IRenderDeviceD3D12> d12(device, IID_RenderDeviceD3D12);
		xess_context_handle_t h = nullptr;
		if (d12 && Xess().D3D12CreateContext(d12->GetD3D12Device(), &h) == XESS_RESULT_SUCCESS && h)
		{
			us.xess = h;
			if (Xess().SetLoggingCallback) Xess().SetLoggingCallback(h, XESS_LOGGING_LEVEL_WARNING, XessLog);
			// The pipelines build in the background (the warm-up rule: nothing compiles on the draw
			// path); Init lands when they are done - RunUpscaleTemporal polls.
			const uint32_t flags = XESS_INIT_FLAG_ENABLE_AUTOEXPOSURE | XESS_INIT_FLAG_RESPONSIVE_PIXEL_MASK | (hdr ? 0u : (uint32_t)XESS_INIT_FLAG_LDR_INPUT_COLOR);
			us.xessFlags = flags;
			if (Xess().D3D12BuildPipelines && Xess().GetPipelineBuildStatus) Xess().D3D12BuildPipelines(h, nullptr, false, flags);
			else us.xessBuilt = true;
		}
	}
#endif
	if (us.ready || us.xess)
		cout << "[NukeDiligent]\tupscaler: " << var.name << " " << iw << "x" << ih << " -> " << ow << "x" << oh << " (quality " << curUpQuality << ")" << endl;
	else
		cout << "[NukeDiligent]\tupscaler: " << var.name << " refused " << iw << "x" << ih << " -> " << ow << "x" << oh << endl;
	if (curUpTemporal && (us.ready || us.xess))
	{
		TextureDesc td; td.Name = "Upscale Out"; td.Type = RESOURCE_DIM_TEX_2D; td.Width = (Uint32)ow; td.Height = (Uint32)oh;
		td.Format = fmt; td.BindFlags = BIND_UNORDERED_ACCESS | BIND_SHADER_RESOURCE | BIND_RENDER_TARGET;
		device->CreateTexture(td, nullptr, &us.out);
	}
	return us;
}

// The reactive mask of this pass (internal size, R8): the water surface from the G-buffer flag
// and the pass's sprite coverage (upscale_reactive.ps). Null when the pipeline is not up yet.
ITextureView* NukeDiligent::Impl::BuildReactiveMask(UpscaleState& us)
{
	if (!upReactPSO || !upReactSRB || !gbufSRV) return nullptr;
	if (!us.react || (int)us.react->GetDesc().Width != curRTW || (int)us.react->GetDesc().Height != curRTH)
	{
		Trash(us.react); us.react.Release();
		TextureDesc td; td.Name = "Upscale Reactive"; td.Type = RESOURCE_DIM_TEX_2D; td.Width = (Uint32)curRTW; td.Height = (Uint32)curRTH;
		td.Format = TEX_FORMAT_R8_UNORM; td.BindFlags = BIND_RENDER_TARGET | BIND_SHADER_RESOURCE;
		device->CreateTexture(td, nullptr, &us.react);
		if (!us.react) return nullptr;
	}
	ITextureView* zero = coverZeroTex ? coverZeroTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE) : nullptr;
	if (auto* v = upReactSRB->GetVariableByName(SHADER_TYPE_PIXEL, "g_GBuffer")) v->Set(gbufSRV);
	if (auto* v = upReactSRB->GetVariableByName(SHADER_TYPE_PIXEL, "g_Cover"))   v->Set(coverSRV ? coverSRV : zero);
	if (auto* v = upReactSRB->GetVariableByName(SHADER_TYPE_PIXEL, "g_CoverWorld")) v->Set(gbufCoverSRV ? gbufCoverSRV : zero);
	ITextureView* rtv = us.react->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET);
	context->SetRenderTargets(1, &rtv, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	Viewport vp; vp.TopLeftX = 0; vp.TopLeftY = 0; vp.Width = (float)curRTW; vp.Height = (float)curRTH; vp.MinDepth = 0; vp.MaxDepth = 1;
	context->SetViewports(1, &vp, curRTW, curRTH);
	context->SetPipelineState(upReactPSO);
	context->CommitShaderResources(upReactSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	DrawAttribs da{3, DRAW_FLAG_VERIFY_STATES};
	context->Draw(da);
	context->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE);
	return us.react->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
}

// Temporal: the HDR chain image (internal size) -> the reconstructed output-size HDR image.
ITextureView* NukeDiligent::Impl::RunUpscaleTemporal(ITextureView* srcSRV)
{
	if (!curUpTemporal || !srcSRV || !gbufDepthSRV || !gbufVelSRV) return nullptr;
	UpscaleState& us = EnsureUpscaler(curRTW, curRTH, outW, outH, HDR_FMT);   // the chain's format (the media / scratch textures)
	if (!us.out) return nullptr;
	if (srcSRV->GetTexture()->GetDesc().Format != HDR_FMT)   // the bare scene target (no media ran): convert into the chain format first
	{
		if (!us.in || (int)us.in->GetDesc().Width != curRTW || (int)us.in->GetDesc().Height != curRTH)
		{
			Trash(us.in); us.in.Release();
			TextureDesc td; td.Name = "Upscale In"; td.Type = RESOURCE_DIM_TEX_2D; td.Width = (Uint32)curRTW; td.Height = (Uint32)curRTH;
			td.Format = HDR_FMT; td.BindFlags = BIND_RENDER_TARGET | BIND_SHADER_RESOURCE;
			device->CreateTexture(td, nullptr, &us.in);
			if (!us.in) return nullptr;
		}
		context->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE);
		BlitTexture(srcSRV, us.in);
		srcSRV = us.in->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
	}
	const float dt = Time::getSingleton() ? (float)std::min(std::max(Time::getSingleton()->gameDelta, 0.0), 0.25) : 0.0f;
	ITextureView* reactSRV = BuildReactiveMask(us);
	ITextureView* outUAV = us.out->GetDefaultView(TEXTURE_VIEW_UNORDERED_ACCESS);
	context->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE);
	if (us.kind == KIND_DILIGENT)
	{
		if (!us.sr) return nullptr;
		ExecuteSuperResolutionAttribs a;
		a.pContext = context;
		a.pColorTextureSRV = srcSRV;
		a.pDepthTextureSRV = gbufDepthSRV;
		a.pMotionVectorsSRV = gbufVelSRV;
		a.pOutputTextureView = outUAV;
		if (us.sr->GetDesc().ReactiveMaskFormat != TEX_FORMAT_UNKNOWN) a.pReactiveMaskTextureSRV = reactSRV;
		a.JitterX = curJitterX; a.JitterY = curJitterY;
		// gbuffer.ps writes UV-space (current - previous); the upscaler wants pixels, previous - current.
		a.MotionVectorScaleX = -(float)curRTW; a.MotionVectorScaleY = -(float)curRTH;
		a.CameraNear = curNear; a.CameraFar = curFar;
		a.CameraFovAngleVert = curFovY;
		a.TimeDeltaInSeconds = dt;
		a.Sharpness = curUpSharp;
		a.ResetHistory = us.fresh ? True : False;
		us.sr->Execute(a);
	}
#ifdef _WIN32
	else if (us.kind == KIND_FFX && useVulkan)
	{
		if (!us.ffx || !nukediligent::FfxVkUpscaleDispatch(*this, us, srcSRV, reactSRV, dt)) return nullptr;
	}
#endif
#if NUKE_FFX
	else if (us.kind == KIND_FFX)
	{
		if (!us.ffx) return nullptr;
		RefCntAutoPtr<IDeviceContextD3D12> ctx12(context, IID_DeviceContextD3D12);
		if (!ctx12) return nullptr;
		// Every input in the shader-read state, the output writable - through Diligent, so its tracking agrees.
		StateTransitionDesc bars[5] = {
			{srcSRV->GetTexture(),       RESOURCE_STATE_UNKNOWN, RESOURCE_STATE_SHADER_RESOURCE,  STATE_TRANSITION_FLAG_UPDATE_STATE},
			{gbufDepthSRV->GetTexture(), RESOURCE_STATE_UNKNOWN, RESOURCE_STATE_SHADER_RESOURCE,  STATE_TRANSITION_FLAG_UPDATE_STATE},
			{gbufVelSRV->GetTexture(),   RESOURCE_STATE_UNKNOWN, RESOURCE_STATE_SHADER_RESOURCE,  STATE_TRANSITION_FLAG_UPDATE_STATE},
			{us.out,                     RESOURCE_STATE_UNKNOWN, RESOURCE_STATE_UNORDERED_ACCESS, STATE_TRANSITION_FLAG_UPDATE_STATE},
			{reactSRV ? reactSRV->GetTexture() : us.out, RESOURCE_STATE_UNKNOWN, reactSRV ? RESOURCE_STATE_SHADER_RESOURCE : RESOURCE_STATE_UNORDERED_ACCESS, STATE_TRANSITION_FLAG_UPDATE_STATE} };
		context->TransitionResourceStates(reactSRV ? 5 : 4, bars);
		auto res12 = [](ITexture* t) -> ID3D12Resource* { RefCntAutoPtr<ITextureD3D12> t12(t, IID_TextureD3D12); return t12 ? t12->GetD3D12Texture() : nullptr; };
		ffxDispatchDescUpscale dd{};
		dd.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
		dd.commandList = ctx12->GetD3D12CommandList();
		dd.color = ffxApiGetResourceDX12(res12(srcSRV->GetTexture()), FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
		dd.depth = ffxApiGetResourceDX12(res12(gbufDepthSRV->GetTexture()), FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
		dd.motionVectors = ffxApiGetResourceDX12(res12(gbufVelSRV->GetTexture()), FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
		dd.output = ffxApiGetResourceDX12(res12(us.out), FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
		if (reactSRV) dd.reactive = ffxApiGetResourceDX12(res12(reactSRV->GetTexture()), FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
		dd.jitterOffset = { curJitterX, curJitterY };
		dd.motionVectorScale = { -(float)curRTW, -(float)curRTH };   // UV (current - previous) -> pixels (previous - current)
		dd.renderSize = { (uint32_t)curRTW, (uint32_t)curRTH };
		dd.upscaleSize = { (uint32_t)outW, (uint32_t)outH };
		dd.enableSharpening = curUpSharp > 0.0f;
		dd.sharpness = curUpSharp;
		dd.frameTimeDelta = dt * 1000.0f;
		dd.preExposure = 1.0f;
		dd.reset = us.fresh;
		dd.cameraNear = curNear; dd.cameraFar = curFar;
		dd.cameraFovAngleVertical = curFovY;
		dd.viewSpaceToMetersFactor = 1.0f;
		dd.flags = hdr ? 0u : (uint32_t)FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_SRGB;
		const ffxReturnCode_t rc = Ffx().Dispatch(&us.ffx, &dd.header);
		if (rc != FFX_API_RETURN_OK) cout << "[NukeDiligent]\tFSR dispatch failed: " << rc << endl;
		ctx12->Flush();   // the vendor recorded its own descriptor heaps / state: a fresh command list
	}
#endif
#ifdef _WIN32
	else if (us.kind == KIND_XESS && useVulkan)
	{
		if (!us.xess || !nukediligent::XessVkUpscaleDispatch(*this, us, srcSRV, reactSRV)) return nullptr;
	}
#endif
#if NUKE_XESS
	else if (us.kind == KIND_XESS)
	{
		if (!us.xess) return nullptr;
		xess_context_handle_t h = (xess_context_handle_t)us.xess;
		if (!us.ready)
		{
			if (!us.xessBuilt)
			{
				const xess_result_t st = Xess().GetPipelineBuildStatus(h);
				if (st == XESS_RESULT_ERROR_OPERATION_IN_PROGRESS) return nullptr;   // still compiling: this frame stretches
				us.xessBuilt = true;
			}
			xess_d3d12_init_params_t ip{};
			ip.outputResolution = { (uint32_t)outW, (uint32_t)outH };
			ip.qualitySetting = XessQualityOf(curUpQuality);
			ip.initFlags = us.xessFlags;
			const xess_result_t r = Xess().D3D12Init(h, &ip);
			if (r != XESS_RESULT_SUCCESS) { cout << "[NukeDiligent]\tXeSS init failed: " << (int)r << endl; DestroyUpscaler(us); return nullptr; }
			Xess().SetVelocityScale(h, -(float)curRTW, -(float)curRTH);   // UV (current - previous) -> pixels (previous - current)
			us.ready = true;
		}
		RefCntAutoPtr<IDeviceContextD3D12> ctx12(context, IID_DeviceContextD3D12);
		if (!ctx12) return nullptr;
		// XeSS wants its inputs in exactly NON_PIXEL_SHADER_RESOURCE and the output in UNORDERED_ACCESS.
		ctx12->TransitionTextureState(srcSRV->GetTexture(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
		ctx12->TransitionTextureState(gbufDepthSRV->GetTexture(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
		ctx12->TransitionTextureState(gbufVelSRV->GetTexture(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
		ctx12->TransitionTextureState(us.out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
		if (reactSRV) ctx12->TransitionTextureState(reactSRV->GetTexture(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
		auto res12 = [](ITexture* t) -> ID3D12Resource* { RefCntAutoPtr<ITextureD3D12> t12(t, IID_TextureD3D12); return t12 ? t12->GetD3D12Texture() : nullptr; };
		xess_d3d12_execute_params_t ep{};
		ep.pResponsivePixelMaskTexture = reactSRV ? res12(reactSRV->GetTexture()) : nullptr;
		ep.pColorTexture = res12(srcSRV->GetTexture());
		ep.pVelocityTexture = res12(gbufVelSRV->GetTexture());
		ep.pDepthTexture = res12(gbufDepthSRV->GetTexture());
		ep.pOutputTexture = res12(us.out);
		ep.jitterOffsetX = curJitterX; ep.jitterOffsetY = curJitterY;
		ep.exposureScale = 1.0f;
		ep.resetHistory = us.fresh ? 1u : 0u;
		ep.inputWidth = (uint32_t)curRTW; ep.inputHeight = (uint32_t)curRTH;
		const xess_result_t r = Xess().D3D12Execute(h, ctx12->GetD3D12CommandList(), &ep);
		if (r != XESS_RESULT_SUCCESS) cout << "[NukeDiligent]\tXeSS execute failed: " << (int)r << endl;
		ctx12->Flush();
	}
#endif
	else return nullptr;
	us.fresh = false;
	lastInstBind.pso = nullptr;   // foreign command recording invalidated the bind cache
	return us.out->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
}

// Spatial (FSR 1): tonemap the chain image at the internal size into an LDR texture of the
// output's format, then the spatial pass into dstRTV at the output size. False = not run.
bool NukeDiligent::Impl::RunUpscaleSpatial(ITextureView* chainSRV, ITextureView* dstRTV, bool toBackbuffer)
{
	if (curUpVariant < 0 || curUpTemporal || !chainSRV || !dstRTV) return false;
	const TEXTURE_FORMAT fmt = dstRTV->GetTexture()->GetDesc().Format;
	UpscaleState& us = EnsureUpscaler(curRTW, curRTH, outW, outH, fmt);
	if (!us.sr) return false;
	if (!us.ldr || (int)us.ldr->GetDesc().Width != curRTW || (int)us.ldr->GetDesc().Height != curRTH || us.ldr->GetDesc().Format != fmt)
	{
		Trash(us.ldr); us.ldr.Release();
		TextureDesc td; td.Name = "Upscale LDR"; td.Type = RESOURCE_DIM_TEX_2D; td.Width = (Uint32)curRTW; td.Height = (Uint32)curRTH;
		td.Format = fmt; td.BindFlags = BIND_RENDER_TARGET | BIND_SHADER_RESOURCE;
		device->CreateTexture(td, nullptr, &us.ldr);
		if (!us.ldr) return false;
	}
	RunPostPass(chainSRV, us.ldr->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET), curRTW, curRTH, toBackbuffer);
	{
		ExecuteSuperResolutionAttribs a;
		a.pContext = context;
		a.pColorTextureSRV = us.ldr->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
		a.pOutputTextureView = dstRTV;
		a.Sharpness = curUpSharp;
		us.sr->Execute(a);
	}
	us.fresh = false;
	lastInstBind.pso = nullptr;
	// The gizmo lines / HUD that follow draw on the bound output.
	context->SetRenderTargets(1, &dstRTV, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	Viewport vp; vp.TopLeftX = 0; vp.TopLeftY = 0; vp.Width = (float)outW; vp.Height = (float)outH; vp.MinDepth = 0; vp.MaxDepth = 1;
	context->SetViewports(1, &vp, outW, outH);
	return true;
}

void NukeDiligent::Impl::PruneUpscaleStates()
{
	const uint64_t kStale = 120;
	for (auto it = upscaleStates.begin(); it != upscaleStates.end();)
	{
		if (frameId - it->second.lastUsed > kStale) { DestroyUpscaler(it->second); it = upscaleStates.erase(it); }
		else ++it;
	}
}

// Shutdown: every upscaler and the vendor contexts before the device they wrap.
void NukeDiligent::Impl::ShutdownUpscalers()
{
	for (auto& kv : upscaleStates) DestroyUpscaler(kv.second);
	upscaleStates.clear();
#if NUKE_XESS
	if (xessProbe) { Xess().DestroyContext((xess_context_handle_t)xessProbe); xessProbe = nullptr; }
#endif
#ifdef _WIN32
	nukediligent::XessVkShutdown();
#endif
	srFactory.Release();
	srVariants.clear(); upVariants.clear();
}

// 4.2: the status the engine's typed API reports (Game.ActiveUpscaler / UpscaleInfo /
// UpscalerAvailable). Vendors in the engine's UpscaleMode numbering = the shader's mode + 1.
void NukeDiligent::getUpscaleStatus(NukeUpscaleStatus* out)
{
	if (!out) return;
	*out = NukeUpscaleStatus();
	Impl& d = *m_impl;
	auto vendorOf = [&](int v) -> int
	{
		if (v < 0 || v >= (int)d.upVariants.size()) return 0;
		const Impl::UpVariant& var = d.upVariants[v];
		if (var.kind == KIND_FFX) return UP_FSR + 1;
		if (var.kind == KIND_XESS) return UP_XESS + 1;
		return var.temporal ? UP_DLSS + 1 : UP_FSR1 + 1;
	};
	if (d.statUpVariant >= 0 && d.frameId - d.statUpFrame <= 2)   // a back-buffer camera resolved it this frame or the last
	{
		out->upscaler = vendorOf(d.statUpVariant); out->quality = d.statUpQuality;
		out->renderW = d.statInW; out->renderH = d.statInH; out->outW = d.statOutW; out->outH = d.statOutH;
		strncpy(out->upscalerName, d.upVariants[d.statUpVariant].name.c_str(), sizeof(out->upscalerName) - 1);
	}
	for (int m = UP_DLSS; m <= UP_FSR1; ++m)
		if (d.UpscaleVariantOf(m) >= 0) out->offeredUpscalers |= 1u << (m + 1);
	out->offeredGenerators = d.FrameGenOffered();
	if (d.fg.attached)
	{
		out->generator = d.fg.kind == FG_DLSS ? UP_DLSS + 1 : d.fg.kind == FG_FSR ? UP_FSR + 1 : d.fg.kind == FG_XESS ? UP_XESS + 1 : 0;
		out->generatedFrames = std::max(1, d.fg.framesLive);
		out->presentsPerFrame = (float)d.fg.ratio;
		strncpy(out->generatorName, nukediligent::FrameGenName(d.fg.kind), sizeof(out->generatorName) - 1);
	}
	out->renderedFps = (float)d.fg.fpsRendered;
}
