#pragma once
// Shared between NukeDiligent_FrameGen.cpp (the vendor-neutral flow + the D3D12 vendors) and
// NukeDiligent_FrameGenVk.cpp (the Vulkan side: Streamline's Vulkan proxies, AMD's Vulkan runtime).
#include "NukeDiligentImpl.h"
#include <string>
#if defined(_WIN32) && NUKE_STREAMLINE
#include <sl.h>
#include <sl_consts.h>
#include <sl_dlss_g.h>
#include <sl_reflex.h>
#include <sl_pcl.h>
#include <sl_helpers.h>
#endif

enum { UP_AUTO = 0, UP_DLSS = 1, UP_FSR = 2, UP_XESS = 3, UP_FSR1 = 4 };   // upscale.post.hlsl g_Mode
enum { FG_NONE = 0, FG_FSR = 1, FG_DLSS = 2, FG_XESS = 3 };

namespace nukediligent {

const char* FrameGenName(int kind);

#ifdef _WIN32
std::wstring ModuleDir();                       // NukeDiligent_Upscale.cpp
HMODULE      LoadVendorDll(const wchar_t* name);

#if NUKE_STREAMLINE
// sl.interposer.dll by name; the feature functions arrive after slInit.
struct SlFns
{
	HMODULE dll = nullptr;
	PFun_slInit*                Init = nullptr;
	PFun_slShutdown*            Shutdown = nullptr;
	PFun_slSetD3DDevice*        SetD3DDevice = nullptr;
	PFun_slUpgradeInterface*    UpgradeInterface = nullptr;
	PFun_slIsFeatureSupported*  IsFeatureSupported = nullptr;
	PFun_slSetFeatureLoaded*    SetFeatureLoaded = nullptr;
	PFun_slGetFeatureFunction*  GetFeatureFunction = nullptr;
	PFun_slSetTagForFrame*      SetTagForFrame = nullptr;
	PFun_slSetConstants*        SetConstants = nullptr;
	PFun_slGetNewFrameToken*    GetNewFrameToken = nullptr;
	PFun_slFreeResources*       FreeResources = nullptr;
	PFun_slDLSSGSetOptions*     DLSSGSetOptions = nullptr;
	PFun_slDLSSGGetState*       DLSSGGetState = nullptr;
	PFun_slReflexSetOptions*    ReflexSetOptions = nullptr;
	PFun_slReflexSleep*         ReflexSleep = nullptr;
	PFun_slPCLSetMarker*        PCLSetMarker = nullptr;
	bool ok() const { return dll && Init && Shutdown && SetD3DDevice && UpgradeInterface && IsFeatureSupported && SetFeatureLoaded && GetFeatureFunction && SetTagForFrame && SetConstants && GetNewFrameToken; }
};
SlFns& Sl();
// slInit (manual hooking) once per process + the DLSS-G / Reflex / PCL feature functions. On D3D12
// with the device (after creation); on Vulkan BEFORE the device (its create proxies need it).
bool SlInitOnce(NukeDiligent::Impl& d, bool vulkan, void* d3dDevice);
bool SlFeatures(NukeDiligent::Impl& d);   // the feature functions, once the device exists (D3D12: inside SlInitOnce)
void SlMark(sl::PCLMarker m, void* token);
sl::float4x4 SlMat(const Diligent::float4x4& m);
extern const sl::ViewportHandle kSlViewport;
#endif

// The Vulkan side (NukeDiligent_FrameGenVk.cpp). FSR 3.1.4 through amd_fidelityfx_vk.dll: the
// upscaler for the upscale stage and the frame-interpolation swap chain for the generator.
bool FfxVkAvailable();
bool FfxVkQueryRenderSize(int quality, int ow, int oh, int& rw, int& rh);
bool FfxVkUpscaleCreate(NukeDiligent::Impl& d, NukeDiligent::Impl::UpscaleState& us, int iw, int ih, int ow, int oh);
bool FfxVkJitter(NukeDiligent::Impl::UpscaleState& us, int index, float& jx, float& jy);
bool FfxVkUpscaleDispatch(NukeDiligent::Impl& d, NukeDiligent::Impl::UpscaleState& us, Diligent::ITextureView* srcSRV, Diligent::ITextureView* reactSRV, float dt);
void FfxVkUpscaleDestroy(NukeDiligent::Impl::UpscaleState& us);
// Intel XeSS super resolution on Vulkan (libxess.dll's VK entry points; XeSS-FG has none).
bool        XessVkAvailable();
std::string XessVkVariantName();
bool XessVkQueryRenderSize(int quality, int ow, int oh, int& rw, int& rh);
bool XessVkUpscaleCreate(NukeDiligent::Impl& d, NukeDiligent::Impl::UpscaleState& us, int ow, int oh);   // context + background pipelines
bool XessVkUpscaleDispatch(NukeDiligent::Impl& d, NukeDiligent::Impl::UpscaleState& us, Diligent::ITextureView* srcSRV, Diligent::ITextureView* reactSRV);   // false while its pipelines still build
void XessVkDestroy(NukeDiligent::Impl::UpscaleState& us);
void XessVkShutdown();
#endif   // _WIN32

}  // namespace nukediligent
