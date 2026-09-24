#include "NukeDiligent_FrameGenShared.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <vector>
#include <iostream>
#include <cstdlib>
#ifdef _WIN32
#ifndef VK_USE_PLATFORM_WIN32_KHR
#define VK_USE_PLATFORM_WIN32_KHR
#endif
#include <volk.h>                 // the loader table Diligent calls through: the vendors' swap-chain entry points replace its globals
#include "RenderDeviceVk.h"
#include "DeviceContextVk.h"
#include "TextureVk.h"
#include "TextureViewVk.h"
#include "SwapChainVk.h"          // DetachVkSwapChain / AttachVkSwapChain / GetVkSwapChainCreateInfo (NUKE PATCH: patches/DiligentCore-vk-hooks.patch)
#include "CommandQueueVk.h"
#if NUKE_XESS
#include <xess/xess.h>                        // Intel XeSS through libxess.dll: its Vulkan entry points
#include <xess/xess_vk.h>
#endif
#if NUKE_FFX_VK
#include <ffx-vk/include/ffx_api.h>            // AMD FidelityFX API 1.1.4: FSR 3.1.4 upscaler + frame interpolation for Vulkan (amd_fidelityfx_vk.dll)
#include <ffx-vk/include/ffx_api_types.h>
#include <ffx-vk/include/ffx_upscale.h>
#include <ffx-vk/include/ffx_framegeneration.h>
#include <ffx-vk/include/vk/ffx_api_vk.h>
#endif

// 4.2 frame generation + FSR on Vulkan. Diligent creates the instance, device and swap chain
// through the loader table (volk); two NUKE PATCH hooks route instance / device creation through
// this file (Streamline's proxies add what DLSS-G needs; the extra queues AMD's frame-interpolation
// swap chain wants are reserved) and the swap-chain entry points are replaced in the table
// afterwards (Streamline's proxies for the life of the device on an NVIDIA adapter; AMD's
// replacement functions while its swap chain is attached).

extern "C" PFN_vkCreateInstance g_NukeVkCreateInstance;   // patches/DiligentCore-vk-hooks.patch
extern "C" PFN_vkCreateDevice   g_NukeVkCreateDevice;

using namespace Diligent;
using namespace nukediligent;
using std::cout; using std::endl;

namespace {

#if NUKE_FFX_VK
void FfxVkMessage(uint32_t type, const wchar_t* msg)
{
	if (!msg) return;
	std::wstring w(msg); std::string s(w.begin(), w.end());
	cout << "[FFX-VK]\t" << (type == FFX_API_MESSAGE_TYPE_ERROR ? "error: " : "") << s << endl;
}
struct FfxVkFns
{
	HMODULE dll = nullptr;
	PfnFfxCreateContext  CreateContext = nullptr;
	PfnFfxDestroyContext DestroyContext = nullptr;
	PfnFfxConfigure      Configure = nullptr;
	PfnFfxQuery          Query = nullptr;
	PfnFfxDispatch       Dispatch = nullptr;
	bool ok() const { return dll && CreateContext && DestroyContext && Configure && Query && Dispatch; }
};
FfxVkFns g_ffxVk; bool g_ffxVkTried = false;
const FfxVkFns& FfxVk()
{
	if (!g_ffxVkTried)
	{
		g_ffxVkTried = true;
		g_ffxVk.dll = LoadVendorDll(L"amd_fidelityfx_vk.dll");
		if (g_ffxVk.dll)
		{
			g_ffxVk.CreateContext  = (PfnFfxCreateContext)GetProcAddress(g_ffxVk.dll, "ffxCreateContext");
			g_ffxVk.DestroyContext = (PfnFfxDestroyContext)GetProcAddress(g_ffxVk.dll, "ffxDestroyContext");
			g_ffxVk.Configure      = (PfnFfxConfigure)GetProcAddress(g_ffxVk.dll, "ffxConfigure");
			g_ffxVk.Query          = (PfnFfxQuery)GetProcAddress(g_ffxVk.dll, "ffxQuery");
			g_ffxVk.Dispatch       = (PfnFfxDispatch)GetProcAddress(g_ffxVk.dll, "ffxDispatch");
			if (g_ffxVk.ok())   // the runtime's own warnings / errors to the console (the swap-chain context has no callback of its own)
			{
				ffxConfigureDescGlobalDebug1 dbg{};
				dbg.header.type = FFX_API_CONFIGURE_DESC_TYPE_GLOBALDEBUG1;
				dbg.fpMessage = FfxVkMessage;
				dbg.debugLevel = FFX_API_CONFIGURE_GLOBALDEBUG_LEVEL_WARNINGS;
				g_ffxVk.Configure(nullptr, &dbg.header);
			}
		}
	}
	return g_ffxVk;
}
ffxReturnCode_t FfxVkFgDispatchCb(ffxDispatchDescFrameGeneration* params, void* user)
{
	return FfxVk().Dispatch((ffxContext*)user, &params->header);
}
uint32_t FfxQualityOf(int q)
{
	switch (q) { case 0: return FFX_UPSCALE_QUALITY_MODE_NATIVEAA; case 1: return FFX_UPSCALE_QUALITY_MODE_QUALITY; case 2: return FFX_UPSCALE_QUALITY_MODE_BALANCED;
	             case 3: return FFX_UPSCALE_QUALITY_MODE_PERFORMANCE; default: return FFX_UPSCALE_QUALITY_MODE_ULTRA_PERFORMANCE; }
}
#endif

#if NUKE_XESS
struct XessVkFns
{
	HMODULE dll = nullptr;
	decltype(&xessVKGetRequiredInstanceExtensions) GetInstanceExtensions = nullptr;
	decltype(&xessVKGetRequiredDeviceExtensions)   GetDeviceExtensions = nullptr;
	decltype(&xessVKGetRequiredDeviceFeatures)     GetDeviceFeatures = nullptr;
	decltype(&xessVKCreateContext)                 CreateContext = nullptr;
	decltype(&xessVKBuildPipelines)                BuildPipelines = nullptr;
	decltype(&xessVKInit)                          Init = nullptr;
	decltype(&xessVKExecute)                       Execute = nullptr;
	decltype(&xessDestroyContext)                  DestroyContext = nullptr;
	decltype(&xessGetOptimalInputResolution)       GetOptimalInputResolution = nullptr;
	decltype(&xessGetPipelineBuildStatus)          GetPipelineBuildStatus = nullptr;
	decltype(&xessSetVelocityScale)                SetVelocityScale = nullptr;
	decltype(&xessSetLoggingCallback)              SetLoggingCallback = nullptr;
	decltype(&xessGetVersion)                      GetVersion = nullptr;
	bool ok() const { return dll && GetInstanceExtensions && GetDeviceExtensions && GetDeviceFeatures && CreateContext && Init && Execute && DestroyContext && GetOptimalInputResolution && SetVelocityScale; }
};
XessVkFns g_xessVk; bool g_xessVkTried = false;
const XessVkFns& XessVk()
{
	if (!g_xessVkTried)
	{
		g_xessVkTried = true;
		g_xessVk.dll = LoadVendorDll(L"libxess.dll");
		if (g_xessVk.dll)
		{
#define NUKE_XVK(fn, sym) g_xessVk.fn = (decltype(g_xessVk.fn))GetProcAddress(g_xessVk.dll, sym)
			NUKE_XVK(GetInstanceExtensions, "xessVKGetRequiredInstanceExtensions");
			NUKE_XVK(GetDeviceExtensions, "xessVKGetRequiredDeviceExtensions");
			NUKE_XVK(GetDeviceFeatures, "xessVKGetRequiredDeviceFeatures");
			NUKE_XVK(CreateContext, "xessVKCreateContext");
			NUKE_XVK(BuildPipelines, "xessVKBuildPipelines");
			NUKE_XVK(Init, "xessVKInit");
			NUKE_XVK(Execute, "xessVKExecute");
			NUKE_XVK(DestroyContext, "xessDestroyContext");
			NUKE_XVK(GetOptimalInputResolution, "xessGetOptimalInputResolution");
			NUKE_XVK(GetPipelineBuildStatus, "xessGetPipelineBuildStatus");
			NUKE_XVK(SetVelocityScale, "xessSetVelocityScale");
			NUKE_XVK(SetLoggingCallback, "xessSetLoggingCallback");
			NUKE_XVK(GetVersion, "xessGetVersion");
#undef NUKE_XVK
		}
	}
	return g_xessVk;
}
void XessVkLog(const char* msg, xess_logging_level_t level) { if (msg) cout << "[XeSS-VK]\t" << (level >= XESS_LOGGING_LEVEL_ERROR ? "error: " : "") << msg << endl; }
xess_quality_settings_t XessVkQualityOf(int q)
{
	switch (q) { case 0: return XESS_QUALITY_SETTING_AA; case 1: return XESS_QUALITY_SETTING_QUALITY; case 2: return XESS_QUALITY_SETTING_BALANCED;
	             case 3: return XESS_QUALITY_SETTING_PERFORMANCE; default: return XESS_QUALITY_SETTING_ULTRA_PERFORMANCE; }
}
void* g_xessVkProbe = nullptr;   // a context that only answers the render-size query
// The byte size of a Vulkan feature struct by its sType (for merging a vendor's chain into the
// engine's); 0 = unknown.
size_t VkFeatureStructSize(VkStructureType t)
{
	switch (t)
	{
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES:                     return sizeof(VkPhysicalDeviceVulkan11Features);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES:                     return sizeof(VkPhysicalDeviceVulkan12Features);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES:                     return sizeof(VkPhysicalDeviceVulkan13Features);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES:            return sizeof(VkPhysicalDeviceShaderFloat16Int8Features);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES:                  return sizeof(VkPhysicalDevice16BitStorageFeatures);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES:                   return sizeof(VkPhysicalDevice8BitStorageFeatures);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES:          return sizeof(VkPhysicalDeviceSubgroupSizeControlFeatures);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES:     return sizeof(VkPhysicalDeviceShaderIntegerDotProductFeatures);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES: return sizeof(VkPhysicalDeviceShaderSubgroupExtendedTypesFeatures);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES:             return sizeof(VkPhysicalDeviceTimelineSemaphoreFeatures);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES:              return sizeof(VkPhysicalDeviceSynchronization2Features);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES:                  return sizeof(VkPhysicalDeviceMaintenance4Features);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES:          return sizeof(VkPhysicalDeviceBufferDeviceAddressFeatures);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES:            return sizeof(VkPhysicalDeviceDescriptorIndexingFeatures);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES:            return sizeof(VkPhysicalDeviceShaderAtomicInt64Features);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES:            return sizeof(VkPhysicalDeviceVulkanMemoryModelFeatures);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES:               return sizeof(VkPhysicalDeviceHostQueryResetFeatures);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES:            return sizeof(VkPhysicalDeviceScalarBlockLayoutFeatures);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFORM_BUFFER_STANDARD_LAYOUT_FEATURES: return sizeof(VkPhysicalDeviceUniformBufferStandardLayoutFeatures);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES:          return sizeof(VkPhysicalDeviceImagelessFramebufferFeatures);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DEMOTE_TO_HELPER_INVOCATION_FEATURES: return sizeof(VkPhysicalDeviceShaderDemoteToHelperInvocationFeatures);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES:              return sizeof(VkPhysicalDeviceDynamicRenderingFeatures);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2:                              return sizeof(VkPhysicalDeviceFeatures2);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MUTABLE_DESCRIPTOR_TYPE_FEATURES_EXT:    return sizeof(VkPhysicalDeviceMutableDescriptorTypeFeaturesEXT);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_FEATURES:           return sizeof(VkPhysicalDeviceInlineUniformBlockFeatures);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_TERMINATE_INVOCATION_FEATURES:    return sizeof(VkPhysicalDeviceShaderTerminateInvocationFeatures);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_CREATION_CACHE_CONTROL_FEATURES: return sizeof(VkPhysicalDevicePipelineCreationCacheControlFeatures);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIVATE_DATA_FEATURES:                   return sizeof(VkPhysicalDevicePrivateDataFeatures);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES:               return sizeof(VkPhysicalDeviceImageRobustnessFeatures);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ZERO_INITIALIZE_WORKGROUP_MEMORY_FEATURES: return sizeof(VkPhysicalDeviceZeroInitializeWorkgroupMemoryFeatures);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXTURE_COMPRESSION_ASTC_HDR_FEATURES:   return sizeof(VkPhysicalDeviceTextureCompressionASTCHDRFeatures);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES:                      return sizeof(VkPhysicalDeviceMultiviewFeatures);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTERS_FEATURES:              return sizeof(VkPhysicalDeviceVariablePointersFeatures);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROTECTED_MEMORY_FEATURES:               return sizeof(VkPhysicalDeviceProtectedMemoryFeatures);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES:       return sizeof(VkPhysicalDeviceSamplerYcbcrConversionFeatures);
	case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES:         return sizeof(VkPhysicalDeviceShaderDrawParametersFeatures);
	default: return 0;
	}
}

// A vendor's feature struct into the engine's device chain: OR-ed into the same struct when the
// engine chained one, appended as our own copy otherwise (the copies live for the process).
void MergeFeatureStruct(VkDeviceCreateInfo& ci2, VkStructureType st, const VkBool32* bits, size_t nbits)
{
	static std::vector<std::vector<uint8_t>> copies;
	const size_t sz = VkFeatureStructSize(st);
	if (!sz) { cout << "[NukeDiligent]\tVulkan: feature struct " << (int)st << " unknown here, not enabled" << endl; return; }
	const size_t n = std::min(nbits, (sz - sizeof(VkBaseOutStructure)) / sizeof(VkBool32));
	for (VkBaseOutStructure* p = (VkBaseOutStructure*)const_cast<void*>(ci2.pNext); p; p = p->pNext)
		if (p->sType == st)
		{
			VkBool32* dst = (VkBool32*)((char*)p + sizeof(VkBaseOutStructure));
			for (size_t i = 0; i < n; ++i) dst[i] |= bits[i];
			return;
		}
	copies.emplace_back(sz, 0);
	VkBaseOutStructure* c = (VkBaseOutStructure*)copies.back().data();
	c->sType = st;
	memcpy((char*)c + sizeof(VkBaseOutStructure), bits, n * sizeof(VkBool32));
	c->pNext = (VkBaseOutStructure*)const_cast<void*>(ci2.pNext);
	ci2.pNext = c;
}

// The Vulkan 1.1 / 1.2 / 1.3 aggregate feature structs a vendor hands over, split into the
// individual promoted structs the engine chains (the spec forbids both forms in one chain).
void SplitAggregateFeatures(VkDeviceCreateInfo& ci2, const VkBaseInStructure* x)
{
	auto any = [](const VkBool32* b, size_t n) { for (size_t i = 0; i < n; ++i) if (b[i]) return true; return false; };
	auto put = [&](VkStructureType st, const VkBool32* b, size_t n) { if (any(b, n)) MergeFeatureStruct(ci2, st, b, n); };
	if (x->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES)
	{
		const auto* f = (const VkPhysicalDeviceVulkan11Features*)x;
		put(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES, &f->storageBuffer16BitAccess, 4);
		put(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES, &f->multiview, 3);
		put(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTERS_FEATURES, &f->variablePointersStorageBuffer, 2);
		put(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROTECTED_MEMORY_FEATURES, &f->protectedMemory, 1);
		put(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES, &f->samplerYcbcrConversion, 1);
		put(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES, &f->shaderDrawParameters, 1);
	}
	else if (x->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES)
	{
		const auto* f = (const VkPhysicalDeviceVulkan12Features*)x;
		put(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES, &f->storageBuffer8BitAccess, 3);
		put(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES, &f->shaderBufferInt64Atomics, 2);
		put(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES, &f->shaderFloat16, 2);
		put(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES, &f->shaderInputAttachmentArrayDynamicIndexing, 20);
		put(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES, &f->scalarBlockLayout, 1);
		put(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES, &f->imagelessFramebuffer, 1);
		put(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFORM_BUFFER_STANDARD_LAYOUT_FEATURES, &f->uniformBufferStandardLayout, 1);
		put(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES, &f->shaderSubgroupExtendedTypes, 1);
		put(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SEPARATE_DEPTH_STENCIL_LAYOUTS_FEATURES, &f->separateDepthStencilLayouts, 1);
		put(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES, &f->hostQueryReset, 1);
		put(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES, &f->timelineSemaphore, 1);
		put(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES, &f->bufferDeviceAddress, 3);
		put(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES, &f->vulkanMemoryModel, 3);
		// samplerMirrorClampToEdge, drawIndirectCount, samplerFilterMinmax, shaderOutputViewportIndex /
		// Layer, subgroupBroadcastDynamicId: core 1.2 without a struct of their own (extension-gated
		// in 1.1); the engine turns the ones it uses on itself.
	}
	else if (x->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES)
	{
		const auto* f = (const VkPhysicalDeviceVulkan13Features*)x;
		put(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES, &f->robustImageAccess, 1);
		put(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_FEATURES, &f->inlineUniformBlock, 2);
		put(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_CREATION_CACHE_CONTROL_FEATURES, &f->pipelineCreationCacheControl, 1);
		put(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIVATE_DATA_FEATURES, &f->privateData, 1);
		put(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DEMOTE_TO_HELPER_INVOCATION_FEATURES, &f->shaderDemoteToHelperInvocation, 1);
		put(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_TERMINATE_INVOCATION_FEATURES, &f->shaderTerminateInvocation, 1);
		put(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES, &f->subgroupSizeControl, 2);
		put(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES, &f->synchronization2, 1);
		put(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXTURE_COMPRESSION_ASTC_HDR_FEATURES, &f->textureCompressionASTC_HDR, 1);
		put(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ZERO_INITIALIZE_WORKGROUP_MEMORY_FEATURES, &f->shaderZeroInitializeWorkgroupMemory, 1);
		put(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES, &f->dynamicRendering, 1);
		put(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES, &f->shaderIntegerDotProduct, 1);
		put(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES, &f->maintenance4, 1);
	}
}
#endif

// Everything the hooks and the replacements share.
struct VkSide
{
	PFN_vkGetInstanceProcAddr slGipa = nullptr;   // Streamline's Vulkan entry points (sl.interposer.dll exports them)
	PFN_vkGetDeviceProcAddr   slGdpa = nullptr;
	VkInstance       instance = VK_NULL_HANDLE;
	VkDevice         device = VK_NULL_HANDLE;
	VkPhysicalDevice physDev = VK_NULL_HANDLE;
	// The extra queues for AMD's swap chain (reserved in the create-device hook).
	bool     wantQueues = false, haveQueues = false;
	uint32_t gfxFamily = ~0u, computeFamily = ~0u, presentIndex = 0, acquireIndex = 0, computeIndex = 0;
	VkQueue  gameQ = VK_NULL_HANDLE, presentQ = VK_NULL_HANDLE, acquireQ = VK_NULL_HANDLE, computeQ = VK_NULL_HANDLE;
	uint32_t gameFamily = ~0u;
	// The swap-chain entry points in place after device creation (the loader's or Streamline's):
	// AMD's replacement goes over them while attached and comes off again.
	PFN_vkCreateSwapchainKHR    baseCreate = nullptr;
	PFN_vkDestroySwapchainKHR   baseDestroy = nullptr;
	PFN_vkGetSwapchainImagesKHR baseGetImages = nullptr;
	PFN_vkAcquireNextImageKHR   baseAcquire = nullptr;
	PFN_vkQueuePresentKHR       basePresent = nullptr;
	PFN_vkGetDeviceProcAddr     realGdpa = nullptr;
	// Streamline's proxies for the same entry points (installed only while DLSS-G is attached).
	PFN_vkCreateSwapchainKHR    slCreate = nullptr;
	PFN_vkDestroySwapchainKHR   slDestroy = nullptr;
	PFN_vkGetSwapchainImagesKHR slGetImages = nullptr;
	PFN_vkAcquireNextImageKHR   slAcquire = nullptr;
	PFN_vkQueuePresentKHR       slPresent = nullptr;
	PFN_vkDeviceWaitIdle        slWaitIdle = nullptr;
	PFN_vkDeviceWaitIdle        baseWaitIdle = nullptr;
	bool slSwapchainLive = false;
#if NUKE_FFX_VK
	ffxContext ffxSc = nullptr;   // the frame-interpolation swap-chain context
	PFN_vkCreateSwapchainFFXAPI   rCreate = nullptr;
	PFN_vkDestroySwapchainFFXAPI  rDestroy = nullptr;
	PFN_getLastPresentCountFFXAPI rCount = nullptr;
	uint64_t countBase = 0; uint64_t countFrame = 0;
#endif
	uint32_t slPresentedAcc = 0, slPresentedN = 0;
} g_vk;

#if NUKE_FFX_VK
// AMD's replacement create / destroy carry the context as a 5th argument: the loader table needs plain ones.
VkResult VKAPI_CALL FfxCreateSwapchainTr(VkDevice d, const VkSwapchainCreateInfoKHR* ci, const VkAllocationCallbacks* a, VkSwapchainKHR* out)
{
	return g_vk.rCreate ? g_vk.rCreate(d, ci, a, out, g_vk.ffxSc) : (g_vk.baseCreate ? g_vk.baseCreate(d, ci, a, out) : VK_ERROR_INITIALIZATION_FAILED);
}
void VKAPI_CALL FfxDestroySwapchainTr(VkDevice d, VkSwapchainKHR s, const VkAllocationCallbacks* a)
{
	if (g_vk.rDestroy) g_vk.rDestroy(d, s, a, g_vk.ffxSc); else if (g_vk.baseDestroy) g_vk.baseDestroy(d, s, a);
}
#endif

// vkCreateInstance through Streamline's proxy when its Vulkan side is live (it adds the instance
// extensions its features need), else the loader's.
VkResult VKAPI_CALL HookCreateInstance(const VkInstanceCreateInfo* ci, const VkAllocationCallbacks* a, VkInstance* out)
{
	PFN_vkCreateInstance fn = vkCreateInstance;
	if (g_vk.slGipa) if (auto p = (PFN_vkCreateInstance)g_vk.slGipa(nullptr, "vkCreateInstance")) fn = p;
	// AMD's Vulkan backend resolves the debug-utils label functions and calls them unguarded: the
	// instance extension goes in whenever the loader offers it.
	VkInstanceCreateInfo ci2 = *ci;
	std::vector<const char*> iext(ci->ppEnabledExtensionNames, ci->ppEnabledExtensionNames + ci->enabledExtensionCount);
	uint32_t nProps = 0;
	vkEnumerateInstanceExtensionProperties(nullptr, &nProps, nullptr);
	std::vector<VkExtensionProperties> props(nProps);
	if (nProps) vkEnumerateInstanceExtensionProperties(nullptr, &nProps, props.data());
	auto offered = [&](const char* name) { for (const auto& pr : props) if (strcmp(pr.extensionName, name) == 0) return true; return false; };
	auto listed  = [&](const char* name) { for (const char* e : iext) if (e && strcmp(e, name) == 0) return true; return false; };
	bool changed = false;
	if (g_vk.wantQueues && offered(VK_EXT_DEBUG_UTILS_EXTENSION_NAME) && !listed(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) { iext.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME); changed = true; }
#if NUKE_XESS
	if (XessVk().ok())   // XeSS names the instance extensions its Vulkan path needs
	{
		uint32_t n = 0; const char* const* names = nullptr; uint32_t minVer = 0;
		if (XessVk().GetInstanceExtensions(&n, &names, &minVer) == XESS_RESULT_SUCCESS && names)
			for (uint32_t i = 0; i < n; ++i) if (names[i] && offered(names[i]) && !listed(names[i])) { iext.push_back(names[i]); changed = true; }
	}
#endif
	if (changed)
	{
		ci2.ppEnabledExtensionNames = iext.data();
		ci2.enabledExtensionCount = (uint32_t)iext.size();
		ci = &ci2;
	}
	cout << "[NukeDiligent]\tVulkan instance through " << (fn != vkCreateInstance ? "Streamline's proxy" : "the loader") << " (" << ci->enabledExtensionCount << " extensions)" << endl;
	const VkResult r = fn(ci, a, out);
	cout << "[NukeDiligent]\tVulkan instance: " << (int)r << endl;
	if (r == VK_SUCCESS && out) g_vk.instance = *out;
	// Streamline maps a physical device to its instance only inside ITS vkEnumeratePhysicalDevices;
	// Diligent enumerates through the loader, so the proxy must see the enumeration once here or
	// its device proxy runs on an empty dispatch table.
	if (r == VK_SUCCESS && g_vk.slGipa && g_vk.instance)
		if (auto en = (PFN_vkEnumeratePhysicalDevices)g_vk.slGipa(g_vk.instance, "vkEnumeratePhysicalDevices"))
		{
			uint32_t n = 0;
			if (en(g_vk.instance, &n, nullptr) == VK_SUCCESS && n)
			{
				std::vector<VkPhysicalDevice> devs(n);
				en(g_vk.instance, &n, devs.data());
			}
		}
	return r;
}

// vkCreateDevice: the extra queues AMD's frame-interpolation swap chain wants (present + image
// acquire in the graphics family, one in a compute family) go into the create info; Streamline's
// proxy (its own queues / extensions) creates when live.
VkResult VKAPI_CALL HookCreateDevice(VkPhysicalDevice pd, const VkDeviceCreateInfo* ci, const VkAllocationCallbacks* a, VkDevice* out)
{
	VkDeviceCreateInfo ci2 = *ci;
	std::vector<VkDeviceQueueCreateInfo> q(ci->pQueueCreateInfos, ci->pQueueCreateInfos + ci->queueCreateInfoCount);
	std::vector<std::vector<float>> prios;
	g_vk.physDev = pd;
	g_vk.haveQueues = false;
	if (g_vk.wantQueues && !q.empty())
	{
		uint32_t n = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, nullptr);
		std::vector<VkQueueFamilyProperties> props(n);
		if (n) vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, props.data());
		const uint32_t gfx = q[0].queueFamilyIndex;   // Diligent's first queue: the graphics one
		if (gfx < n)
		{
			const uint32_t used = q[0].queueCount, avail = props[gfx].queueCount > used ? props[gfx].queueCount - used : 0;
			if (avail >= 2)
			{
				g_vk.gfxFamily = gfx; g_vk.presentIndex = used; g_vk.acquireIndex = used + 1;
				q[0].queueCount += 2;
				// A compute-only family for the interpolation work (else one more graphics queue).
				// Not with Streamline's proxy live: it appends its own entry for the compute family and
				// a family listed twice is an invalid device create info.
				uint32_t cf = ~0u;
				for (uint32_t i = 0; i < n && cf == ~0u && !g_vk.slGipa; ++i)
				{
					if (i == gfx || !(props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) || (props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) continue;
					bool taken = false; for (const auto& qi : q) if (qi.queueFamilyIndex == i) taken = true;
					if (!taken && props[i].queueCount > 0) cf = i;
				}
				if (cf != ~0u)
				{
					VkDeviceQueueCreateInfo c{};
					c.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO; c.queueFamilyIndex = cf; c.queueCount = 1;
					q.push_back(c);
					g_vk.computeFamily = cf; g_vk.computeIndex = 0;
					g_vk.haveQueues = true;
				}
				else if (avail >= 3) { g_vk.computeFamily = gfx; g_vk.computeIndex = used + 2; q[0].queueCount += 1; g_vk.haveQueues = true; }
				else { q[0].queueCount -= 2; }
			}
		}
		prios.resize(q.size());
		for (size_t i = 0; i < q.size(); ++i) { prios[i].assign(q[i].queueCount, 1.0f); q[i].pQueuePriorities = prios[i].data(); }
		ci2.pQueueCreateInfos = q.data();
		ci2.queueCreateInfoCount = (uint32_t)q.size();
	}
	// AMD's Vulkan backend (SDK 1.1.4) takes its capabilities from what the physical device OFFERS
	// and never checks what the device enabled: it calls vkGetBufferMemoryRequirements2KHR
	// unguarded once dedicated allocation is offered, uses synchronization2 / full subgroups /
	// fp16 shaders the same way, and its swap chain runs on timeline semaphores. Those extensions
	// and features go in here when the device offers them and the engine did not ask itself.
	std::vector<const char*> dext(ci->ppEnabledExtensionNames, ci->ppEnabledExtensionNames + ci->enabledExtensionCount);
	static VkPhysicalDeviceTimelineSemaphoreFeatures     s_timeline{};
	static VkPhysicalDeviceSynchronization2Features      s_sync2{};
	static VkPhysicalDeviceShaderFloat16Int8Features     s_f16{};
	static VkPhysicalDeviceSubgroupSizeControlFeatures   s_subgroup{};
	if (g_vk.wantQueues)
	{
		uint32_t n = 0;
		vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, nullptr);
		std::vector<VkExtensionProperties> props(n);
		if (n) vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, props.data());
		auto offered = [&](const char* name) { for (const auto& pr : props) if (strcmp(pr.extensionName, name) == 0) return true; return false; };
		auto listed  = [&](const char* name) { for (const char* e : dext) if (e && strcmp(e, name) == 0) return true; return false; };
		const char* want[] = { VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME, VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
		                       VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME, VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME, VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME };
		for (const char* w : want) if (offered(w) && !listed(w)) dext.push_back(w);
		ci2.ppEnabledExtensionNames = dext.data();
		ci2.enabledExtensionCount = (uint32_t)dext.size();
		// The features the device supports (only those get switched on).
		VkPhysicalDeviceTimelineSemaphoreFeatures   qTimeline{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES };
		VkPhysicalDeviceSynchronization2Features    qSync2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES, &qTimeline };
		VkPhysicalDeviceShaderFloat16Int8Features   qF16{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES, &qSync2 };
		VkPhysicalDeviceSubgroupSizeControlFeatures qSub{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES, &qF16 };
		VkPhysicalDeviceFeatures2 f2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &qSub };
		vkGetPhysicalDeviceFeatures2(pd, &f2);
		// Set the bit in the engine's own struct when it chained one, else append ours.
		auto enable = [&](VkStructureType st, VkBaseOutStructure* mine, auto setter)
		{
			for (VkBaseOutStructure* p = (VkBaseOutStructure*)const_cast<void*>(ci2.pNext); p; p = p->pNext)
				if (p->sType == st) { setter(p); return; }
			mine->sType = st; setter(mine);
			mine->pNext = (VkBaseOutStructure*)const_cast<void*>(ci2.pNext);
			ci2.pNext = mine;
		};
		if (qTimeline.timelineSemaphore) enable(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES, (VkBaseOutStructure*)&s_timeline, [](VkBaseOutStructure* p) { ((VkPhysicalDeviceTimelineSemaphoreFeatures*)p)->timelineSemaphore = VK_TRUE; });
		if (qSync2.synchronization2)     enable(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES, (VkBaseOutStructure*)&s_sync2, [](VkBaseOutStructure* p) { ((VkPhysicalDeviceSynchronization2Features*)p)->synchronization2 = VK_TRUE; });
		if (qF16.shaderFloat16)          enable(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES, (VkBaseOutStructure*)&s_f16, [](VkBaseOutStructure* p) { ((VkPhysicalDeviceShaderFloat16Int8Features*)p)->shaderFloat16 = VK_TRUE; });
		if (qSub.subgroupSizeControl)    enable(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES, (VkBaseOutStructure*)&s_subgroup, [&](VkBaseOutStructure* p) { auto* f = (VkPhysicalDeviceSubgroupSizeControlFeatures*)p; f->subgroupSizeControl = VK_TRUE; if (qSub.computeFullSubgroups) f->computeFullSubgroups = VK_TRUE; });
		// The Vulkan 1.2 / 1.3 aggregate structs, if the engine used them instead, carry the same bits.
		for (VkBaseOutStructure* p = (VkBaseOutStructure*)const_cast<void*>(ci2.pNext); p; p = p->pNext)
		{
			if (p->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES) { auto* f = (VkPhysicalDeviceVulkan12Features*)p; if (qTimeline.timelineSemaphore) f->timelineSemaphore = VK_TRUE; if (qF16.shaderFloat16) f->shaderFloat16 = VK_TRUE; }
			if (p->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES) { auto* f = (VkPhysicalDeviceVulkan13Features*)p; if (qSync2.synchronization2) f->synchronization2 = VK_TRUE; if (qSub.subgroupSizeControl) f->subgroupSizeControl = VK_TRUE; if (qSub.computeFullSubgroups) f->computeFullSubgroups = VK_TRUE; }
		}
	}
#if NUKE_XESS
	if (XessVk().ok() && g_vk.instance)   // XeSS's device extensions and features (its chain merged into the engine's)
	{
		if (!g_vk.wantQueues) { dext.assign(ci->ppEnabledExtensionNames, ci->ppEnabledExtensionNames + ci->enabledExtensionCount); }
		uint32_t n = 0; const char* const* names = nullptr;
		if (XessVk().GetDeviceExtensions(g_vk.instance, pd, &n, &names) == XESS_RESULT_SUCCESS && names)
		{
			uint32_t np = 0;
			vkEnumerateDeviceExtensionProperties(pd, nullptr, &np, nullptr);
			std::vector<VkExtensionProperties> dprops(np);
			if (np) vkEnumerateDeviceExtensionProperties(pd, nullptr, &np, dprops.data());
			for (uint32_t i = 0; i < n; ++i)
			{
				if (!names[i]) continue;
				bool off = false, lst = false;
				for (const auto& pr : dprops) if (strcmp(pr.extensionName, names[i]) == 0) off = true;
				for (const char* e : dext) if (e && strcmp(e, names[i]) == 0) lst = true;
				if (off && !lst) dext.push_back(names[i]);
			}
		}
		ci2.ppEnabledExtensionNames = dext.data();
		ci2.enabledExtensionCount = (uint32_t)dext.size();
		void* feats = nullptr;
		if (XessVk().GetDeviceFeatures(g_vk.instance, pd, &feats) == XESS_RESULT_SUCCESS && feats)
		{
			for (const VkBaseInStructure* x = (const VkBaseInStructure*)feats; x; x = x->pNext)
			{
				if (x->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES || x->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES || x->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES)
				{
					SplitAggregateFeatures(ci2, x);   // the engine chains the individual structs: no aggregates beside them
					continue;
				}
				if (x->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2)   // the core features ride pEnabledFeatures
				{
					if (ci2.pEnabledFeatures)
					{
						VkBool32* dst = (VkBool32*)const_cast<VkPhysicalDeviceFeatures*>(ci2.pEnabledFeatures);
						const VkBool32* src = (const VkBool32*)&((const VkPhysicalDeviceFeatures2*)x)->features;
						for (size_t i = 0; i < sizeof(VkPhysicalDeviceFeatures) / sizeof(VkBool32); ++i) dst[i] |= src[i];
					}
					continue;
				}
				const size_t sz = VkFeatureStructSize(x->sType);
				if (!sz) { cout << "[NukeDiligent]\tXeSS (Vulkan): feature struct " << (int)x->sType << " unknown here, not enabled" << endl; continue; }
				MergeFeatureStruct(ci2, x->sType, (const VkBool32*)((const char*)x + sizeof(VkBaseInStructure)), (sz - sizeof(VkBaseOutStructure)) / sizeof(VkBool32));
			}
		}
	}
#endif
	PFN_vkCreateDevice fn = vkCreateDevice;
	if (g_vk.slGipa && g_vk.instance) if (auto p = (PFN_vkCreateDevice)g_vk.slGipa(g_vk.instance, "vkCreateDevice")) fn = p;
	cout << "[NukeDiligent]\tVulkan device through " << (fn != vkCreateDevice ? "Streamline's proxy" : "the loader") << " (" << ci2.queueCreateInfoCount << " queue families, " << ci2.enabledExtensionCount << " extensions)" << endl;
	const VkResult r = fn(pd, &ci2, a, out);
	cout << "[NukeDiligent]\tVulkan device: " << (int)r << endl;
	if (r == VK_SUCCESS && out) g_vk.device = *out;
	else g_vk.haveQueues = false;
	return r;
}

VkFormat VkFormatOf(TEXTURE_FORMAT f)
{
	switch (f)
	{
	case TEX_FORMAT_RGBA8_UNORM:      return VK_FORMAT_R8G8B8A8_UNORM;
	case TEX_FORMAT_RGBA8_UNORM_SRGB: return VK_FORMAT_R8G8B8A8_SRGB;
	case TEX_FORMAT_BGRA8_UNORM:      return VK_FORMAT_B8G8R8A8_UNORM;
	case TEX_FORMAT_BGRA8_UNORM_SRGB: return VK_FORMAT_B8G8R8A8_SRGB;
	case TEX_FORMAT_RGB10A2_UNORM:    return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
	case TEX_FORMAT_RGBA16_FLOAT:     return VK_FORMAT_R16G16B16A16_SFLOAT;
	case TEX_FORMAT_RGBA32_FLOAT:     return VK_FORMAT_R32G32B32A32_SFLOAT;
	case TEX_FORMAT_RG16_FLOAT:       return VK_FORMAT_R16G16_SFLOAT;
	case TEX_FORMAT_RG32_FLOAT:       return VK_FORMAT_R32G32_SFLOAT;
	case TEX_FORMAT_R16_FLOAT:        return VK_FORMAT_R16_SFLOAT;
	case TEX_FORMAT_R32_FLOAT:        return VK_FORMAT_R32_SFLOAT;
	case TEX_FORMAT_R8_UNORM:         return VK_FORMAT_R8_UNORM;
	case TEX_FORMAT_RG8_UNORM:        return VK_FORMAT_R8G8_UNORM;
	case TEX_FORMAT_R11G11B10_FLOAT:  return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
	case TEX_FORMAT_D32_FLOAT:        return VK_FORMAT_D32_SFLOAT;
	case TEX_FORMAT_D24_UNORM_S8_UINT: return VK_FORMAT_D24_UNORM_S8_UINT;
	case TEX_FORMAT_D16_UNORM:        return VK_FORMAT_D16_UNORM;
	case TEX_FORMAT_D32_FLOAT_S8X24_UINT: return VK_FORMAT_D32_SFLOAT_S8_UINT;
	default:                          return VK_FORMAT_UNDEFINED;
	}
}
bool IsDepth(TEXTURE_FORMAT f) { return f == TEX_FORMAT_D32_FLOAT || f == TEX_FORMAT_D24_UNORM_S8_UINT || f == TEX_FORMAT_D16_UNORM || f == TEX_FORMAT_D32_FLOAT_S8X24_UINT; }
VkImage ImageOf(ITexture* t) { RefCntAutoPtr<ITextureVk> tv(t, IID_TextureVk); return tv ? tv->GetVkImage() : VK_NULL_HANDLE; }
VkImageView ViewOf(ITextureView* v) { RefCntAutoPtr<ITextureViewVk> vv(v, IID_TextureViewVk); return vv ? vv->GetVulkanImageView() : VK_NULL_HANDLE; }
VkImageLayout LayoutOf(ITexture* t) { RefCntAutoPtr<ITextureVk> tv(t, IID_TextureVk); return tv ? tv->GetLayout() : VK_IMAGE_LAYOUT_UNDEFINED; }

#if NUKE_FFX_VK
// A Diligent texture as an FFX Vulkan resource (the description comes from a create info built
// from the texture's own).
FfxApiResource FfxResVk(ITexture* t, uint32_t state, bool uav)
{
	if (!t) return FfxApiResource{};
	const TextureDesc& d = t->GetDesc();
	VkImageCreateInfo ci{};
	ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	ci.imageType = VK_IMAGE_TYPE_2D;
	ci.format = VkFormatOf(d.Format);
	ci.extent = { d.Width, d.Height, 1u };
	ci.mipLevels = d.MipLevels ? d.MipLevels : 1u;
	ci.arrayLayers = 1;
	ci.samples = VK_SAMPLE_COUNT_1_BIT;
	ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | (uav ? VK_IMAGE_USAGE_STORAGE_BIT : 0u) | (IsDepth(d.Format) ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
	const VkImage img = ImageOf(t);
	return ffxApiGetResourceVK((void*)img, ffxApiGetImageResourceDescriptionVK(img, ci, uav ? FFX_API_RESOURCE_USAGE_UAV : 0u), state);
}
#endif

}  // namespace

// ---- the hooks around device creation ------------------------------------------------------------

void NukeDiligent::Impl::FrameGenVkBeforeDevice(bool nvAdapter)
{
	if (fgVk.hooked) return;
	fgVk.hooked = true;
#if NUKE_FFX_VK
	g_vk.wantQueues = FfxVk().ok();   // AMD's swap chain needs its own queues: reserved only when its runtime is here
#endif
#if NUKE_XESS
	if (XessVk().ok()) cout << "[NukeDiligent]\tXeSS (Vulkan): libxess.dll loaded, its instance / device requirements go in at creation" << endl;
#endif
#if NUKE_STREAMLINE
	const char* slOff = std::getenv("NUKE_SL_DISABLE");   // NUKE_SL_DISABLE=1: no Streamline on this Vulkan run (diagnostics)
	if (nvAdapter && !(slOff && *slOff == '1') && Sl().ok() && SlInitOnce(*this, true, nullptr))
	{
		g_vk.slGipa = (PFN_vkGetInstanceProcAddr)GetProcAddress(Sl().dll, "vkGetInstanceProcAddr");
		g_vk.slGdpa = (PFN_vkGetDeviceProcAddr)GetProcAddress(Sl().dll, "vkGetDeviceProcAddr");
		if (g_vk.slGipa && g_vk.slGdpa)
		{
			const char* keep = std::getenv("NUKE_SL_KEEP_LOADED");   // diagnostics: leave DLSS-G loaded for the first swap chain
			const sl::Result r = (keep && *keep == '1') ? sl::Result::eOk : Sl().SetFeatureLoaded(sl::kFeatureDLSS_G, false);   // the first swap chain goes up without the generator
			fgVk.slProxied = true;
			cout << "[NukeDiligent]\tStreamline: Vulkan create proxies installed (DLSS-G unloaded for the first swap chain: " << sl::getResultAsStr(r) << ")" << endl;
		}
		else cout << "[NukeDiligent]\tStreamline: no Vulkan entry points in sl.interposer.dll - DLSS-G stays off on Vulkan" << endl;
	}
#else
	(void)nvAdapter;
#endif
	g_NukeVkCreateInstance = HookCreateInstance;
	g_NukeVkCreateDevice = HookCreateDevice;
}

void NukeDiligent::Impl::FrameGenVkAfterDevice()
{
	if (!fgVk.hooked || !device) return;
	RefCntAutoPtr<IRenderDeviceVk> dvk(device, IID_RenderDeviceVk);
	if (!dvk) return;
	g_vk.device = dvk->GetVkDevice(); g_vk.physDev = dvk->GetVkPhysicalDevice(); g_vk.instance = dvk->GetVkInstance();
	g_vk.realGdpa = vkGetDeviceProcAddr;
	g_vk.baseCreate = vkCreateSwapchainKHR; g_vk.baseDestroy = vkDestroySwapchainKHR; g_vk.baseGetImages = vkGetSwapchainImagesKHR;
	g_vk.baseAcquire = vkAcquireNextImageKHR; g_vk.basePresent = vkQueuePresentKHR; g_vk.baseWaitIdle = vkDeviceWaitIdle;
	if (fgVk.slProxied)
	{
		// The surface entry points go to Streamline for good (the generator resolves the window
		// through the interposer's surface map); the swap-chain / present ones only while DLSS-G
		// is attached (SwapchainProxies) - the loader's own otherwise, so AMD's swap chain and
		// the plain path never run through the interposer.
		auto dev = [&](const char* n) { return g_vk.slGdpa(g_vk.device, n); };
		auto ins = [&](const char* n) { return g_vk.slGipa(g_vk.instance, n); };
		g_vk.slCreate = (PFN_vkCreateSwapchainKHR)dev("vkCreateSwapchainKHR");
		g_vk.slDestroy = (PFN_vkDestroySwapchainKHR)dev("vkDestroySwapchainKHR");
		g_vk.slGetImages = (PFN_vkGetSwapchainImagesKHR)dev("vkGetSwapchainImagesKHR");
		g_vk.slAcquire = (PFN_vkAcquireNextImageKHR)dev("vkAcquireNextImageKHR");
		g_vk.slPresent = (PFN_vkQueuePresentKHR)dev("vkQueuePresentKHR");
		g_vk.slWaitIdle = (PFN_vkDeviceWaitIdle)dev("vkDeviceWaitIdle");
		if (auto p = ins("vkCreateWin32SurfaceKHR")) vkCreateWin32SurfaceKHR = (PFN_vkCreateWin32SurfaceKHR)p;
		if (auto p = ins("vkDestroySurfaceKHR"))     vkDestroySurfaceKHR = (PFN_vkDestroySurfaceKHR)p;
		if (!(g_vk.slCreate && g_vk.slDestroy && g_vk.slGetImages && g_vk.slAcquire && g_vk.slPresent && g_vk.slWaitIdle)) { fgVk.slProxied = false; cout << "[NukeDiligent]\tStreamline: Vulkan swap-chain proxies missing - DLSS-G stays off" << endl; }
		else cout << "[NukeDiligent]\tStreamline: Vulkan proxies ready" << (SlFeatures(*this) ? " (DLSS-G feature functions ready)" : " (DLSS-G feature functions missing)") << endl;
	}
	if (g_vk.haveQueues)
	{
		vkGetDeviceQueue(g_vk.device, g_vk.gfxFamily, g_vk.presentIndex, &g_vk.presentQ);
		vkGetDeviceQueue(g_vk.device, g_vk.gfxFamily, g_vk.acquireIndex, &g_vk.acquireQ);
		vkGetDeviceQueue(g_vk.device, g_vk.computeFamily, g_vk.computeIndex, &g_vk.computeQ);
		fgVk.queues = g_vk.presentQ && g_vk.acquireQ && g_vk.computeQ;
	}
#if NUKE_FFX_VK
	if (FfxVk().ok()) cout << "[NukeDiligent]\tFSR 3.1 (Vulkan): amd_fidelityfx_vk.dll loaded" << (fgVk.queues ? ", frame-generation queues reserved" : ", no spare queues for frame generation") << endl;
#endif
}

// Streamline's swap-chain / present entry points in, or the loader's back.
static void SwapchainProxies(bool on)
{
	if (on)
	{
		vkCreateSwapchainKHR = g_vk.slCreate; vkDestroySwapchainKHR = g_vk.slDestroy; vkGetSwapchainImagesKHR = g_vk.slGetImages;
		vkAcquireNextImageKHR = g_vk.slAcquire; vkQueuePresentKHR = g_vk.slPresent; vkDeviceWaitIdle = g_vk.slWaitIdle;
	}
	else
	{
		vkCreateSwapchainKHR = g_vk.baseCreate; vkDestroySwapchainKHR = g_vk.baseDestroy; vkGetSwapchainImagesKHR = g_vk.baseGetImages;
		vkAcquireNextImageKHR = g_vk.baseAcquire; vkQueuePresentKHR = g_vk.basePresent; vkDeviceWaitIdle = g_vk.baseWaitIdle;
	}
	g_vk.slSwapchainLive = on;
}

// ---- the generator ---------------------------------------------------------------------------------

bool NukeDiligent::Impl::FrameGenVkAttach(int kind, int frames)
{
	RefCntAutoPtr<ISwapChainVk> sc(swapChain, IID_SwapChainVk);
	if (!sc || !g_vk.device) return false;
	std::string detail;
	auto flushIdle = [&]() { context->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE); context->Flush(); device->IdleGPU(); };
#if NUKE_FFX_VK
	if (kind == FG_FSR)
	{
		if (!FfxVk().ok()) detail = "amd_fidelityfx_vk.dll not next to NukeRenderDiligent.dll";
		else if (!fgVk.queues) detail = "no spare Vulkan queues for the frame-interpolation swap chain";
		else
		{
			VkSwapchainCreateInfoKHR ci{};
			sc->GetVkSwapChainCreateInfo(&ci);
			const int W = (int)ci.imageExtent.width, H = (int)ci.imageExtent.height;
			ffxCreateBackendVKDesc be{};
			be.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
			be.vkDevice = g_vk.device; be.vkPhysicalDevice = g_vk.physDev; be.vkDeviceProcAddr = g_vk.realGdpa;
			ffxCreateContextDescFrameGenerationHudless hl{};
			hl.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION_HUDLESS;
			hl.hudlessBackBufferFormat = ffxApiGetSurfaceFormatVK(ci.imageFormat);
			hl.header.pNext = &be.header;
			ffxCreateContextDescFrameGeneration cd{};
			cd.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
			cd.header.pNext = &hl.header;
			cd.flags = hdr10Active ? FFX_FRAMEGENERATION_ENABLE_HIGH_DYNAMIC_RANGE : 0u;
			cd.displaySize = { (uint32_t)W, (uint32_t)H };
			cd.maxRenderSize = { (uint32_t)W, (uint32_t)H };
			cd.backBufferFormat = ffxApiGetSurfaceFormatVK(ci.imageFormat);
			ffxReturnCode_t rc = FfxVk().CreateContext(&fg.ffx, &cd.header, nullptr);
			if (rc != FFX_API_RETURN_OK || !fg.ffx) { detail = "FG context creation failed: " + std::to_string(rc); fg.ffx = nullptr; }
			else
			{
				// Diligent's game queue (family from its command queue).
				{
					ICommandQueue* q = context->LockCommandQueue();
					RefCntAutoPtr<ICommandQueueVk> qv(q, IID_CommandQueueVk);
					if (qv) { g_vk.gameQ = qv->GetVkQueue(); g_vk.gameFamily = qv->GetQueueFamilyIndex(); }
					context->UnlockCommandQueue();
				}
				flushIdle();
				cout << "[NukeDiligent]\tFSR FG (Vulkan): FG context ok, detaching the swap chain" << endl;
				VkSwapchainKHR cur = sc->DetachVkSwapChain();   // AMD retires it (its own goes up over the same surface)
				cout << "[NukeDiligent]\tFSR FG (Vulkan): detached, creating the frame-interpolation swap chain (" << ci.imageExtent.width << "x" << ci.imageExtent.height << ", " << ci.minImageCount << " images, format " << ci.imageFormat << ")" << endl;
				ffxCreateContextDescFrameGenerationSwapChainVK sd{};
				sd.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FGSWAPCHAIN_VK;
				sd.physicalDevice = g_vk.physDev; sd.device = g_vk.device;
				sd.swapchain = &cur; sd.allocator = nullptr;
				sd.createInfo = ci; sd.createInfo.oldSwapchain = cur;
				sd.gameQueue = { g_vk.gameQ, g_vk.gameFamily, nullptr };
				sd.asyncComputeQueue = { g_vk.computeQ, g_vk.computeFamily, nullptr };
				sd.presentQueue = { g_vk.presentQ, g_vk.gfxFamily, nullptr };
				sd.imageAcquireQueue = { g_vk.acquireQ, g_vk.gfxFamily, nullptr };
				rc = FfxVk().CreateContext(&g_vk.ffxSc, &sd.header, nullptr);
				cout << "[NukeDiligent]\tFSR FG (Vulkan): swap-chain context rc=" << rc << endl;
				if (rc != FFX_API_RETURN_OK || !g_vk.ffxSc || cur == VK_NULL_HANDLE)
				{
					detail = "frame-interpolation swap chain failed: " + std::to_string(rc);
					g_vk.ffxSc = nullptr;
					FfxVk().DestroyContext(&fg.ffx, nullptr); fg.ffx = nullptr;
					sc->AttachVkSwapChain(VK_NULL_HANDLE);   // the old one was retired by the attempt: a fresh own chain
				}
				else
				{
					ffxQueryDescSwapchainReplacementFunctionsVK rf{};
					rf.header.type = FFX_API_QUERY_DESC_TYPE_FGSWAPCHAIN_FUNCTIONS_VK;
					if (FfxVk().Query(&g_vk.ffxSc, &rf.header) == FFX_API_RETURN_OK && rf.pOutGetSwapchainImagesKHR && rf.pOutAcquireNextImageKHR && rf.pOutQueuePresentKHR)
					{
						g_vk.rCreate = rf.pOutCreateSwapchainFFXAPI; g_vk.rDestroy = rf.pOutDestroySwapchainFFXAPI; g_vk.rCount = rf.pOutGetLastPresentCountFFXAPI;
						vkCreateSwapchainKHR = FfxCreateSwapchainTr;
						vkDestroySwapchainKHR = FfxDestroySwapchainTr;
						vkGetSwapchainImagesKHR = rf.pOutGetSwapchainImagesKHR;
						vkAcquireNextImageKHR = rf.pOutAcquireNextImageKHR;
						vkQueuePresentKHR = rf.pOutQueuePresentKHR;
						cout << "[NukeDiligent]\tFSR FG (Vulkan): replacement functions installed, attaching" << endl;
						sc->AttachVkSwapChain(cur);
						cout << "[NukeDiligent]\tFSR FG (Vulkan): attached" << endl;
						g_vk.countBase = 0; g_vk.countFrame = 0;
						fg.framesLive = 1;
						return true;
					}
					detail = "no replacement functions from the frame-interpolation swap chain";
					FfxVk().DestroyContext(&g_vk.ffxSc, nullptr); g_vk.ffxSc = nullptr;   // destroys its chain
					FfxVk().DestroyContext(&fg.ffx, nullptr); fg.ffx = nullptr;
					sc->AttachVkSwapChain(VK_NULL_HANDLE);
				}
			}
		}
	}
#endif
#if NUKE_STREAMLINE
	if (kind == FG_DLSS)
	{
		if (!fgVk.slProxied || !fg.slReady) detail = "Streamline's Vulkan proxies are not live (NVIDIA adapter + sl.interposer.dll)";
		else
		{
			sl::AdapterInfo ai{};
			ai.vkPhysicalDevice = g_vk.physDev;
			const sl::Result sup = Sl().IsFeatureSupported(sl::kFeatureDLSS_G, ai);
			if (sup != sl::Result::eOk) detail = std::string("DLSS-G not supported here: ") + sl::getResultAsStr(sup);
			else
			{
				// The generator attaches to swap chains created through Streamline while it is loaded:
				// the loader's chain goes, the proxies come in, ours is recreated through them.
				flushIdle();
				VkSwapchainKHR old = sc->DetachVkSwapChain();
				if (old != VK_NULL_HANDLE) vkDestroySwapchainKHR(g_vk.device, old, nullptr);   // the loader's (its creator)
				SwapchainProxies(true);
				Sl().SetFeatureLoaded(sl::kFeatureDLSS_G, true);
				sc->AttachVkSwapChain(VK_NULL_HANDLE);
				sl::DLSSGState st{};
				uint32_t cap = 1;
				if (Sl().DLSSGGetState(kSlViewport, st, nullptr) == sl::Result::eOk && st.numFramesToGenerateMax > 0) cap = st.numFramesToGenerateMax;
				fg.framesLive = (int)std::min<uint32_t>((uint32_t)std::max(frames, 1), cap);
				g_vk.slPresentedAcc = 0; g_vk.slPresentedN = 0;
				return true;
			}
		}
	}
#endif
	if (detail.empty()) detail = kind == FG_XESS ? "XeSS-FG has no Vulkan runtime" : "not built into this renderer";
	cout << "[NukeDiligent]\tframe generation: " << FrameGenName(kind) << " unavailable on Vulkan - " << detail << endl;
	(void)frames;
	return false;
}

void NukeDiligent::Impl::FrameGenVkDetach()
{
	RefCntAutoPtr<ISwapChainVk> sc(swapChain, IID_SwapChainVk);
	if (!sc) return;
#if NUKE_FFX_VK
	if (fg.kind == FG_FSR)
	{
		if (fg.ffx)   // generation off for the last presents before the chain goes
		{
			ffxConfigureDescFrameGeneration cfg{};
			cfg.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
			cfg.swapChain = (void*)sc->GetVkSwapChain();
			cfg.frameGenerationEnabled = false;
			cfg.frameGenerationCallback = FfxVkFgDispatchCb; cfg.frameGenerationCallbackUserContext = &fg.ffx;
			cfg.frameID = fg.frameIndex;
			FfxVk().Configure(&fg.ffx, &cfg.header);
		}
		VkSwapchainKHR proxy = sc->DetachVkSwapChain();   // owned by the context: its destroy destroys it
		(void)proxy;
		if (g_vk.ffxSc) { FfxVk().DestroyContext(&g_vk.ffxSc, nullptr); g_vk.ffxSc = nullptr; }
		g_vk.rCreate = nullptr; g_vk.rDestroy = nullptr; g_vk.rCount = nullptr;
		vkCreateSwapchainKHR = g_vk.baseCreate; vkDestroySwapchainKHR = g_vk.baseDestroy; vkGetSwapchainImagesKHR = g_vk.baseGetImages;
		vkAcquireNextImageKHR = g_vk.baseAcquire; vkQueuePresentKHR = g_vk.basePresent;
		if (fg.ffx) { FfxVk().DestroyContext(&fg.ffx, nullptr); fg.ffx = nullptr; }
		sc->AttachVkSwapChain(VK_NULL_HANDLE);
	}
#endif
#if NUKE_STREAMLINE
	if (fg.kind == FG_DLSS)
	{
		// The generator's chain goes while the plugin is still loaded (its destroy hook must see it,
		// or it keeps counting the chain and refuses the next one: "only one swap-chain can be used").
		VkSwapchainKHR old = sc->DetachVkSwapChain();
		if (old != VK_NULL_HANDLE) vkDestroySwapchainKHR(g_vk.device, old, nullptr);   // through the proxy: its creator
		Sl().SetFeatureLoaded(sl::kFeatureDLSS_G, false);
		SwapchainProxies(false);
		sc->AttachVkSwapChain(VK_NULL_HANDLE);
	}
#endif
}

void NukeDiligent::Impl::FrameGenVkBeforePresent(bool ready, float dt, int W, int H)
{
	RefCntAutoPtr<IDeviceContextVk> ctxVk(context, IID_DeviceContextVk);
	RefCntAutoPtr<ISwapChainVk> sc(swapChain, IID_SwapChainVk);
	if (!ctxVk || !sc) return;
	const bool wantLive = ready;
	if (ready)
	{
		context->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE);
		ctxVk->TransitionImageLayout(fg.depth, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		ctxVk->TransitionImageLayout(fg.vel, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		ctxVk->TransitionImageLayout(fg.hudless, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	}
	const float right[3] = { fg.view.m[0][0], fg.view.m[1][0], fg.view.m[2][0] };
	const float up[3]    = { fg.view.m[0][1], fg.view.m[1][1], fg.view.m[2][1] };
	const float fwd[3]   = { fg.view.m[0][2], fg.view.m[1][2], fg.view.m[2][2] };
	(void)right; (void)up; (void)fwd;
#if NUKE_FFX_VK
	if (fg.kind == FG_FSR && fg.ffx)
	{
		if (ready)
		{
			ffxDispatchDescFrameGenerationPrepareCameraInfo cam{};
			cam.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE_CAMERAINFO;
			memcpy(cam.cameraPosition, fg.pos, sizeof(cam.cameraPosition));
			memcpy(cam.cameraUp, up, sizeof(cam.cameraUp));
			memcpy(cam.cameraRight, right, sizeof(cam.cameraRight));
			memcpy(cam.cameraForward, fwd, sizeof(cam.cameraForward));
			ffxDispatchDescFrameGenerationPrepare pd{};
			pd.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE;
			pd.header.pNext = &cam.header;
			pd.frameID = fg.frameIndex;
			pd.flags = 0;
			pd.commandList = ctxVk->GetVkCommandBuffer();
			pd.renderSize = { (uint32_t)fg.rw, (uint32_t)fg.rh };
			pd.jitterOffset = { fg.jx, fg.jy };
			pd.motionVectorScale = { -(float)fg.rw, -(float)fg.rh };   // UV (current - previous) -> pixels (previous - current)
			pd.frameTimeDelta = dt * 1000.0f;
			pd.unused_reset = fg.reset;
			pd.cameraNear = fg.nearZ; pd.cameraFar = fg.farZ;
			pd.cameraFovAngleVertical = fg.fov;
			pd.viewSpaceToMetersFactor = 1.0f;
			pd.depth = FfxResVk(fg.depth, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ, false);
			pd.motionVectors = FfxResVk(fg.vel, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ, false);
			const ffxReturnCode_t rc = FfxVk().Dispatch(&fg.ffx, &pd.header);
			if (rc != FFX_API_RETURN_OK && fg.lastLog != frameId) { fg.lastLog = frameId; cout << "[NukeDiligent]\tFSR FG (Vulkan) prepare failed: " << rc << endl; }
			context->Flush();   // the vendor recorded its own pipelines / sets on our command buffer: submit, start a fresh one
		}
		ffxConfigureDescFrameGeneration cfg{};
		cfg.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
		cfg.swapChain = (void*)sc->GetVkSwapChain();
		cfg.presentCallback = nullptr;
		cfg.frameGenerationCallback = FfxVkFgDispatchCb;
		cfg.frameGenerationCallbackUserContext = &fg.ffx;
		cfg.frameGenerationEnabled = wantLive;
		cfg.allowAsyncWorkloads = false;
		if (ready) cfg.HUDLessColor = FfxResVk(fg.hudless, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ, false);
		cfg.flags = 0;
		cfg.onlyPresentGenerated = false;
		cfg.generationRect = { 0, 0, W, H };
		cfg.frameID = fg.frameIndex;
		const ffxReturnCode_t rc = FfxVk().Configure(&fg.ffx, &cfg.header);
		if (rc != FFX_API_RETURN_OK && fg.lastLog != frameId) { fg.lastLog = frameId; cout << "[NukeDiligent]\tFSR FG (Vulkan) configure failed: " << rc << endl; }
		fg.live = wantLive;
	}
#endif
#if NUKE_STREAMLINE
	if (fg.kind == FG_DLSS && fg.slReady)
	{
		if (wantLive != fg.live && Sl().DLSSGSetOptions && frameId != fg.attachFrame)
		{
			sl::DLSSGOptions o{};
			o.mode = wantLive ? sl::DLSSGMode::eOn : sl::DLSSGMode::eOff;
			o.numFramesToGenerate = (uint32_t)std::max(fg.framesLive, 1);
			o.flags = sl::DLSSGFlags::eRetainResourcesWhenOff;
			Sl().DLSSGSetOptions(kSlViewport, o);
			fg.live = wantLive;
		}
		if (ready && fg.slToken)
		{
			const sl::FrameToken& tok = *(const sl::FrameToken*)fg.slToken;
			auto res = [&](ITexture* t) -> sl::Resource
			{
				const TextureDesc& d = t->GetDesc();
				sl::Resource r(sl::ResourceType::eTex2d, (void*)ImageOf(t), nullptr, (void*)ViewOf(t->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE)), (uint32_t)LayoutOf(t));
				r.width = d.Width; r.height = d.Height; r.nativeFormat = (uint32_t)VkFormatOf(d.Format); r.mipLevels = 1; r.arrayLayers = 1;
				r.usage = VK_IMAGE_USAGE_SAMPLED_BIT | (IsDepth(d.Format) ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
				return r;
			};
			sl::Resource depth = res(fg.depth), mvec = res(fg.vel), hudless = res(fg.hudless);
			sl::Extent rExt{ 0u, 0u, (uint32_t)fg.rw, (uint32_t)fg.rh };
			sl::Extent oExt{ 0u, 0u, (uint32_t)W, (uint32_t)H };
			sl::ResourceTag tags[4] = {
				sl::ResourceTag(&depth,   sl::kBufferTypeDepth,         sl::ResourceLifecycle::eValidUntilPresent, &rExt),
				sl::ResourceTag(&mvec,    sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eValidUntilPresent, &rExt),
				sl::ResourceTag(&hudless, sl::kBufferTypeHUDLessColor,  sl::ResourceLifecycle::eValidUntilPresent, &oExt),
				sl::ResourceTag(nullptr,  sl::kBufferTypeBackbuffer,    sl::ResourceLifecycle::eValidUntilPresent, &oExt) };
			Sl().SetTagForFrame(tok, kSlViewport, tags, 4, (sl::CommandBuffer*)ctxVk->GetVkCommandBuffer());
			FrameGenSlConstants(W, H);
		}
	}
#endif
}

double NukeDiligent::Impl::FrameGenVkPresentRatio()
{
#if NUKE_FFX_VK
	if (fg.kind == FG_FSR && g_vk.rCount)
	{
		RefCntAutoPtr<ISwapChainVk> sc(swapChain, IID_SwapChainVk);
		if (!sc) return 0.0;
		const uint64_t n = g_vk.rCount(sc->GetVkSwapChain());
		double r = 0.0;
		if (g_vk.countBase && n > g_vk.countBase && fg.frameIndex > g_vk.countFrame) r = (double)(n - g_vk.countBase) / (double)(fg.frameIndex - g_vk.countFrame);
		g_vk.countBase = n; g_vk.countFrame = fg.frameIndex;
		return r;
	}
#endif
#if NUKE_STREAMLINE
	if (fg.kind == FG_DLSS && fg.slReady && Sl().DLSSGGetState)
	{
		sl::DLSSGState st{};
		if (Sl().DLSSGGetState(kSlViewport, st, nullptr) == sl::Result::eOk && st.numFramesActuallyPresented > 0) return (double)st.numFramesActuallyPresented;
	}
#endif
	return 0.0;
}

// ---- FSR 3.1.4 upscaling on Vulkan (the upscale stage's KIND_FFX variant there) -------------------

namespace nukediligent {

bool FfxVkAvailable()
{
#if NUKE_FFX_VK
	return FfxVk().ok();
#else
	return false;
#endif
}

bool FfxVkQueryRenderSize(int quality, int ow, int oh, int& rw, int& rh)
{
#if NUKE_FFX_VK
	if (!FfxVk().ok()) return false;
	uint32_t w = 0, h = 0;
	ffxQueryDescUpscaleGetRenderResolutionFromQualityMode q{};
	q.header.type = FFX_API_QUERY_DESC_TYPE_UPSCALE_GETRENDERRESOLUTIONFROMQUALITYMODE;
	q.displayWidth = (uint32_t)ow; q.displayHeight = (uint32_t)oh; q.qualityMode = FfxQualityOf(quality);
	q.pOutRenderWidth = &w; q.pOutRenderHeight = &h;
	if (FfxVk().Query(nullptr, &q.header) == FFX_API_RETURN_OK && w > 0 && h > 0) { rw = (int)w; rh = (int)h; return true; }
#else
	(void)quality; (void)ow; (void)oh; (void)rw; (void)rh;
#endif
	return false;
}

bool FfxVkUpscaleCreate(NukeDiligent::Impl& d, NukeDiligent::Impl::UpscaleState& us, int iw, int ih, int ow, int oh)
{
#if NUKE_FFX_VK
	if (!FfxVk().ok() || !g_vk.device) return false;
	ffxCreateBackendVKDesc be{};
	be.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
	be.vkDevice = g_vk.device; be.vkPhysicalDevice = g_vk.physDev; be.vkDeviceProcAddr = g_vk.realGdpa ? g_vk.realGdpa : vkGetDeviceProcAddr;
	ffxCreateContextDescUpscale cd{};
	cd.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
	cd.header.pNext = &be.header;
	cd.flags = FFX_UPSCALE_ENABLE_AUTO_EXPOSURE | (d.hdr ? FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE : FFX_UPSCALE_ENABLE_NON_LINEAR_COLORSPACE);
	cd.maxRenderSize = { (uint32_t)iw, (uint32_t)ih };
	cd.maxUpscaleSize = { (uint32_t)ow, (uint32_t)oh };
	cd.fpMessage = FfxVkMessage;
	cout << "[NukeDiligent]\tFSR (Vulkan): creating the upscaler context " << iw << "x" << ih << " -> " << ow << "x" << oh << endl;
	const ffxReturnCode_t crc = FfxVk().CreateContext(&us.ffx, &cd.header, nullptr);
	cout << "[NukeDiligent]\tFSR (Vulkan): upscaler context rc=" << crc << endl;
	if (crc != FFX_API_RETURN_OK || !us.ffx) { us.ffx = nullptr; return false; }
	int32_t phase = 0;
	ffxQueryDescUpscaleGetJitterPhaseCount q{};
	q.header.type = FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT;
	q.renderWidth = (uint32_t)iw; q.displayWidth = (uint32_t)ow; q.pOutPhaseCount = &phase;
	if (FfxVk().Query(&us.ffx, &q.header) == FFX_API_RETURN_OK && phase > 0) us.phaseCount = phase;
	return true;
#else
	(void)d; (void)us; (void)iw; (void)ih; (void)ow; (void)oh;
	return false;
#endif
}

bool FfxVkJitter(NukeDiligent::Impl::UpscaleState& us, int index, float& jx, float& jy)
{
#if NUKE_FFX_VK
	if (!us.ffx) return false;
	float x = 0.0f, y = 0.0f;
	ffxQueryDescUpscaleGetJitterOffset q{};
	q.header.type = FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTEROFFSET;
	q.index = index; q.phaseCount = std::max(us.phaseCount, 1); q.pOutX = &x; q.pOutY = &y;
	if (FfxVk().Query(&us.ffx, &q.header) == FFX_API_RETURN_OK) { jx = x; jy = y; return true; }
#else
	(void)us; (void)index; (void)jx; (void)jy;
#endif
	return false;
}

bool FfxVkUpscaleDispatch(NukeDiligent::Impl& d, NukeDiligent::Impl::UpscaleState& us, ITextureView* srcSRV, ITextureView* reactSRV, float dt)
{
#if NUKE_FFX_VK
	if (!us.ffx || !srcSRV || !d.gbufDepthSRV || !d.gbufVelSRV || !us.out) return false;
	RefCntAutoPtr<IDeviceContextVk> ctxVk(d.context, IID_DeviceContextVk);
	if (!ctxVk) return false;
	// The inputs shader-readable, the output general (UAV) - through Diligent, so its layouts agree.
	ctxVk->TransitionImageLayout(srcSRV->GetTexture(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	ctxVk->TransitionImageLayout(d.gbufDepthSRV->GetTexture(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	ctxVk->TransitionImageLayout(d.gbufVelSRV->GetTexture(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	if (reactSRV) ctxVk->TransitionImageLayout(reactSRV->GetTexture(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	ctxVk->TransitionImageLayout(us.out, VK_IMAGE_LAYOUT_GENERAL);
	ffxDispatchDescUpscale dd{};
	dd.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
	dd.commandList = ctxVk->GetVkCommandBuffer();
	dd.color = FfxResVk(srcSRV->GetTexture(), FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ, false);
	dd.depth = FfxResVk(d.gbufDepthSRV->GetTexture(), FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ, false);
	dd.motionVectors = FfxResVk(d.gbufVelSRV->GetTexture(), FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ, false);
	dd.output = FfxResVk(us.out, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS, true);
	if (reactSRV) dd.reactive = FfxResVk(reactSRV->GetTexture(), FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ, false);
	dd.jitterOffset = { d.curJitterX, d.curJitterY };
	dd.motionVectorScale = { -(float)d.curRTW, -(float)d.curRTH };   // UV (current - previous) -> pixels (previous - current)
	dd.renderSize = { (uint32_t)d.curRTW, (uint32_t)d.curRTH };
	dd.upscaleSize = { (uint32_t)d.outW, (uint32_t)d.outH };
	dd.enableSharpening = d.curUpSharp > 0.0f;
	dd.sharpness = d.curUpSharp;
	dd.frameTimeDelta = dt * 1000.0f;
	dd.preExposure = 1.0f;
	dd.reset = us.fresh;
	dd.cameraNear = d.curNear; dd.cameraFar = d.curFar;
	dd.cameraFovAngleVertical = d.curFovY;
	dd.viewSpaceToMetersFactor = 1.0f;
	dd.flags = d.hdr ? 0u : (uint32_t)FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_SRGB;
	static int dispatchLog = 0;
	const bool say = dispatchLog < 2;
	if (say) { ++dispatchLog; cout << "[NukeDiligent]\tFSR (Vulkan): dispatch " << d.curRTW << "x" << d.curRTH << " -> " << d.outW << "x" << d.outH << (reactSRV ? " with reactive" : "") << endl; }
	const ffxReturnCode_t rc = FfxVk().Dispatch(&us.ffx, &dd.header);
	if (rc != FFX_API_RETURN_OK) cout << "[NukeDiligent]\tFSR (Vulkan) dispatch failed: " << rc << endl;
	else if (say) cout << "[NukeDiligent]\tFSR (Vulkan): dispatch ok" << endl;
	d.context->Flush();   // the vendor recorded its own pipelines / sets on our command buffer: submit, start a fresh one (as the D3D12 path does)
	return rc == FFX_API_RETURN_OK;
#else
	(void)d; (void)us; (void)srcSRV; (void)reactSRV; (void)dt;
	return false;
#endif
}

void FfxVkUpscaleDestroy(NukeDiligent::Impl::UpscaleState& us)
{
#if NUKE_FFX_VK
	if (us.ffx && FfxVk().ok()) { FfxVk().DestroyContext(&us.ffx, nullptr); us.ffx = nullptr; }
#else
	(void)us;
#endif
}

// ---- XeSS super resolution on Vulkan ---------------------------------------------------------

bool XessVkAvailable()
{
#if NUKE_XESS
	return XessVk().ok() && g_vk.device != VK_NULL_HANDLE;
#else
	return false;
#endif
}

std::string XessVkVariantName()
{
#if NUKE_XESS
	xess_version_t ver{};
	if (XessVk().GetVersion) XessVk().GetVersion(&ver);
	return "XeSS " + std::to_string(ver.major) + "." + std::to_string(ver.minor) + "." + std::to_string(ver.patch) + " (Vulkan)";
#else
	return "XeSS (Vulkan)";
#endif
}

bool XessVkQueryRenderSize(int quality, int ow, int oh, int& rw, int& rh)
{
#if NUKE_XESS
	if (!XessVk().ok() || !g_vk.device) return false;
	if (!g_xessVkProbe)
	{
		xess_context_handle_t h = nullptr;
		if (XessVk().CreateContext(g_vk.instance, g_vk.physDev, g_vk.device, &h) == XESS_RESULT_SUCCESS) g_xessVkProbe = h;
	}
	if (!g_xessVkProbe) return false;
	xess_2d_t out{ (uint32_t)ow, (uint32_t)oh }, opt{}, mn{}, mx{};
	if (XessVk().GetOptimalInputResolution((xess_context_handle_t)g_xessVkProbe, &out, XessVkQualityOf(quality), &opt, &mn, &mx) == XESS_RESULT_SUCCESS && opt.x > 0 && opt.y > 0)
	{ rw = (int)opt.x; rh = (int)opt.y; return true; }
#else
	(void)quality; (void)ow; (void)oh; (void)rw; (void)rh;
#endif
	return false;
}

bool XessVkUpscaleCreate(NukeDiligent::Impl& d, NukeDiligent::Impl::UpscaleState& us, int ow, int oh)
{
#if NUKE_XESS
	(void)ow; (void)oh;
	if (!XessVk().ok() || !g_vk.device) return false;
	xess_context_handle_t h = nullptr;
	const xess_result_t r = XessVk().CreateContext(g_vk.instance, g_vk.physDev, g_vk.device, &h);
	if (r != XESS_RESULT_SUCCESS || !h) { cout << "[NukeDiligent]\tXeSS (Vulkan): context refused: " << (int)r << endl; return false; }
	us.xess = h;
	if (XessVk().SetLoggingCallback) XessVk().SetLoggingCallback(h, XESS_LOGGING_LEVEL_WARNING, XessVkLog);
	// The pipelines build in the background (the warm-up rule); Init lands when they are done.
	const uint32_t flags = XESS_INIT_FLAG_ENABLE_AUTOEXPOSURE | XESS_INIT_FLAG_RESPONSIVE_PIXEL_MASK | (d.hdr ? 0u : (uint32_t)XESS_INIT_FLAG_LDR_INPUT_COLOR);
	us.xessFlags = flags;
	if (XessVk().BuildPipelines && XessVk().GetPipelineBuildStatus) XessVk().BuildPipelines(h, VK_NULL_HANDLE, false, flags);
	else us.xessBuilt = true;
	return true;
#else
	(void)d; (void)us; (void)ow; (void)oh;
	return false;
#endif
}

bool XessVkUpscaleDispatch(NukeDiligent::Impl& d, NukeDiligent::Impl::UpscaleState& us, ITextureView* srcSRV, ITextureView* reactSRV)
{
#if NUKE_XESS
	if (!us.xess || !srcSRV || !d.gbufDepthSRV || !d.gbufVelSRV || !us.out) return false;
	xess_context_handle_t h = (xess_context_handle_t)us.xess;
	if (!us.ready)
	{
		if (!us.xessBuilt)
		{
			const xess_result_t st = XessVk().GetPipelineBuildStatus(h);
			if (st == XESS_RESULT_ERROR_OPERATION_IN_PROGRESS) return false;   // still compiling: this frame stretches
			us.xessBuilt = true;
		}
		xess_vk_init_params_t ip{};
		ip.outputResolution = { (uint32_t)d.outW, (uint32_t)d.outH };
		ip.qualitySetting = XessVkQualityOf(d.curUpQuality);
		ip.initFlags = us.xessFlags;
		const xess_result_t r = XessVk().Init(h, &ip);
		if (r != XESS_RESULT_SUCCESS) { cout << "[NukeDiligent]\tXeSS (Vulkan) init failed: " << (int)r << endl; XessVk().DestroyContext(h); us.xess = nullptr; return false; }
		XessVk().SetVelocityScale(h, -(float)d.curRTW, -(float)d.curRTH);   // UV (current - previous) -> pixels (previous - current)
		us.ready = true;
		cout << "[NukeDiligent]\tXeSS (Vulkan): initialised " << d.curRTW << "x" << d.curRTH << " -> " << d.outW << "x" << d.outH << endl;
	}
	RefCntAutoPtr<IDeviceContextVk> ctxVk(d.context, IID_DeviceContextVk);
	if (!ctxVk) return false;
	auto info = [&](ITextureView* v) -> xess_vk_image_view_info
	{
		xess_vk_image_view_info i{};
		if (!v) return i;
		ITexture* t = v->GetTexture();
		const TextureDesc& td = t->GetDesc();
		i.imageView = ViewOf(v); i.image = ImageOf(t);
		i.subresourceRange = { (VkImageAspectFlags)(IsDepth(td.Format) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), 0u, 1u, 0u, 1u };
		i.format = VkFormatOf(td.Format); i.width = td.Width; i.height = td.Height;
		return i;
	};
	d.context->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE);
	ctxVk->TransitionImageLayout(srcSRV->GetTexture(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	ctxVk->TransitionImageLayout(d.gbufDepthSRV->GetTexture(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	ctxVk->TransitionImageLayout(d.gbufVelSRV->GetTexture(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	if (reactSRV) ctxVk->TransitionImageLayout(reactSRV->GetTexture(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	ctxVk->TransitionImageLayout(us.out, VK_IMAGE_LAYOUT_GENERAL);
	xess_vk_execute_params_t ep{};
	ep.colorTexture = info(srcSRV);
	ep.velocityTexture = info(d.gbufVelSRV);
	ep.depthTexture = info(d.gbufDepthSRV);
	if (reactSRV) ep.responsivePixelMaskTexture = info(reactSRV);
	ep.outputTexture = info(us.out->GetDefaultView(TEXTURE_VIEW_UNORDERED_ACCESS));
	ep.jitterOffsetX = d.curJitterX; ep.jitterOffsetY = d.curJitterY;
	ep.exposureScale = 1.0f;
	ep.resetHistory = us.fresh ? 1u : 0u;
	ep.inputWidth = (uint32_t)d.curRTW; ep.inputHeight = (uint32_t)d.curRTH;
	const xess_result_t r = XessVk().Execute(h, ctxVk->GetVkCommandBuffer(), &ep);
	if (r != XESS_RESULT_SUCCESS) cout << "[NukeDiligent]\tXeSS (Vulkan) execute failed: " << (int)r << endl;
	d.context->Flush();   // the vendor recorded its own pipelines / sets on our command buffer: submit, start a fresh one
	return r == XESS_RESULT_SUCCESS;
#else
	(void)d; (void)us; (void)srcSRV; (void)reactSRV;
	return false;
#endif
}

void XessVkDestroy(NukeDiligent::Impl::UpscaleState& us)
{
#if NUKE_XESS
	if (us.xess && XessVk().ok()) { XessVk().DestroyContext((xess_context_handle_t)us.xess); us.xess = nullptr; }
#else
	(void)us;
#endif
}

void XessVkShutdown()
{
#if NUKE_XESS
	if (g_xessVkProbe && XessVk().ok()) { XessVk().DestroyContext((xess_context_handle_t)g_xessVkProbe); g_xessVkProbe = nullptr; }
#endif
}

}  // namespace nukediligent

#else   // !_WIN32: no vendor Vulkan runtimes ship for Linux / macOS in these SDKs

void NukeDiligent::Impl::FrameGenVkBeforeDevice(bool) {}
void NukeDiligent::Impl::FrameGenVkAfterDevice() {}
bool NukeDiligent::Impl::FrameGenVkAttach(int, int) { return false; }
void NukeDiligent::Impl::FrameGenVkDetach() {}
void NukeDiligent::Impl::FrameGenVkBeforePresent(bool, float, int, int) {}
double NukeDiligent::Impl::FrameGenVkPresentRatio() { return 0.0; }

#endif
