#include "NukeDiligentImpl.h"
#include "RenderDeviceD3D12.h"   // drain the D3D12 debug layer into OUR console (see below)
#include "SwapChainD3D12.h"      // ISwapChainD3D12::GetDXGISwapChain (bind into DComp)
#include "SwapChainD3D11.h"
#include <config.h>              // nuke::WindowMode (window display mode)
#include <d3d12.h>
#include <dxgidebug.h>           // IDXGIInfoQueue: DXGI's OWN error queue (swapchain/present faults)
#include <dcomp.h>               // DirectComposition (per-pixel window transparency)
#include <cstdlib>               // std::getenv (NUKE_GPU_VALIDATION opt-in)
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>   // shader/PSO cache file IO
#include <iterator>              // istreambuf_iterator (cache load)

// NUKE PATCH global (DEFINED in the vendored SwapChainD3DBase.cpp so the Diligent DLLs resolve
// it too): true => the PRIMARY swap chain is created for DirectComposition (premultiplied
// alpha) instead of the HWND. Set only around the transparent window's swap-chain creation.
extern "C" bool g_NukeCompositionSwapChain;

#include "DebugOutput.h"   // Diligent::SetDebugMessageCallback

// Diligent's DEFAULT debug output writes to the console from whatever thread logs — the
// async shader-compile workers included. Concurrent fwrite on one stream trips the debug
// CRT ("Inconsistent Stream Count"). Route every Diligent message through one mutex.
static void NukeDiligentLogCallback(Diligent::DEBUG_MESSAGE_SEVERITY sev, const Diligent::Char* msg,
                                    const char* /*func*/, const char* /*file*/, int /*line*/)
{
	static std::mutex logMutex;
	std::lock_guard<std::mutex> lk(logMutex);
	const char* tag = sev == Diligent::DEBUG_MESSAGE_SEVERITY_FATAL_ERROR ? "FATAL"
	                : sev == Diligent::DEBUG_MESSAGE_SEVERITY_ERROR       ? "ERROR"
	                : sev == Diligent::DEBUG_MESSAGE_SEVERITY_WARNING     ? "Warning" : "Info";
	std::cout << "Diligent Engine: " << tag << ": " << (msg ? msg : "") << std::endl;
}

// The D3D12 debug layer reports the ACTUAL invalid operation (what the "Failed to close
// the command list" / device-removed asserts are symptoms of) — but only into the
// debugger's output window, which nobody sees on a console run. Drain the info queue
// into stdout every frame so the console names the real error, not the aftermath.
// On device removal: print the reason + the DRED breadcrumb trail (the EXACT command
// list/op the GPU faulted on — command lists execute async, so the CPU-side assert
// location is meaningless) + the page-fault allocation, once.
static void DumpDeviceRemoval(ID3D12Device* d3dDev)
{
	const HRESULT reason = d3dDev->GetDeviceRemovedReason();
	if (SUCCEEDED(reason)) return;
	static bool dumped = false;
	if (dumped) return;
	dumped = true;
	std::cout << "[D3D12] ===== DEVICE REMOVED, reason=0x" << std::hex << (unsigned long)reason << std::dec
	          << " (887A0005=REMOVED/page fault, 887A0006=HUNG, 887A0007=RESET, 887A0020=DRIVER_INTERNAL) =====" << std::endl;
	ID3D12DeviceRemovedExtendedData* dred = nullptr;
	if (SUCCEEDED(d3dDev->QueryInterface(__uuidof(ID3D12DeviceRemovedExtendedData), (void**)&dred)) && dred)
	{
		D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT bc{};
		if (SUCCEEDED(dred->GetAutoBreadcrumbsOutput(&bc)))
		{
			// Removals that aren't a classic page fault (e.g. ACCESS_DENIED) often leave every
			// list either untouched or fully completed — the old mid-list-only filter printed
			// NOTHING for those ("empty report"). Print one line per list so the state right
			// before the removal is always visible; mid-list ones are the actual suspects.
			int lists = 0;
			for (const D3D12_AUTO_BREADCRUMB_NODE* n = bc.pHeadAutoBreadcrumbNode; n; n = n->pNext)
			{
				++lists;
				const UINT done = n->pLastBreadcrumbValue ? *n->pLastBreadcrumbValue : 0;
				const bool midList = done != 0 && done != n->BreadcrumbCount;
				std::cout << "[D3D12] command list '"
				          << (n->pCommandListDebugNameA ? n->pCommandListDebugNameA : "?")
				          << (done == 0 ? "' not started " : midList ? "' STOPPED MID-LIST (fault suspect) at " : "' completed ")
				          << done << "/" << n->BreadcrumbCount
				          << ", last op id=" << (n->pCommandHistory && n->BreadcrumbCount
				                                 ? (int)n->pCommandHistory[done ? done - 1 : 0] : -1)
				          << " (2=Draw 3=DrawIndexed 4=ExecuteIndirect 8=CopyResource 13=Dispatch 27=DispatchRays 30=BuildRaytracingAS)" << std::endl;
			}
			if (lists == 0)
				std::cout << "[D3D12] no DRED breadcrumbs recorded (arm with NUKE_DRED=1)" << std::endl;
		}
		D3D12_DRED_PAGE_FAULT_OUTPUT pf{};
		if (SUCCEEDED(dred->GetPageFaultAllocationOutput(&pf)) && pf.PageFaultVA)
		{
			std::cout << "[D3D12] PAGE FAULT at GPU VA 0x" << std::hex << (unsigned long long)pf.PageFaultVA << std::dec << std::endl;
			for (const D3D12_DRED_ALLOCATION_NODE* a = pf.pHeadExistingAllocationNode; a; a = a->pNext)
				if (a->ObjectNameA) std::cout << "[D3D12]   live allocation near VA: " << a->ObjectNameA << std::endl;
			for (const D3D12_DRED_ALLOCATION_NODE* a = pf.pHeadRecentFreedAllocationNode; a; a = a->pNext)
				if (a->ObjectNameA) std::cout << "[D3D12]   RECENTLY FREED at VA (use-after-free suspect): " << a->ObjectNameA << std::endl;
		}
		dred->Release();
	}
	std::cout << "[D3D12] ===== end of device-removal report =====" << std::endl;
}

static void DrainD3D12DebugMessages(Diligent::IRenderDevice* dev, bool useD3D12)
{
	if (!useD3D12 || !dev) return;
	static ID3D12InfoQueue* iq = nullptr;   // process-lifetime cache
	static ID3D12Device* d3dDev = nullptr;
	static bool tried = false;
	if (!tried)
	{
		tried = true;
		Diligent::RefCntAutoPtr<Diligent::IRenderDeviceD3D12> d12(dev, Diligent::IID_RenderDeviceD3D12);
		if (d12 && d12->GetD3D12Device())
		{
			d3dDev = d12->GetD3D12Device();
			d3dDev->QueryInterface(__uuidof(ID3D12InfoQueue), (void**)&iq);
		}
	}
	if (d3dDev) DumpDeviceRemoval(d3dDev);   // async GPU faults surface HERE, with breadcrumbs
	if (!iq) return;
	const Diligent::Uint64 n = iq->GetNumStoredMessages();
	static Diligent::Uint64 seen = 0;
	if (n < seen) seen = 0;                 // queue was cleared/rolled — start over
	for (Diligent::Uint64 i = seen; i < n; ++i)
	{
		SIZE_T len = 0;
		iq->GetMessage(i, nullptr, &len);
		if (!len) continue;
		std::vector<char> buf(len);
		D3D12_MESSAGE* m = reinterpret_cast<D3D12_MESSAGE*>(buf.data());
		if (SUCCEEDED(iq->GetMessage(i, m, &len))
		    && m->Severity <= D3D12_MESSAGE_SEVERITY_WARNING && m->pDescription)
			std::cout << "[D3D12] " << m->pDescription << std::endl;
	}
	seen = n;

	// DXGI keeps its OWN info queue — swapchain/present errors land THERE, never in the
	// D3D12 device queue. The "silent" ACCESS_DENIED removals were DXGI naming the guilty
	// call into a queue nobody read.
#ifdef _WIN32
	// DXGI_DEBUG_ALL without dxguid.lib (same GUID, local definition).
	static const GUID kDxgiDebugAll = { 0xe48ae283, 0xda80, 0x490b, { 0x87, 0xe6, 0x43, 0xe9, 0xa9, 0xcf, 0xda, 0x08 } };
	static IDXGIInfoQueue* dxgiIq = nullptr;
	static bool dxgiTried = false;
	if (!dxgiTried)
	{
		dxgiTried = true;
		if (HMODULE dbg = LoadLibraryA("dxgidebug.dll"))
		{
			typedef HRESULT(WINAPI* PFN)(UINT, REFIID, void**);
			if (PFN get = (PFN)GetProcAddress(dbg, "DXGIGetDebugInterface1"))
				get(0, __uuidof(IDXGIInfoQueue), (void**)&dxgiIq);
		}
	}
	if (dxgiIq)
	{
		const UINT64 dn = dxgiIq->GetNumStoredMessages(kDxgiDebugAll);
		static UINT64 dseen = 0;
		if (dn < dseen) dseen = 0;
		for (UINT64 i = dseen; i < dn; ++i)
		{
			SIZE_T len = 0;
			dxgiIq->GetMessage(kDxgiDebugAll, i, nullptr, &len);
			if (!len) continue;
			std::vector<char> buf(len);
			DXGI_INFO_QUEUE_MESSAGE* m = reinterpret_cast<DXGI_INFO_QUEUE_MESSAGE*>(buf.data());
			if (SUCCEEDED(dxgiIq->GetMessage(kDxgiDebugAll, i, m, &len))
			    && m->Severity <= DXGI_INFO_QUEUE_MESSAGE_SEVERITY_WARNING && m->pDescription)
				std::cout << "[DXGI] " << m->pDescription << std::endl;
		}
		dseen = dn;
	}
#endif
}

bool NukeDiligent::Impl::DeviceRemoved()
{
	if (!useD3D12 || !device) return false;
	if (!d3d12DevCache)
	{
		Diligent::RefCntAutoPtr<Diligent::IRenderDeviceD3D12> d12(device, Diligent::IID_RenderDeviceD3D12);
		if (d12) d3d12DevCache = d12->GetD3D12Device();
	}
	ID3D12Device* dev = (ID3D12Device*)d3d12DevCache;
	if (!dev || SUCCEEDED(dev->GetDeviceRemovedReason())) return false;
	DumpDeviceRemoval(dev);   // prints reason once (+ DRED breadcrumbs when NUKE_GPU_VALIDATION is on)
	return true;
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
static void cb_scroll(GLFWwindow* w, double xo, double yo)
{
	auto* self = static_cast<NukeDiligent*>(glfwGetWindowUserPointer(w));
	if (!self) return;
	{
		std::lock_guard<std::mutex> l(self->m_uiInput.m);
		self->m_uiInput.scrollX += xo;
		self->m_uiInput.scrollY += yo;
	}
	if (!self->_UImouseWheel) return;
	double x = 0, y = 0; glfwGetCursorPos(w, &x, &y);
	self->_UImouseWheel(0, (int)yo, (int)x, (int)y);
}
static void cb_key(GLFWwindow* w, int key, int /*scancode*/, int action, int mods)
{
	auto* self = static_cast<NukeDiligent*>(glfwGetWindowUserPointer(w));
	if (!self) return;
	{
		std::lock_guard<std::mutex> l(self->m_uiInput.m);
		self->m_uiInput.keys.push_back({ key, action, mods });
		if (self->m_uiInput.keys.size() > 512) self->m_uiInput.keys.pop_front();
	}
	if (self->_UIkey) self->_UIkey(key, action, mods);
}
static void cb_char(GLFWwindow* w, unsigned int cp)
{
	auto* self = static_cast<NukeDiligent*>(glfwGetWindowUserPointer(w));
	if (!self) return;
	{
		std::lock_guard<std::mutex> l(self->m_uiInput.m);
		self->m_uiInput.chars.push_back(cp);
		if (self->m_uiInput.chars.size() > 512) self->m_uiInput.chars.pop_front();
	}
	if (self->_UIchar) self->_UIchar(cp);
}

// --- runtime-GUI input seam (2.5) --------------------------------------------------------
int NukeDiligent::fetchUIChars(unsigned int* out, int max)
{
	if (!out || max <= 0) return 0;
	std::lock_guard<std::mutex> l(m_uiInput.m);
	int n = 0;
	while (n < max && !m_uiInput.chars.empty())
	{
		out[n++] = m_uiInput.chars.front();
		m_uiInput.chars.pop_front();
	}
	return n;
}

int NukeDiligent::fetchUIKeys(int* keys, int* actions, int* mods, int max)
{
	if (!keys || !actions || !mods || max <= 0) return 0;
	std::lock_guard<std::mutex> l(m_uiInput.m);
	int n = 0;
	while (n < max && !m_uiInput.keys.empty())
	{
		const UIInput::Key& k = m_uiInput.keys.front();
		keys[n] = k.key; actions[n] = k.action; mods[n] = k.mods;
		++n;
		m_uiInput.keys.pop_front();
	}
	return n;
}

void NukeDiligent::getScrollDelta(double& x, double& y)
{
	std::lock_guard<std::mutex> l(m_uiInput.m);
	x = m_uiInput.scrollX; y = m_uiInput.scrollY;
	m_uiInput.scrollX = m_uiInput.scrollY = 0.0;
}

const char* NukeDiligent::getClipboardText()
{
	const char* t = m_window ? glfwGetClipboardString(m_window) : nullptr;
	return t ? t : "";
}

void NukeDiligent::setClipboardText(const char* text)
{
	if (m_window && text) glfwSetClipboardString(m_window, text);
}

NukeDiligent::NukeDiligent() : m_impl(new Impl()) {}
NukeDiligent::~NukeDiligent() { delete m_impl; }

void NukeDiligent::setShaderSource(const char* name, const char* source)
{
	if (name && source) m_impl->shaderSrc[name] = source;
}

int NukeDiligent::init(const WindowDesc& desc)
{
	int w = desc.w, h = desc.h;
	cout << "[NukeDiligent]\tinit(" << w << ", " << h << ")" << endl;
	// Serialize Diligent's log output (async shader-compile workers log concurrently).
	Diligent::SetDebugMessageCallback(&NukeDiligentLogCallback);
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
	// Always create WINDOWED; a fullscreen mode is applied right after init via applyWindow
	// (one code path for launch + runtime). Avoids the exclusive-at-create special case.
	const char*  title   = (desc.title && desc.title[0]) ? desc.title : "NukeEngine";
	m_window = glfwCreateWindow(w, h, title, nullptr, nullptr);
	if (!m_window) { cout << "[NukeDiligent]\tglfwCreateWindow failed" << endl; glfwTerminate(); return 1; }
	// GLFW window hints are STICKY (process-global). Reset transparency IMMEDIATELY: every window
	// created later through this GLFW instance (ImGui multi-viewport secondary windows) would
	// inherit it — and our patched GLFW gives transparent windows WS_EX_NOREDIRECTIONBITMAP, on
	// which an ordinary HWND swap chain's Present fails with DXGI_ERROR_ACCESS_DENIED and REMOVES
	// THE DEVICE (the "opening any detached window crashes" bug when the config was transparent).
	glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, GLFW_FALSE);

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

	m_impl->useD3D12  = (desc.backend == 1);
	m_impl->useVulkan = (desc.backend == 2);
	// Per-pixel transparency: the PRIMARY swap chain is built for DirectComposition (see the
	// vendored SwapChainD3DBase.hpp patch). D3D backends only — Vulkan has no DComp path.
	// Set the flag around creation only; reset after so secondary (UI viewport) swap chains
	// stay ordinary opaque HWND chains.
	g_NukeCompositionSwapChain = desc.transparent && !m_impl->useVulkan;
	Win32NativeWindow Window{ hWnd };
	SwapChainDesc SCDesc;
	// Match the World PSO + offscreen render targets (RGBA8_UNORM). Diligent defaults the
	// backbuffer to *_SRGB, which mismatches the world PSO when rendering straight to the
	// window (the Player) — "RTV format does not match PSO" spam. Keep one format everywhere.
	// HDR10 display output (Player): a 10-bit backbuffer carries the PQ-encoded HDR signal. Plain RGBA8 otherwise.
	SCDesc.ColorBufferFormat = m_impl->hdrOutput ? TEX_FORMAT_RGB10A2_UNORM : TEX_FORMAT_RGBA8_UNORM;
	IEngineFactory* engFactory = nullptr;
	if (m_impl->useD3D12)
	{
		auto* pFactory = GetEngineFactoryD3D12(); engFactory = pFactory;
		EngineD3D12CreateInfo EngineCI;
#ifdef _DEBUG
		// GPU validation + DRED are OPT-IN even in Debug: the D3D12 validation layer and DRED
		// auto-breadcrumbs (a marker before/after every command) cost real per-command time and
		// can HALVE the frame rate. 99% of Debug runs don't need them — enable ONLY when chasing a
		// device-removed / GPU fault: set env NUKE_GPU_VALIDATION=1 and rerun (no rebuild). Then the
		// actual invalid op lands in the console (DrainD3D12DebugMessages) instead of a bare fence assert.
		const char* gpuValEnv = std::getenv("NUKE_GPU_VALIDATION");
		const bool  wantVal = desc.gpuValidation                                   // config/main.json (works for double-click)
		                   || (gpuValEnv && gpuValEnv[0] && gpuValEnv[0] != '0');   // or the env var (terminal launches)
		if (wantVal)
		{
			// Validation errors (the CAUSE of a device removal) land in THIS log.
			EngineCI.SetValidationLevel(VALIDATION_LEVEL_1);
			cout << "[NukeDiligent]\tD3D12 debug layer ENABLED (gpuValidation) — expect lower FPS" << endl;
		}
		// DRED is a SEPARATE opt-in (env NUKE_DRED=1): its auto-breadcrumbs instrument EVERY command
		// list — and instrumented DXR dispatches (TraceRays / AS builds) intermittently wedge the
		// queue or remove the device with DXGI_ERROR_ACCESS_DENIED on some drivers (seen on RTX 50xx:
		// random first-RT-frame removals with an EMPTY DRED report and zero validation messages).
		// Never tie it to gpuValidation — enable it only when hunting an actual GPU fault.
		const char* dredEnv = std::getenv("NUKE_DRED");
		if (dredEnv && dredEnv[0] && dredEnv[0] != '0')
		{
			ID3D12DeviceRemovedExtendedDataSettings* dredSettings = nullptr;
			if (SUCCEEDED(D3D12GetDebugInterface(__uuidof(ID3D12DeviceRemovedExtendedDataSettings),
			                                     (void**)&dredSettings)) && dredSettings)
			{
				dredSettings->SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
				dredSettings->SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
				dredSettings->Release();
			}
			cout << "[NukeDiligent]\tDRED ENABLED (NUKE_DRED) — breadcrumb instrumentation on every command list" << endl;
		}
#endif
		// Editor-class descriptor budgets. The UI commits its SRB on every texture switch
		// (dynamic vars) and the editor renders EXTRA worlds (asset previews) plus one UI
		// pass per detached OS window — the library defaults overflow on heavy frames
		// (window restore = RT resizes + full redraw): "Failed to allocate N dynamic
		// GPU-visible CBV/SRV/UAV descriptors".
		// NOTE: the SAMPLER heap is capped at 2048 descriptors TOTAL by D3D12 itself —
		// it can only be REPARTITIONED (static vs dynamic), never grown. We use few
		// unique static samplers, so give the dynamic side the bigger share.
		EngineCI.GPUDescriptorHeapDynamicSize[0] = 131072;   // CBV/SRV/UAV (default 8k)
		EngineCI.GPUDescriptorHeapSize[0]        = 32768;    // static/mutable CBV/SRV/UAV
		EngineCI.GPUDescriptorHeapSize[1]        = 512;      // static/mutable samplers
		EngineCI.GPUDescriptorHeapDynamicSize[1] = 1536;     // dynamic samplers (512+1536 = the 2048 cap)
		pFactory->CreateDeviceAndContextsD3D12(EngineCI, &m_impl->device, &m_impl->context);
		if (!m_impl->device) { cout << "[NukeDiligent]\tD3D12 device creation failed" << endl; return 1; }
		pFactory->CreateSwapChainD3D12(m_impl->device, m_impl->context, SCDesc,
		                               FullScreenModeDesc{}, Window, &m_impl->swapChain);
	}
	else if (m_impl->useVulkan)
	{
		// VULKAN (task #138): same Diligent API surface, no DXGI anywhere — swap chains,
		// resizes and presents go through the Vulkan WSI, which is the stack every other
		// engine runs multi-window on. Shaders stay HLSL: Diligent compiles them to
		// SPIR-V with the vendored glslang (DXC is only used by the D3D12 RT path).
		auto* pFactory = GetEngineFactoryVk(); engFactory = pFactory;
		EngineVkCreateInfo EngineCI;
#ifdef _DEBUG
		const char* vkValEnv = std::getenv("NUKE_GPU_VALIDATION");
		if (desc.gpuValidation || (vkValEnv && vkValEnv[0] && vkValEnv[0] != '0'))
		{
			EngineCI.SetValidationLevel(VALIDATION_LEVEL_1);
			cout << "[NukeDiligent]\tVulkan validation layers ENABLED (gpuValidation)" << endl;
		}
#endif
		// Editor-class dynamic budgets, mirroring the D3D12 branch: the UI + preview worlds
		// + host windows burn through per-frame dynamic memory faster than the defaults.
		EngineCI.DynamicHeapSize = 32u << 20;
		// BACKGROUND shader compilation: cache-miss shaders compile on a worker pool in
		// parallel instead of serializing the boot (glslang is the slow part on Vulkan).
		EngineCI.Features.AsyncShaderCompilation = DEVICE_FEATURE_STATE_OPTIONAL;
		// Hardware ray tracing (VK_KHR_ray_tracing_pipeline / ray_query): request it —
		// unlike D3D12, Vulkan device features must be opted into at creation.
		EngineCI.Features.RayTracing = DEVICE_FEATURE_STATE_OPTIONAL;
		// RT shaders are SM6.x HLSL and need DXC (glslang can't parse them). ONE vendored
		// dxcompiler.dll (the official release) serves both backends — it emits DXIL for
		// D3D12 and SPIR-V for Vulkan; point Diligent at it instead of its default
		// "spv_dxcompiler.dll" name so we don't ship the compiler twice.
		EngineCI.pDxCompilerPath = "dxcompiler.dll";
		pFactory->CreateDeviceAndContextsVk(EngineCI, &m_impl->device, &m_impl->context);
		if (!m_impl->device) { cout << "[NukeDiligent]\tVulkan device creation failed" << endl; return 1; }
		pFactory->CreateSwapChainVk(m_impl->device, m_impl->context, SCDesc, Window, &m_impl->swapChain);
	}
	else
	{
		auto* pFactory = GetEngineFactoryD3D11(); engFactory = pFactory;
		EngineD3D11CreateInfo EngineCI;
		pFactory->CreateDeviceAndContextsD3D11(EngineCI, &m_impl->device, &m_impl->context);
		if (!m_impl->device) { cout << "[NukeDiligent]\tD3D11 device creation failed" << endl; return 1; }
		pFactory->CreateSwapChainD3D11(m_impl->device, m_impl->context, SCDesc,
		                               FullScreenModeDesc{}, Window, &m_impl->swapChain);
	}
	if (!m_impl->swapChain) { cout << "[NukeDiligent]\tswap chain creation failed" << endl; return 1; }
	g_NukeCompositionSwapChain = false;   // primary done — secondary swap chains stay opaque
	m_impl->transparent = desc.transparent;   // drives the alpha-0 clear + premultiplied final pass

	// Transparent window: bind the composition swap chain into a DirectComposition visual on
	// the HWND so its per-pixel alpha shows the desktop (the swap chain alone doesn't compose).
	if (desc.transparent && m_impl->useVulkan)
		cout << "[NukeDiligent]\twindow transparency is D3D-only (DirectComposition) — opaque on Vulkan" << endl;
	if (desc.transparent && !m_impl->useVulkan)
	{
		IDXGISwapChain* dxgiSC = nullptr;
		if (m_impl->useD3D12)
		{
			RefCntAutoPtr<ISwapChainD3D12> sc(m_impl->swapChain, IID_SwapChainD3D12);
			if (sc) dxgiSC = sc->GetDXGISwapChain();
		}
		else
		{
			RefCntAutoPtr<ISwapChainD3D11> sc(m_impl->swapChain, IID_SwapChainD3D11);
			if (sc) dxgiSC = sc->GetDXGISwapChain();
		}
		IDCompositionDevice* dcDev = nullptr;
		if (dxgiSC && SUCCEEDED(DCompositionCreateDevice(nullptr, __uuidof(IDCompositionDevice), (void**)&dcDev)))
		{
			IDCompositionTarget* dcTarget = nullptr;
			IDCompositionVisual* dcVisual = nullptr;
			HRESULT hrT = dcDev->CreateTargetForHwnd(hWnd, TRUE, &dcTarget);
			HRESULT hrV = dcDev->CreateVisual(&dcVisual);
			if (SUCCEEDED(hrT) && SUCCEEDED(hrV) && dcTarget && dcVisual)
			{
				dcVisual->SetContent(dxgiSC);       // AddRefs the swap chain
				dcTarget->SetRoot(dcVisual);
				dcDev->Commit();
				m_impl->dcompDevice = dcDev;
				m_impl->dcompTarget = dcTarget;
				m_impl->dcompVisual = dcVisual;
				cout << "[NukeDiligent]\tDirectComposition transparency active (premultiplied alpha)" << endl;
			}
			else
			{
				if (dcVisual) dcVisual->Release();
				if (dcTarget) dcTarget->Release();
				dcDev->Release();
				cout << "[NukeDiligent]\tDComp visual setup failed (hrT=0x" << std::hex << hrT
				     << " hrV=0x" << hrV << std::dec << ") — window opaque" << endl;
			}
		}
		else
			cout << "[NukeDiligent]\tDComp device creation failed — window opaque" << endl;
	}
	// Shader #include resolver (+ RT shader loader): resolves "rt_common.hlsl" etc. from the shaders directory.
	if (engFactory) engFactory->CreateDefaultShaderSourceStreamFactory("shaders", &m_impl->shaderFactory);
	// Ray tracing: D3D12 (DXR) or Vulkan (VK_KHR_ray_tracing) + a capable GPU/driver.
	// The whole RT path is Diligent-API (BLAS/TLAS/SBT); shaders compile through DXC —
	// DXIL on D3D12, SPIR-V on Vulkan (Diligent picks the target per backend).
	m_impl->rtSupported = (m_impl->useD3D12 || m_impl->useVulkan) && m_impl->device &&
	                      (m_impl->device->GetAdapterInfo().RayTracing.CapFlags & RAY_TRACING_CAP_FLAG_STANDALONE_SHADERS) != 0;
	cout << "[NukeDiligent]\tbackend=" << (m_impl->useD3D12 ? "D3D12" : m_impl->useVulkan ? "Vulkan" : "D3D11")
	     << " rayTracing=" << (m_impl->rtSupported ? "yes" : "no") << endl;
	// NOTE: the RT fallback TLAS is built at the TOP OF THE FIRST FRAME, not here — on
	// Vulkan an acceleration-structure build before the frame loop starts deadlocks in
	// the upload path (a fence with no submission to signal it). See render().

	if (m_impl->hdrOutput && !m_impl->useVulkan)
		m_impl->SetupHDROutput();   // HDR10 colour space via DXGI — D3D backends only for now

	const SwapChainDesc& scd = m_impl->swapChain->GetDesc();
	m_impl->CreateUIPipeline(scd.ColorBufferFormat, scd.DepthBufferFormat);
	m_impl->CreateWorldPipeline();

	width  = w;
	height = h;

	cout << "[NukeDiligent]\tdevice=" << m_impl->device.RawPtr()
	     << " swapChain=" << m_impl->swapChain.RawPtr() << endl;

	// Launch straight into the requested display mode (window created windowed above; the
	// swap chain follows the framebuffer on the first frame). Same one path as the runtime API.
	m_windowMode = (int)WindowMode::Windowed;
	if (desc.mode != (int)WindowMode::Windowed)
		applyWindow(desc);

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
#ifdef _DEBUG
	DrainD3D12DebugMessages(m_impl->device, m_impl->useD3D12);   // real validation errors -> console
#endif

	// A removed device can't run ANY of the frame below — mapping/creating on it just cascades
	// ("Failed to create dynamic page", "Buffer already mapped" asserts). Suspend rendering
	// entirely (events still pump, the reason was already printed once) so the process stays
	// alive and the console keeps the REAL cause on top instead of post-mortem noise.
	if (m_impl->DeviceRemoved())
	{
		static bool said = false;
		if (!said) { said = true; cout << "[NukeDiligent]\trendering SUSPENDED (device removed — see the report above)" << endl; }
		return 1;
	}

	// Centralized GPU lifetime: advance the frame clock and free trash old enough that no in-flight
	// command list or recorded draw data can still reference it.
	++m_impl->frameId;
	m_impl->PurgeTrash();
	// Secondary-window swap chains: apply queued creations/resizes NOW, before anything is
	// recorded — doing it mid-frame under load intermittently wedged the queue.
	m_impl->ApplyPendingViewportOps();
	// RT fallback TLAS on the FIRST frame (idempotent): building it at init deadlocks
	// Vulkan's upload path (no frame in flight to signal the wait fence).
	if (m_impl->rtSupported && !m_impl->fallbackTLAS) m_impl->EnsureRTFallback();

	// Frame stats: latch the completed frame's counters for getFrameStats, start fresh.
	m_impl->statDrawsOut = m_impl->statDraws;
	m_impl->statTrisOut  = m_impl->statTris;
	m_impl->statDraws = 0;
	m_impl->statTris  = 0;

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
		// D3D12 removes the device if the swap-chain back buffers are still bound or referenced by
		// in-flight GPU work when Resize() runs — and a resize DRAG fires this every frame. Unbind +
		// flush + idle first (same as the secondary-viewport path in NukeDiligent_UI.cpp).
		m_impl->context->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE);
		m_impl->context->Flush();
		m_impl->device->IdleGPU();
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
	// New frame: drop last frame's debug/gizmo lines (emission happens in onRender below).
	// The depth batch is normally consumed per camera; clearing covers a pass that never ran.
	{
		std::lock_guard<std::mutex> lock(m_impl->debugMutex);
		m_impl->debugVerts.clear();
		m_impl->debugVertsDepth.clear();
	}
	// Deferred shadow-resolution change (rebuilds the shadow maps; never mid-frame).
	if (m_impl->pendingShadowRes > 0)
	{
		if (m_impl->pendingShadowRes != m_impl->shadowRes)
		{
			m_impl->shadowRes = m_impl->pendingShadowRes;
			m_impl->CreateShadowResources();
			std::cout << "[NukeDiligent]\tshadow res -> " << m_impl->shadowRes << std::endl;
		}
		m_impl->pendingShadowRes = 0;
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

	if (m_impl->DeviceRemoved()) return 1;   // device lost this frame: skip present, keep the app alive
	m_impl->swapChain->Present(m_impl->vsync ? 1 : 0);   // SyncInterval 1 = vsync, 0 = uncapped
	// Secondary windows present AFTER the main chain: their draw commands were recorded in
	// the frame body; presenting them mid-frame (Present flushes) split the stream between
	// an RT write and its sampling and tripped the debug layer into removing the device.
	// VULKAN native viewports: present the per-window swapchains after the main present.
	for (void* h : m_impl->vpPresentQueue)
	{
		auto it = m_impl->uiVpSC.find(h);
		if (it == m_impl->uiVpSC.end() || !it->second) continue;
		it->second->Present(0);
	}
	m_impl->vpPresentQueue.clear();
	// D3D detached windows get their pixels via GDI from the offscreen RTs rendered
	// during the frame (NO secondary swap chains — see uiViewportRender/BlitHostWindows).
	// After the main Present keeps the readback maps off the frame's critical path.
	m_impl->BlitHostWindows();
	m_impl->PollShaderSaves();   // persist finished background compiles into the disk cache
	return 1;
}

void NukeDiligent::setVSync(bool on) { m_impl->vsync = on; }   // takes effect on the next Present
bool NukeDiligent::getVSync()        { return m_impl->vsync; }
void NukeDiligent::setWireframe(bool on) { m_impl->wireframe = on; }   // renderObject picks psoWire per draw
bool NukeDiligent::getWireframe()        { return m_impl->wireframe; }

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

void NukeDiligent::requestClose()
{
	if (m_window) glfwSetWindowShouldClose(m_window, GLFW_TRUE);
}

// OUR shader-bytecode cache (Vulkan): key the FULL compile inputs, store the SPIR-V.
// A hit creates the shader from ByteCode — glslang never runs for it.
void NukeDiligent::Impl::CreateShaderCached(const ShaderCreateInfo& ci, IShader** pp)
{
	if (!useVulkan || !ci.Source)   // only source-based shaders on the slow-compile backend
	{
		device->CreateShader(ci, pp);
		return;
	}
	// FNV-1a over everything that affects codegen.
	auto fnv = [](uint64_t h, const void* data, size_t n)
	{
		const unsigned char* p = (const unsigned char*)data;
		for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
		return h;
	};
	uint64_t h = 1469598103934665603ull;
	const size_t srcLen = ci.SourceLength ? ci.SourceLength : strlen(ci.Source);
	h = fnv(h, ci.Source, srcLen);
	if (ci.EntryPoint) h = fnv(h, ci.EntryPoint, strlen(ci.EntryPoint));
	h = fnv(h, &ci.Desc.ShaderType, sizeof(ci.Desc.ShaderType));
	h = fnv(h, &ci.Desc.UseCombinedTextureSamplers, sizeof(bool));
	h = fnv(h, &ci.CompileFlags, sizeof(ci.CompileFlags));
	h = fnv(h, &ci.HLSLVersion, sizeof(ci.HLSLVersion));
	for (Uint32 i = 0; i < ci.Macros.Count; ++i)
	{
		if (ci.Macros[i].Name)       h = fnv(h, ci.Macros[i].Name, strlen(ci.Macros[i].Name));
		if (ci.Macros[i].Definition) h = fnv(h, ci.Macros[i].Definition, strlen(ci.Macros[i].Definition));
	}

	namespace bfs = boost::filesystem;
	char hex[24]; snprintf(hex, sizeof(hex), "%016llx", (unsigned long long)h);
	const bfs::path dir("config/shadercache_vk");
	const bfs::path file = dir / (std::string(hex) + ".spv");

	boost::system::error_code ec;
	if (bfs::exists(file, ec))
	{
		bfs::ifstream f(file, std::ios::binary);
		std::vector<char> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
		if (!bytes.empty())
		{
			ShaderCreateInfo c2 = ci;
			c2.Source = nullptr; c2.SourceLength = 0; c2.FilePath = nullptr;
			c2.ByteCode = bytes.data();
			c2.ByteCodeSize = bytes.size();
			device->CreateShader(c2, pp);
			if (*pp) return;   // corrupt/stale bytecode falls through to a fresh compile
			cout << "[NukeDiligent]\tshader cache entry rejected, recompiling (" << hex << ")" << endl;
		}
	}

	// Cache miss: compile in the BACKGROUND (worker pool) when the device supports it —
	// the batch of shaders created around this one compiles in parallel; the PSO helper
	// waits for readiness, so nothing downstream ever sees a half-compiled shader.
	ShaderCreateInfo cc = ci;
	if (device->GetDeviceInfo().Features.AsyncShaderCompilation)
		cc.CompileFlags |= SHADER_COMPILE_FLAG_ASYNCHRONOUS;
	device->CreateShader(cc, pp);
	if (*pp)
		pendingShaderSaves.emplace_back(RefCntAutoPtr<IShader>(*pp), file.string());
}

// Write finished cache-miss compiles to disk (called once per frame — the bytecode of an
// async shader only exists after its worker finishes).
void NukeDiligent::Impl::PollShaderSaves()
{
	if (pendingShaderSaves.empty()) return;
	namespace bfs = boost::filesystem;
	for (size_t i = 0; i < pendingShaderSaves.size(); )
	{
		IShader* s = pendingShaderSaves[i].first;
		const SHADER_STATUS st = s->GetStatus(false);
		if (st == SHADER_STATUS_COMPILING || st == SHADER_STATUS_UNINITIALIZED) { ++i; continue; }
		if (st == SHADER_STATUS_READY)
		{
			const void* bc = nullptr; Uint64 n = 0;
			s->GetBytecode(&bc, n);
			if (bc && n)
			{
				boost::system::error_code ec;
				const bfs::path file(pendingShaderSaves[i].second);
				bfs::create_directories(file.parent_path(), ec);
				bfs::ofstream f(file, std::ios::binary | std::ios::trunc);
				if (f) f.write((const char*)bc, (std::streamsize)n);
			}
		}
		pendingShaderSaves.erase(pendingShaderSaves.begin() + i);
	}
}

void NukeDiligent::deinit()
{
	for (auto& cb : m_impl->onClose) cb();
	// Drain the GPU trash AFTER the queue settles — parked objects must not outlive the device.
	if (m_impl->context && m_impl->device)
	{
		m_impl->context->Flush();
		m_impl->device->IdleGPU();
	}
	m_impl->PurgeTrash(true);
	// DirectComposition (transparent window): release visual -> target -> device before the
	// swap chain they reference.
	if (m_impl->dcompVisual) { m_impl->dcompVisual->Release(); m_impl->dcompVisual = nullptr; }
	if (m_impl->dcompTarget) { m_impl->dcompTarget->Release(); m_impl->dcompTarget = nullptr; }
	if (m_impl->dcompDevice) { m_impl->dcompDevice->Release(); m_impl->dcompDevice = nullptr; }
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

// Runtime window change (Game.Set* -> iRender::applyWindow). The swap chain follows the new
// framebuffer size in the render loop (glfwGetFramebufferSize), so we only drive the WINDOW.
void NukeDiligent::applyWindow(const WindowDesc& d)
{
	if (!m_window) return;
	GLFWmonitor* mon = glfwGetPrimaryMonitor();
	const GLFWvidmode* vm = mon ? glfwGetVideoMode(mon) : nullptr;

	// Leaving windowed: remember the current windowed rect so a later return restores it.
	if (m_windowMode == (int)WindowMode::Windowed && d.mode != (int)WindowMode::Windowed)
		glfwGetWindowPos(m_window, &m_winX, &m_winY), glfwGetWindowSize(m_window, &m_winW, &m_winH);

	if (d.mode == (int)WindowMode::Windowed)
	{
		// Decoration can only change while NOT monitor-fullscreen.
		glfwSetWindowMonitor(m_window, nullptr,
		                     m_winX >= 0 ? m_winX : 64, m_winY >= 0 ? m_winY : 64,
		                     d.w > 0 ? d.w : 1280, d.h > 0 ? d.h : 720, GLFW_DONT_CARE);
		glfwSetWindowAttrib(m_window, GLFW_DECORATED, d.decorated ? GLFW_TRUE : GLFW_FALSE);
		glfwSetWindowAttrib(m_window, GLFW_RESIZABLE, d.resizable ? GLFW_TRUE : GLFW_FALSE);
	}
	else if (d.mode == (int)WindowMode::BorderlessFullscreen)
	{
		// Undecorated window covering the monitor at the DESKTOP resolution (no mode switch).
		int mx = 0, my = 0; if (mon) glfwGetMonitorPos(mon, &mx, &my);
		glfwSetWindowAttrib(m_window, GLFW_DECORATED, GLFW_FALSE);
		glfwSetWindowMonitor(m_window, nullptr, mx, my,
		                     vm ? vm->width : (d.w > 0 ? d.w : 1280),
		                     vm ? vm->height : (d.h > 0 ? d.h : 720), GLFW_DONT_CARE);
	}
	else   // WindowMode::ExclusiveFullscreen — real GLFW fullscreen, monitor switches to our resolution
	{
		int rw = d.w > 0 ? d.w : (vm ? vm->width : 1280);
		int rh = d.h > 0 ? d.h : (vm ? vm->height : 720);
		glfwSetWindowMonitor(m_window, mon, 0, 0, rw, rh, vm ? vm->refreshRate : GLFW_DONT_CARE);
	}

	glfwSetWindowOpacity(m_window, d.opacity <= 0.0f ? 1.0f : d.opacity);
	m_windowMode = d.mode;
	if (d.transparent)
		cout << "[NukeDiligent]\tper-pixel transparency is a creation-time property — applies on next launch" << endl;
	cout << "[NukeDiligent]\tapplyWindow mode=" << d.mode << " " << d.w << "x" << d.h
	     << " decorated=" << d.decorated << " opacity=" << d.opacity << endl;
}
void NukeDiligent::getCursorPos(double& x, double& y) { x = y = 0; if (m_window) glfwGetCursorPos(m_window, &x, &y); }
bool NukeDiligent::isMouseButtonDown(int b) { return m_window && glfwGetMouseButton(m_window, b) == GLFW_PRESS; }

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
