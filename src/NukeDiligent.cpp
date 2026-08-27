#include "NukeDiligentImpl.h"
#ifdef _WIN32
#include "RenderDeviceD3D12.h"   // D3D12 debug-layer drain
#include "SwapChainD3D12.h"      // ISwapChainD3D12::GetDXGISwapChain (bind into DComp)
#include "SwapChainD3D11.h"
#endif
#include <config.h>              // nuke::WindowMode (window display mode)
#include <API/Model/Game.h>      // Game::FlushScreenshot (end-of-frame capture)
#ifdef _WIN32
#include <d3d12.h>
#include <dxgidebug.h>           // IDXGIInfoQueue: DXGI's OWN error queue (swapchain/present faults)
#include <dcomp.h>               // DirectComposition (per-pixel window transparency)
#endif
#include <cstdlib>               // std::getenv (NUKE_GPU_VALIDATION opt-in)
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>   // shader/PSO cache file IO
#include <boost/dll/runtime_symbol_info.hpp>   // program_location: cache dir is exe-relative
#include "ShaderSourceFactoryUtils.h"   // memory #include resolver over pushed shader sources
#include <iterator>              // istreambuf_iterator (cache load)

// NUKE PATCH global, defined in the vendored SwapChainD3DBase.cpp: true => the PRIMARY swap
// chain is created for DirectComposition (premultiplied alpha) instead of the HWND.
#ifdef _WIN32
extern "C" bool g_NukeCompositionSwapChain;
#endif

// NUKE PATCH globals (patches/DiligentCore-vk-alpha.patch, defined in SwapChainVkImpl.cpp):
// AlphaComposite => the PRIMARY Vulkan swap chain prefers an alpha-compositing mode — the
// transparent window's Vk/MoltenVK counterpart of the DComp path (CAMetalLayer.opaque
// follows it). HDR10 => request an ST2084 surface format (macOS: PQ CAMetalLayer + EDR);
// HDR10Active reports the outcome after creation.
extern "C" bool g_NukeVkAlphaComposite;
extern "C" bool g_NukeVkHDR10;
extern "C" bool g_NukeVkHDR10Active;

#include "DebugOutput.h"   // Diligent::SetDebugMessageCallback
#include <API/Model/CrashReport.h>   // probe runs: symbolized stack on a Diligent assert

#if !defined(_WIN32) && !defined(__APPLE__)
// Window-icon PNG decode (X11 has no icon-in-binary concept — the AppDir png is the icon).
// STATIC: DiligentCore vendors its own stb internally; two implementations must not meet.
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include <stb_image.h>
// Downscale before handing to the WM: _NET_WM_ICON stores 8 bytes/pixel, and a full-size
// logo blows past the X request limit — the server drops it SILENTLY (empty property).
#define STB_IMAGE_RESIZE_STATIC
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize2.h>
#endif

// Serializes Diligent log output: its workers log concurrently and concurrent fwrite on one
// stream trips the debug CRT.
static void NukeDiligentLogCallback(Diligent::DEBUG_MESSAGE_SEVERITY sev, const Diligent::Char* msg,
                                    const char* /*func*/, const char* /*file*/, int /*line*/)
{
	static std::mutex logMutex;
	std::lock_guard<std::mutex> lk(logMutex);
	const char* tag = sev == Diligent::DEBUG_MESSAGE_SEVERITY_FATAL_ERROR ? "FATAL"
	                : sev == Diligent::DEBUG_MESSAGE_SEVERITY_ERROR       ? "ERROR"
	                : sev == Diligent::DEBUG_MESSAGE_SEVERITY_WARNING     ? "Warning" : "Info";
	std::cout << "Diligent Engine: " << tag << ": " << (msg ? msg : "") << std::endl;
	// Probe runs (NUKE_ASSERT_STDERR): a failed Diligent VERIFY would pop a modal Abort/Retry
	// box right after this callback and hang a headless run forever — print and die instead.
	static const bool headlessAsserts = std::getenv("NUKE_ASSERT_STDERR") != nullptr;
	if (headlessAsserts && sev >= Diligent::DEBUG_MESSAGE_SEVERITY_ERROR
	    && msg && std::strstr(msg, "Debug assertion failed"))
	{
		std::fprintf(stderr, "Diligent ASSERT: %s\n", msg);
		nuke::CrashReport::PrintBacktrace();
		std::fflush(stderr);
		std::_Exit(3);
	}
}

#ifdef _WIN32
// Prints the device-removal reason plus the DRED breadcrumb trail and page-fault
// allocation, once per process.
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
			// One line per list: non-page-fault removals leave lists untouched or complete,
			// so a mid-list-only filter would print nothing.
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
	if (d3dDev) DumpDeviceRemoval(d3dDev);   // async GPU faults surface here
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

	// DXGI keeps its OWN info queue: swapchain/present errors land there, never in the D3D12 device queue.
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
	DumpDeviceRemoval(dev);   // prints once
	return true;
}
#else
// No D3D off Windows: the Vulkan backend has no device-removal notion to poll here.
static void DrainD3D12DebugMessages(Diligent::IRenderDevice*, bool) {}
bool NukeDiligent::Impl::DeviceRemoved() { return false; }
#endif   // _WIN32

static void glfw_error(int code, const char* desc)
{
	fprintf(stderr, "[NukeDiligent] GLFW error %d: %s\n", code, desc);
}

// GLFW input -> iRender neutral callbacks (raw input forwarded; the UI module interprets it).
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

NukeDiligent::Impl* NukeDiligent::nativeImpl = nullptr;

NukeDiligent::NukeDiligent() : m_impl(new Impl()) { nativeImpl = m_impl; }
NukeDiligent::~NukeDiligent() { if (nativeImpl == m_impl) nativeImpl = nullptr; delete m_impl; }

void NukeDiligent::setShaderSource(const char* name, const char* source)
{
	if (!name || !source) return;
	boost::mutex::scoped_lock l(m_impl->shaderLock);
	m_impl->shaderSrc[name] = source;
	++m_impl->shaderSrcVersion;   // the factory rebuilds lazily at the next ShaderFactory()
}

// The lock-guarded snapshot every compile holds for its whole duration. Lazy: pushes since the
// last build only bump the version; the replaced factory retires (never dies) because modules
// keep raw pointers from the native seam.
RefCntAutoPtr<IShaderSourceInputStreamFactory> NukeDiligent::Impl::ShaderFactory()
{
	boost::mutex::scoped_lock l(shaderLock);
	if (shaderFactoryVersion != shaderSrcVersion)
	{
		if (shaderFactory) retiredShaderFactories.push_back(shaderFactory);
		RebuildShaderFactory();
		shaderFactoryVersion = shaderSrcVersion;
	}
	return shaderFactory;
}

// Rebuilds the memory #include resolver over the sources pushed via setShaderSource (no disk IO).
// Every entry resolves both as "<name>" and "<name>.hlsl", since sources are pushed by stem.
// shaderLock is HELD by the caller.
void NukeDiligent::Impl::RebuildShaderFactory()
{
	std::vector<MemoryShaderSourceFileInfo> files;
	std::vector<std::string> hlslNames;                  // backing storage for "<name>.hlsl"
	files.reserve(shaderSrc.size() * 2);
	hlslNames.reserve(shaderSrc.size());                 // no realloc: c_str()s must stay valid
	// Cache keys hash only the top-level source, so fold every include (dot-less name) into one
	// hash CreateShaderCached mixes in — otherwise an include edit serves stale bytecode.
	uint64_t epoch = 1469598103934665603ull;
	for (auto& kv : shaderSrc)
	{
		if (kv.second.empty()) continue;
		files.emplace_back(kv.first.c_str(), kv.second.c_str(), (Uint32)kv.second.size());
		hlslNames.push_back(kv.first + ".hlsl");
		files.emplace_back(hlslNames.back().c_str(), kv.second.c_str(), (Uint32)kv.second.size());
		const bool isInc = kv.first.find('.') == std::string::npos
		                || (kv.first.size() > 6 && kv.first.compare(kv.first.size() - 6, 6, ".hlsli") == 0);
		if (isInc)
			for (unsigned char ch : kv.second) { epoch ^= ch; epoch *= 1099511628211ull; }
	}
	includeEpoch.store(epoch);
	MemoryShaderSourceFactoryCreateInfo mci{ files.data(), (Uint32)files.size(), True /*CopySources*/ };
	shaderFactory.Release();
	CreateMemoryShaderSourceFactory(mci, &shaderFactory);
}

// Hide-from-capture: the window renders for the user, but screen capture (screenshots,
// recorders, vision pipelines) sees the content BEHIND it. Live-toggleable. Windows:
// SetWindowDisplayAffinity; macOS: NSWindow.sharingType (cocoa shim); X11/Wayland: no protocol.
static void ApplyCaptureAffinity(GLFWwindow* wnd, bool hide)
{
	if (!wnd) return;
#ifdef _WIN32
#ifndef WDA_EXCLUDEFROMCAPTURE
#define WDA_EXCLUDEFROMCAPTURE 0x00000011   // pre-2004 SDK headers lack it; the OS call still works
#endif
	if (HWND h = glfwGetWin32Window(wnd))
		if (!SetWindowDisplayAffinity(h, hide ? WDA_EXCLUDEFROMCAPTURE : WDA_NONE))
			cout << "[NukeDiligent]\thideFromCapture failed (needs Windows 10 2004+)" << endl;
#elif defined(__APPLE__)
	NukeCocoaSetHiddenFromCapture(wnd, hide);
#else
	if (hide)
		cout << "[NukeDiligent]\thideFromCapture: X11/Wayland have no capture-exclusion protocol — ignored" << endl;
#endif
}

int NukeDiligent::init(const WindowDesc& desc)
{
	int w = desc.w, h = desc.h;
	cout << "[NukeDiligent]\tinit(" << w << ", " << h << ")" << endl;
	Diligent::SetDebugMessageCallback(&NukeDiligentLogCallback);
	if (w <= 0 || h <= 0) { cout << "[NukeDiligent]\tbad size, using 1280x720" << endl; w = 1280; h = 720; }

	glfwSetErrorCallback(glfw_error);
#ifdef __APPLE__
	// GLFW's Cocoa default chdir's into the bundle's Contents/Resources at init — the hosts
	// pinned CWD to the run root already, and every engine-relative path rides on it.
	glfwInitHint(GLFW_COCOA_CHDIR_RESOURCES, GLFW_FALSE);
#endif
#if !defined(_WIN32) && !defined(__APPLE__) && defined(GLFW_PLATFORM_X11)
	// Display backend: native Wayland BY DEFAULT when the session offers it, X11 (XWayland)
	// as the fallback — or whatever NUKE_DISPLAY_BACKEND=wayland|x11 demands. On Wayland the
	// UI module turns imgui multi-viewport off (no client-side window positioning there).
	bool wantedWayland = false;
	{
		const char* want = std::getenv("NUKE_DISPLAY_BACKEND");
		bool wayland = !(want && strcmp(want, "x11") == 0);
#ifdef GLFW_PLATFORM_WAYLAND
		wayland = wayland && std::getenv("WAYLAND_DISPLAY")
		                  && glfwPlatformSupported(GLFW_PLATFORM_WAYLAND);
#else
		wayland = false;
#endif
		wantedWayland = wayland;
#ifdef GLFW_PLATFORM_WAYLAND
		glfwInitHint(GLFW_PLATFORM, wayland ? GLFW_PLATFORM_WAYLAND : GLFW_PLATFORM_X11);
#else
		glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);
#endif
		cout << "[NukeDiligent]\tdisplay backend: " << (wayland ? "wayland (native)" : "x11")
		     << (want && *want ? " (NUKE_DISPLAY_BACKEND)" : "") << endl;
	}
	if (!glfwInit())
	{
		if (!wantedWayland) { cout << "[NukeDiligent]\tglfwInit failed" << endl; return 1; }
		cout << "[NukeDiligent]\tWayland init failed — falling back to X11" << endl;
		glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);
		if (!glfwInit()) { cout << "[NukeDiligent]\tglfwInit failed" << endl; return 1; }
	}
#else
	if (!glfwInit()) { cout << "[NukeDiligent]\tglfwInit failed" << endl; return 1; }
#endif

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // Diligent owns the graphics API
	glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);     // create hidden; show after the dark title-bar attr is set
	glfwWindowHint(GLFW_DECORATED, desc.decorated ? GLFW_TRUE : GLFW_FALSE);
	glfwWindowHint(GLFW_RESIZABLE, desc.resizable ? GLFW_TRUE : GLFW_FALSE);
	glfwWindowHint(GLFW_FLOATING,  desc.floating  ? GLFW_TRUE : GLFW_FALSE);
	glfwWindowHint(GLFW_MAXIMIZED, desc.maximized ? GLFW_TRUE : GLFW_FALSE);
#ifdef GLFW_MOUSE_PASSTHROUGH   // GLFW 3.4+: overlay windows let input fall through to the desktop
	glfwWindowHint(GLFW_MOUSE_PASSTHROUGH, desc.clickThrough ? GLFW_TRUE : GLFW_FALSE);
#endif
	if (desc.transparent)
		glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, GLFW_TRUE); // see-through also needs swapchain alpha (DComp)
	// Always create WINDOWED; fullscreen is applied right after init via applyWindow.
	const char*  title   = (desc.title && desc.title[0]) ? desc.title : "NukeEngine";
	m_window = glfwCreateWindow(w, h, title, nullptr, nullptr);
	if (!m_window) { cout << "[NukeDiligent]\tglfwCreateWindow failed" << endl; glfwTerminate(); return 1; }
	// GLFW hints are sticky/process-global: reset transparency at once, or later windows inherit
	// WS_EX_NOREDIRECTIONBITMAP (patched GLFW) and an HWND swap chain's Present fails ACCESS_DENIED.
	glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, GLFW_FALSE);

	glfwSetWindowUserPointer(m_window, this);
	glfwSetCursorPosCallback(m_window, cb_cursorpos);
	glfwSetMouseButtonCallback(m_window, cb_mousebtn);
	glfwSetScrollCallback(m_window, cb_scroll);
	glfwSetKeyCallback(m_window, cb_key);
	glfwSetCharCallback(m_window, cb_char);

#ifdef _WIN32
	HWND hWnd = glfwGetWin32Window(m_window);

	// Window icon = the host exe's first icon group (no fixed resource id).
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

	// Dark title bar — must be set while the window is still hidden (first non-client paint).
	{
		BOOL dark = TRUE;
		HRESULT hr = DwmSetWindowAttribute(hWnd, DWMWA_USE_IMMERSIVE_DARK_MODE, &dark, sizeof(dark));
		cout << "[NukeDiligent]\tdark title bar hr=0x" << std::hex << hr << std::dec << endl;
	}
#elif !defined(__APPLE__)
	// Window icon: X11 has no icon-in-binary concept — the run root's icon png IS the icon
	// (games: "<exe stem>.png" stamped by the packager; editor image: nukeengine-editor.png /
	// .DirIcon; dev tree: the editor's res/logo.png). GLFW publishes it as _NET_WM_ICON and
	// the UI module mirrors it onto secondary windows. macOS keeps using the bundle icon;
	// Wayland taskbars take it from the .desktop entry (glfwSetWindowIcon is a no-op there).
	if (!NukeGlfwIsWayland())
	{
		namespace ibfs = boost::filesystem;
		boost::system::error_code iec;
		const ibfs::path exe  = boost::dll::program_location(iec);
		const ibfs::path base = nuke::Config::baseDir();
		const ibfs::path cands[] = {
			base / (exe.stem().string() + ".png"),
			base / "nukeengine-editor.png",
			base / ".DirIcon",
			base.parent_path().parent_path().parent_path() / "NukeEngine-Editor" / "res" / "logo.png",
		};
		bool iconSet = false;
		for (const ibfs::path& c : cands)
			if (ibfs::exists(c, iec) && !ibfs::is_directory(c, iec))
			{
				int iw = 0, ih = 0, comp = 0;
				if (unsigned char* px = stbi_load(c.string().c_str(), &iw, &ih, &comp, 4))
				{
					// Standard WM sizes; the source stays untouched when already small.
					static const int kSizes[] = { 128, 48, 32, 16 };
					std::vector<std::vector<unsigned char>> scaled;
					std::vector<GLFWimage> imgs;
					if (iw <= 256 && ih <= 256)
						imgs.push_back({ iw, ih, px });
					else
						for (int s : kSizes)
						{
							scaled.emplace_back((size_t)s * s * 4);
							if (stbir_resize_uint8_srgb(px, iw, ih, 0, scaled.back().data(), s, s, 0, STBIR_RGBA))
								imgs.push_back({ s, s, scaled.back().data() });
						}
					if (!imgs.empty())
					{
						glfwSetWindowIcon(m_window, (int)imgs.size(), imgs.data());
						cout << "[NukeDiligent]\twindow icon: " << c.string() << " (" << iw << "x" << ih
						     << (imgs[0].pixels == px ? "" : ", downscaled") << ")" << endl;
						iconSet = true;
					}
					stbi_image_free(px);
					if (iconSet) break;
				}
				else
					cout << "[NukeDiligent]\twindow icon: " << c.string() << " failed to decode ("
					     << (stbi_failure_reason() ? stbi_failure_reason() : "?") << ")" << endl;
			}
		if (!iconSet)
			cout << "[NukeDiligent]\tno window icon found near " << base.string() << endl;
	}
#endif   // _WIN32 (icon/dark-titlebar are host-OS niceties; macOS titlebars follow the system theme)
	if (desc.opacity < 1.0f)
		glfwSetWindowOpacity(m_window, desc.opacity);
	if (desc.hideFromCapture)
		ApplyCaptureAffinity(m_window, true);
	if (const char* dv = std::getenv("NUKE_DEBUG_VIEW"))   // dev hook: boot straight into a debug view
		m_impl->debugView = atoi(dv);
	glfwShowWindow(m_window);

	m_impl->useD3D12  = (desc.backend == 1);
	m_impl->useVulkan = (desc.backend == 2);
#ifndef _WIN32
	// The D3D backends exist only on Windows — everything else runs Vulkan (macOS: MoltenVK).
	if (!m_impl->useVulkan)
		cout << "[NukeDiligent]\tbackend " << desc.backend << " is Windows-only — forcing Vulkan" << endl;
	m_impl->useD3D12  = false;
	m_impl->useVulkan = true;
#endif
#ifdef _WIN32
	// D3D only (vendored SwapChainD3DBase.hpp patch): set around PRIMARY creation only, so
	// secondary UI-viewport swap chains stay ordinary opaque HWND chains.
	g_NukeCompositionSwapChain = desc.transparent && !m_impl->useVulkan;
	Win32NativeWindow Window{ hWnd };
#elif defined(__APPLE__)
	MacOSNativeWindow Window{ NukeCocoaMetalView(m_window) };
#else
	// Wayland: wl_display + wl_surface. X11: the Xlib pair (pDisplay + WindowId, no XCB
	// connection). Runtime choice — see the display-backend ladder at glfwInit.
	LinuxNativeWindow Window{};
	if (NukeGlfwIsWayland())
	{
		Window.pDisplay        = NukeGlfwWaylandDisplay();
		Window.pWaylandSurface = NukeGlfwWaylandWindow(m_window);
	}
	else
	{
		Window.pDisplay = glfwGetX11Display();
		Window.WindowId = (Uint32)glfwGetX11Window(m_window);
	}
#endif
	SwapChainDesc SCDesc;
	// Must match the World PSO + offscreen RTs (Diligent would default the backbuffer to *_SRGB);
	// HDR10 output needs a 10-bit backbuffer for the PQ-encoded signal.
	SCDesc.ColorBufferFormat = m_impl->hdrOutput ? TEX_FORMAT_RGB10A2_UNORM : TEX_FORMAT_RGBA8_UNORM;
	// 3, not Diligent's default 2: Vulkan MAILBOX with 2 images blocks acquire until vblank.
	SCDesc.BufferCount = 3;
	// The DESIRED size must be explicit: a Wayland surface reports currentExtent as
	// "undefined" (0xFFFFFFFF), and with zero desired size Diligent passes that straight
	// into vkCreateSwapchainKHR — NVIDIA answers ERROR_OUT_OF_DEVICE_MEMORY. Everywhere
	// else a defined currentExtent simply overrides these, so this is Wayland-only in effect.
	{
		int fbw = 0, fbh = 0;
		glfwGetFramebufferSize(m_window, &fbw, &fbh);
		if (fbw > 0 && fbh > 0) { SCDesc.Width = (Uint32)fbw; SCDesc.Height = (Uint32)fbh; }
	}
	IEngineFactory* engFactory = nullptr;
#ifdef _WIN32
	if (m_impl->useD3D12)
	{
		auto* pFactory = GetEngineFactoryD3D12(); engFactory = pFactory;
		EngineD3D12CreateInfo EngineCI;
#ifdef _DEBUG
		// Opt-in even in Debug (the validation layer can halve the frame rate): NUKE_GPU_VALIDATION=1.
		const char* gpuValEnv = std::getenv("NUKE_GPU_VALIDATION");
		const bool  wantVal = desc.gpuValidation                                   // config/main.json (works for double-click)
		                   || (gpuValEnv && gpuValEnv[0] && gpuValEnv[0] != '0');   // or the env var (terminal launches)
		if (wantVal)
		{
			EngineCI.SetValidationLevel(VALIDATION_LEVEL_1);
			cout << "[NukeDiligent]\tD3D12 debug layer ENABLED (gpuValidation) — expect lower FPS" << endl;
		}
		// Separate opt-in (NUKE_DRED=1), never tied to gpuValidation: breadcrumb-instrumented DXR
		// dispatches wedge the queue / remove the device with ACCESS_DENIED on some drivers.
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
		// Editor-class descriptor budgets; the library defaults overflow on heavy frames.
		// D3D12 caps the SAMPLER heap at 2048 descriptors TOTAL — it can only be repartitioned, never grown.
		EngineCI.GPUDescriptorHeapDynamicSize[0] = 131072;   // CBV/SRV/UAV (default 8k)
		EngineCI.GPUDescriptorHeapSize[0]        = 32768;    // static/mutable CBV/SRV/UAV
		EngineCI.GPUDescriptorHeapSize[1]        = 512;      // static/mutable samplers
		EngineCI.GPUDescriptorHeapDynamicSize[1] = 1536;     // dynamic samplers (512+1536 = the 2048 cap)
		// Hull/domain stages (water tessellation); OPTIONAL — consumers check the feature and fall back.
		EngineCI.Features.Tessellation = DEVICE_FEATURE_STATE_OPTIONAL;
		pFactory->CreateDeviceAndContextsD3D12(EngineCI, &m_impl->device, &m_impl->context);
		if (!m_impl->device) { cout << "[NukeDiligent]\tD3D12 device creation failed" << endl; return 1; }
		pFactory->CreateSwapChainD3D12(m_impl->device, m_impl->context, SCDesc,
		                               FullScreenModeDesc{}, Window, &m_impl->swapChain);
	}
	else
#endif   // _WIN32
	if (m_impl->useVulkan)
	{
		// Vulkan WSI path: no DXGI anywhere; HLSL shaders compile to SPIR-V via the vendored glslang.
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
		// Editor-class dynamic budgets, mirroring the D3D12 branch.
		EngineCI.DynamicHeapSize = 32u << 20;
		EngineCI.Features.AsyncShaderCompilation = DEVICE_FEATURE_STATE_OPTIONAL;
		// Unlike D3D12, Vulkan device features must be opted into at device creation.
		EngineCI.Features.RayTracing = DEVICE_FEATURE_STATE_OPTIONAL;
		EngineCI.Features.Tessellation = DEVICE_FEATURE_STATE_OPTIONAL;
#ifdef _WIN32
		// RT shaders are SM6.x HLSL and need DXC; point Diligent at the one vendored
		// dxcompiler.dll (emits both DXIL and SPIR-V) instead of its "spv_dxcompiler.dll" default.
		EngineCI.pDxCompilerPath = "dxcompiler.dll";
#elif !defined(__APPLE__)
		// Same arrangement on Linux: vendored libdxcompiler.so (deps/dxc-linux, deployed next
		// to the hosts) — dlopen resolves the bare name through this module's $ORIGIN/..
		// runpath. Missing file → Diligent falls back to glslang (SM5, no RayQuery).
		EngineCI.pDxCompilerPath = "libdxcompiler.so";
#endif   // macOS: no DXC — SM5 HLSL compiles via glslang, RT off (MoltenVK reports no caps anyway)
		pFactory->CreateDeviceAndContextsVk(EngineCI, &m_impl->device, &m_impl->context);
		if (!m_impl->device) { cout << "[NukeDiligent]\tVulkan device creation failed" << endl; return 1; }
		// Transparent window on Vulkan: prefer an alpha-compositing swap chain (macOS: the
		// chosen mode drives CAMetalLayer.opaque). PRIMARY only — secondary UI chains stay
		// opaque, mirroring the DComp arrangement. HDR10: request an ST2084 surface format —
		// the Vulkan counterpart of the DXGI SetColorSpace1 path (macOS: PQ layer + EDR).
		g_NukeVkAlphaComposite = desc.transparent;
		g_NukeVkHDR10          = m_impl->hdrOutput;
		pFactory->CreateSwapChainVk(m_impl->device, m_impl->context, SCDesc, Window, &m_impl->swapChain);
#if !defined(_WIN32) && !defined(__APPLE__)
		// Wayland maps the surface ASYNCHRONOUSLY: until the compositor's configure lands,
		// a swapchain can't bind to it (NVIDIA reports the unconfigured state as
		// ERROR_OUT_OF_DEVICE_MEMORY). Pump events and retry — arrives within a few frames.
		if (!m_impl->swapChain && NukeGlfwIsWayland())
			for (int tries = 0; tries < 40 && !m_impl->swapChain; ++tries)
			{
				glfwWaitEventsTimeout(0.025);
				pFactory->CreateSwapChainVk(m_impl->device, m_impl->context, SCDesc, Window, &m_impl->swapChain);
			}
#endif
		g_NukeVkAlphaComposite = false;
		g_NukeVkHDR10          = false;
		if (m_impl->hdrOutput)
		{
			m_impl->hdr10Active = g_NukeVkHDR10Active;
			cout << "[NukeDiligent]\tHDR10 output "
			     << (m_impl->hdr10Active ? "ACTIVE (ST2084 swap chain)"
			                             : "off (surface offers no HDR10 format)") << endl;
		}
	}
#ifdef _WIN32
	else
	{
		auto* pFactory = GetEngineFactoryD3D11(); engFactory = pFactory;
		EngineD3D11CreateInfo EngineCI;
		pFactory->CreateDeviceAndContextsD3D11(EngineCI, &m_impl->device, &m_impl->context);
		if (!m_impl->device) { cout << "[NukeDiligent]\tD3D11 device creation failed" << endl; return 1; }
		pFactory->CreateSwapChainD3D11(m_impl->device, m_impl->context, SCDesc,
		                               FullScreenModeDesc{}, Window, &m_impl->swapChain);
	}
#endif
	if (!m_impl->swapChain) { cout << "[NukeDiligent]\tswap chain creation failed" << endl; return 1; }
#ifdef _WIN32
	g_NukeCompositionSwapChain = false;   // primary done — secondary swap chains stay opaque
#endif
	m_impl->transparent = desc.transparent;   // drives the alpha-0 clear + premultiplied final pass

	// Transparent window: the composition swap chain must be bound into a DComp visual on the
	// HWND — the swap chain alone does not compose.
	if (desc.transparent && m_impl->useVulkan)
		cout << "[NukeDiligent]\ttransparent window: alpha-composited Vulkan swap chain requested"
		        " (opaque fallback if the surface can't composite)" << endl;
#ifdef _WIN32
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
#endif   // _WIN32 (DirectComposition)
	m_impl->ShaderFactory();   // build the resolver over everything pushed so far
	// Ray tracing: D3D12 (DXR) or Vulkan (VK_KHR_ray_tracing) + a capable GPU/driver — and a
	// shader compiler that speaks SM6.x. Off Windows no DXC is vendored (yet): glslang cannot
	// even parse RayQuery HLSL (RT_ENABLED world.ps), so a Linux GPU with RT caps must still
	// take the raster path or every world PSO fails to build.
#ifdef _WIN32
	const bool rtCompilerOk = true;                  // vendored dxcompiler.dll (deps/dxc)
#elif defined(__APPLE__)
	const bool rtCompilerOk = false;                 // no DXC on macOS (MoltenVK has no RT caps anyway)
#else
	// Vendored libdxcompiler.so deployed next to the hosts (deps/dxc-linux) unlocks SM6.x.
	// PRESENT is not LOADABLE: the official DXC build wants glibc 2.38+ — on an older distro
	// dlopen fails inside Diligent and the RT shader path would poison every world PSO, so
	// probe the load for real (the handle refcounts; Diligent's own dlopen reuses it).
	boost::system::error_code rtec;
	const boost::filesystem::path dxcPath = nuke::Config::baseDir() / "libdxcompiler.so";
	bool rtCompilerOk = boost::filesystem::exists(dxcPath, rtec);
	if (!rtCompilerOk)
		cout << "[NukeDiligent]\tno libdxcompiler.so next to the host — ray tracing unavailable" << endl;
	else if (void* dxcProbe = dlopen(dxcPath.string().c_str(), RTLD_LAZY | RTLD_LOCAL))
		dlclose(dxcProbe);
	else
	{
		rtCompilerOk = false;
		const char* err = dlerror();
		cout << "[NukeDiligent]\tlibdxcompiler.so present but not loadable ("
		     << (err ? err : "?") << ") — ray tracing unavailable (raster path)" << endl;
	}
#endif
	m_impl->rtSupported = rtCompilerOk &&
	                      desc.rayTracing &&   // config kill switch: window.rayTracing=false forces the raster path
	                      (m_impl->useD3D12 || m_impl->useVulkan) && m_impl->device &&
	                      (m_impl->device->GetAdapterInfo().RayTracing.CapFlags & RAY_TRACING_CAP_FLAG_STANDALONE_SHADERS) != 0;
	cout << "[NukeDiligent]\tbackend=" << (m_impl->useD3D12 ? "D3D12" : m_impl->useVulkan ? "Vulkan" : "D3D11")
	     << " rayTracing=" << (m_impl->rtSupported ? "yes" : (desc.rayTracing ? "no" : "off (config)")) << endl;
	// Texture streaming: the config budget (0 = off) — live-adjustable via setTextureStreaming.
	m_impl->streamBudget = (long long)(desc.textureStreamMB < 0 ? 0 : desc.textureStreamMB) << 20;
	if (m_impl->streamBudget > 0)
		cout << "[NukeDiligent]\ttexture streaming: " << desc.textureStreamMB << " MB budget" << endl;
	// The RT fallback TLAS is built at the top of the first frame, not here: on Vulkan an
	// AS build before the frame loop deadlocks in the upload path (fence with no submission).

	if (m_impl->hdrOutput && !m_impl->useVulkan)
		m_impl->SetupHDROutput();   // HDR10 colour space via DXGI — D3D backends only for now

	m_impl->InitPSOCache();   // before the first pipeline: every creation goes through the cache
	m_impl->StorageInit();    // DirectStorage provider (D3D12): pak reads + VRAM-direct textures
	const SwapChainDesc& scd = m_impl->swapChain->GetDesc();
	m_impl->CreateUIPipeline(scd.ColorBufferFormat, scd.DepthBufferFormat);
	m_impl->CreateWorldPipeline();

	width  = w;
	height = h;

	cout << "[NukeDiligent]\tdevice=" << m_impl->device.RawPtr()
	     << " swapChain=" << m_impl->swapChain.RawPtr() << endl;

	// Launch straight into the requested display mode; the swap chain follows the framebuffer.
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
	// Dev hook: NUKE_FRAME_DEBUG=1 — CPU timings of the frame sections, printed every ~2s
	// (frame-pacing probes: which section eats the frame).
	static int s_frameDbg = -1;
	if (s_frameDbg < 0) { const char* e = std::getenv("NUKE_FRAME_DEBUG"); s_frameDbg = (e && *e == '1') ? 1 : 0; }
	using dbgclock = std::chrono::steady_clock;
	dbgclock::time_point dt0, dt1, dt2, dt3, dt4, dt5;
	auto dbgnow = [&]() { return s_frameDbg ? dbgclock::now() : dbgclock::time_point(); };
	dt0 = dbgnow();

	glfwPollEvents();
	if (s_frameDbg) dt1 = dbgclock::now();
#ifdef _DEBUG
	DrainD3D12DebugMessages(m_impl->device, m_impl->useD3D12);   // real validation errors -> console
#endif

	// A removed device can't run any of the frame below: mapping/creating on it only cascades
	// asserts. Suspend rendering entirely; events still pump.
	if (m_impl->DeviceRemoved())
	{
		static bool said = false;
		if (!said) { said = true; cout << "[NukeDiligent]\trendering SUSPENDED (device removed — see the report above)" << endl; }
		return 1;
	}

	// GPU lifetime: advance the frame clock, free trash no in-flight command list can reference.
	++m_impl->frameId;
	m_impl->PurgeTrash();
	m_impl->StreamPump();   // Texture streaming: residency step (budgeted rebuilds/evictions)
	// Queued secondary-swap-chain creations/resizes must run BEFORE anything is recorded.
	m_impl->ApplyPendingViewportOps();
	// RT fallback TLAS on the first frame (idempotent): building it at init deadlocks Vulkan's upload path.
	if (m_impl->rtSupported && !m_impl->fallbackTLAS) m_impl->EnsureRTFallback();

	m_impl->GpuFrame();   // resolve the GPU timings of an earlier frame, rotate the query ring

	// Latch the completed frame's counters for getFrameStats, start fresh.
	m_impl->statDrawsOut = m_impl->statDraws;
	m_impl->statTrisOut  = m_impl->statTris;
	m_impl->statDraws = 0;
	m_impl->statTris  = 0;
	++m_impl->occlFrame;

	// Follow the window: resize the swap chain when the framebuffer changes; skip when minimized.
	int fbw = 0, fbh = 0;
	glfwGetFramebufferSize(m_window, &fbw, &fbh);
	if (fbw <= 0 || fbh <= 0)
		return 1;
	if (fbw != width || fbh != height)
	{
		width  = fbw;
		height = fbh;
		// D3D12 removes the device if back buffers are still bound or referenced by in-flight work
		// when Resize() runs: unbind + flush + idle first.
		m_impl->context->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE);
		m_impl->context->Flush();
		m_impl->device->IdleGPU();
		m_impl->swapChain->Resize((Uint32)fbw, (Uint32)fbh);
	}

	// Deferred MSAA / HDR: apply BETWEEN frames only — rebuilding mid-frame frees RT textures the
	// UI's pending draw lists still reference. Both flip RTV formats, so one rebuild covers both.
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

	// 0) Clear the backbuffer up front. It must NOT be cleared again after onRender, or the
	//    Player's world-rendered-to-backbuffer would be wiped.
	ITextureView* pRTV = m_impl->swapChain->GetCurrentBackBufferRTV();
	ITextureView* pDSV = m_impl->swapChain->GetDepthBufferDSV();
	const float clearColor[] = { 0.10f, 0.12f, 0.16f, 1.0f };
	m_impl->context->SetRenderTargets(1, &pRTV, pDSV, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	m_impl->context->ClearRenderTarget(pRTV, clearColor, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	m_impl->context->ClearDepthStencil(pDSV, CLEAR_DEPTH_FLAG, 1.f, 0, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

	// 1) World passes: onRender drives World::Render (beginCamera + renderObject).
	if (s_frameDbg) dt2 = dbgclock::now();
	for (auto& cb : m_impl->onRender) cb();
	if (s_frameDbg) dt3 = dbgclock::now();

	// Reset debug/gizmo buffers HERE, not at frame start: lines emitted later (editor UI
	// overlays) must survive into the next frame's camera passes.
	{
		std::lock_guard<std::mutex> lock(m_impl->debugMutex);
		m_impl->debugVerts.clear();
		m_impl->debugVertsDepth.clear();
	}

	// 2) UI pass: rebind the backbuffer WITHOUT clearing, then draw the UI on top.
	m_impl->context->SetRenderTargets(1, &pRTV, pDSV, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	{
		Impl::GpuMark mk(m_impl, "ui");
		for (auto& cb : m_impl->onGUI) cb();
	}

	m_impl->DrawOverlayPass();  // fullscreen video: over the finished frame, under the cursor
	m_impl->DrawCursorPass();   // software cursor: topmost, over the finished UI
	Game::FlushScreenshot();    // queued Game.Screenshot: the presented image is complete here

	if (m_impl->DeviceRemoved()) return 1;   // device lost this frame: skip present, keep the app alive
	if (s_frameDbg) dt4 = dbgclock::now();
	m_impl->swapChain->Present(m_impl->vsync ? 1 : 0);   // SyncInterval 1 = vsync, 0 = uncapped
	if (s_frameDbg) dt5 = dbgclock::now();
	// Secondary (Vulkan native viewport) swapchains present AFTER the main chain: Present flushes,
	// and doing it mid-frame splits an RT write from its sampling and removes the device.
	for (void* h : m_impl->vpPresentQueue)
	{
		auto it = m_impl->uiVpSC.find(h);
		if (it == m_impl->uiVpSC.end() || !it->second) continue;
		it->second->Present(0);
	}
	m_impl->vpPresentQueue.clear();
	// D3D detached windows get their pixels via GDI from offscreen RTs (no secondary swap chains).
	dbgclock::time_point dp0 = dbgnow();
	m_impl->BlitHostWindows();
	dbgclock::time_point dp1 = dbgnow();
	m_impl->AdoptBuiltPipes();      // publish what the builder thread finished (pipes + jobs)
	dbgclock::time_point dp2 = dbgnow();
	m_impl->StoragePump();          // publish textures DirectStorage landed; issue prefetches
	dbgclock::time_point dp3 = dbgnow();
	m_impl->PumpPipelineWarmup();   // build a slice of the pending pipelines, off the draw path
	dbgclock::time_point dp4 = dbgnow();
	m_impl->PollShaderSaves();      // persist finished background compiles into the disk cache
	dbgclock::time_point dp5 = dbgnow();
	m_impl->SavePSOCache(false);    // ...and the driver pipeline blobs (throttled, when new ones appeared)
	if (s_frameDbg)
	{
		static dbgclock::time_point s_lastPrint;
		const dbgclock::time_point end = dbgclock::now();
		if (end - s_lastPrint > std::chrono::seconds(2))
		{
			s_lastPrint = end;
			auto ms = [](dbgclock::time_point a, dbgclock::time_point b)
			{ return std::chrono::duration<double, std::milli>(b - a).count(); };
			std::cout << "[NukeDiligent]\tframe ms: poll " << ms(dt0, dt1) << "  pre " << ms(dt1, dt2)
			          << "  world " << ms(dt2, dt3) << "  ui+3 " << ms(dt3, dt4)
			          << "  present " << ms(dt4, dt5) << "  post " << ms(dt5, end)
			          << " [blit " << ms(dp0, dp1) << " adopt " << ms(dp1, dp2) << " storage " << ms(dp2, dp3)
			          << " warmup " << ms(dp3, dp4) << " saves " << ms(dp4, dp5) << " psocache " << ms(dp5, end) << "]" << std::endl;
		}
	}
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

// Creates a shader through the on-disk bytecode cache: the key hashes every compile input, a
// hit creates the shader from bytecode, a miss compiles and queues a save. Every backend gets
// one — bytecode is portable across drivers, and a shipped game that recompiles its whole
// shader set on each launch stalls exactly where it must not.
void NukeDiligent::Impl::CreateShaderCached(const ShaderCreateInfo& ci, IShader** pp)
{
	if (!ci.Source)   // bytecode already in hand: nothing to cache
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
	const uint64_t epochNow = includeEpoch.load();
	h = fnv(h, &epochNow, sizeof(epochNow));   // include edits must invalidate too
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
	// Cache in the engine's WRITABLE root (run root on Windows/dev; the per-user dir for an
	// installed bundle — nothing may write beside or inside a deployed .app).
	bfs::path cacheRoot = nuke::Config::writableDir();
	// Per backend: SPIR-V and DXBC/DXIL are different products of the same source.
	const char* backendTag = useVulkan ? "vk" : (useD3D12 ? "d3d12" : "d3d11");
	const bfs::path dir = cacheRoot / "config" / (std::string("shadercache_") + backendTag);
	const bfs::path file = dir / (std::string(hex) + ".bin");

	boost::system::error_code ec;
	const double t0 = nuke::Log::Uptime();
	auto report = [&](const char* what)
	{
		const double ms = (nuke::Log::Uptime() - t0) * 1000.0;
		if (ms > 30.0)
			cout << "[NukeDiligent]\tshader '" << (ci.Desc.Name ? ci.Desc.Name : "?") << "' " << what << " " << (int)ms << " ms" << endl;
	};
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
			if (*pp) { report("cache load"); return; }   // corrupt/stale bytecode falls through to a fresh compile
			cout << "[NukeDiligent]\tshader cache entry rejected, recompiling (" << hex << ")" << endl;
		}
	}

	// Cache miss: compile in the background when supported; the PSO helper waits for readiness.
	ShaderCreateInfo cc = ci;
	if (device->GetDeviceInfo().Features.AsyncShaderCompilation)
		cc.CompileFlags |= SHADER_COMPILE_FLAG_ASYNCHRONOUS;
	device->CreateShader(cc, pp);
	if (*pp)
	{
		std::lock_guard<std::mutex> lock(shaderSaveMutex);
		pendingShaderSaves.emplace_back(RefCntAutoPtr<IShader>(*pp), file.string());
	}
	report((cc.CompileFlags & SHADER_COMPILE_FLAG_ASYNCHRONOUS) ? "COMPILE submit" : "COMPILE sync");
}

// Runs pending pipeline builders under a time budget. Called once per frame after the passes,
// so the sample count and scene format of the pass just drawn are the ones the pipelines are
// built for, and the next frame can already use them. A change of either retires every
// pipeline built against the old pair and re-arms the builders.
void NukeDiligent::Impl::PumpPipelineWarmup()
{
	if (warmups.empty()) return;
	const TEXTURE_FORMAT fmt = SceneFmt();
	if (samples != warmSamples || fmt != warmFmt)
	{
		warmSamples = samples; warmFmt = fmt;
		for (WarmEntry& e : warmups) e.done = false;
	}
	const double t0 = glfwGetTime();
	for (WarmEntry& e : warmups)
	{
		if (e.done || !e.fn) continue;
		e.done = e.fn(e.user);
		if ((glfwGetTime() - t0) * 1000.0 >= warmBudgetMs) break;   // the rest waits a frame
	}
}

// Persistent driver pipeline cache: config/psocache_<backend>.bin. D3D11 has no such object
// (the driver caches on its own); a stale/foreign blob is simply ignored by the driver.
static boost::filesystem::path PSOCacheFile(bool vk, bool d3d12)
{
	const char* tag = vk ? "vk" : (d3d12 ? "d3d12" : "d3d11");
	return boost::filesystem::path(nuke::Config::writableDir()) / "config" / (std::string("psocache_") + tag + ".bin");
}

void NukeDiligent::Impl::InitPSOCache()
{
	if (!device || (!useVulkan && !useD3D12)) return;
	namespace bfs = boost::filesystem;
	std::vector<char> bytes;
	boost::system::error_code ec;
	const bfs::path file = PSOCacheFile(useVulkan, useD3D12);
	if (bfs::exists(file, ec))
	{
		bfs::ifstream f(file, std::ios::binary);
		bytes.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
	}
	PipelineStateCacheCreateInfo ci;
	ci.Desc.Name = "Nuke PSO cache";
	ci.Desc.Mode = PSO_CACHE_MODE_LOAD_STORE;
	ci.pCacheData = bytes.empty() ? nullptr : bytes.data();
	ci.CacheDataSize = (Uint32)bytes.size();
	device->CreatePipelineStateCache(ci, &psoCache);
	cout << "[NukeDiligent]\tpipeline cache " << (psoCache ? "ready" : "UNAVAILABLE")
	     << " (" << bytes.size() / 1024 << " KB loaded)" << endl;
	psoCacheSavedAt = nuke::Log::Uptime();
}

void NukeDiligent::Impl::SavePSOCache(bool force)
{
	if (!psoCache || !psoCacheDirty) return;
	const double now = nuke::Log::Uptime();
	if (!force && now - psoCacheSavedAt < 5.0) return;   // batch: new pipelines keep arriving during warm-up
	RefCntAutoPtr<IDataBlob> blob;
	psoCache->GetData(&blob);
	psoCacheDirty = false; psoCacheSavedAt = now;
	if (!blob || !blob->GetSize()) return;
	namespace bfs = boost::filesystem;
	boost::system::error_code ec;
	const bfs::path file = PSOCacheFile(useVulkan, useD3D12);
	bfs::create_directories(file.parent_path(), ec);
	const bfs::path tmp = file.string() + ".tmp";
	{
		bfs::ofstream f(tmp, std::ios::binary | std::ios::trunc);
		if (!f) return;
		f.write((const char*)blob->GetConstDataPtr(), (std::streamsize)blob->GetSize());
	}
	bfs::rename(tmp, file, ec);   // atomic swap: a crash mid-write never leaves a torn cache
	if (ec) { bfs::remove(file, ec); bfs::rename(tmp, file, ec); }
	cout << "[NukeDiligent]\tpipeline cache saved (" << blob->GetSize() / 1024 << " KB)" << endl;
}

// Writes finished cache-miss compiles to disk; called once per frame, since an async
// shader's bytecode only exists after its worker finishes.
void NukeDiligent::Impl::PollShaderSaves()
{
	std::lock_guard<std::mutex> lock(shaderSaveMutex);
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
	m_impl->StopPipeBuilder();    // join: no device call may still run on the builder thread
	m_impl->retiredShaderFactories.clear();   // nothing compiles any more; the graveyard may go
	m_impl->StorageShutdown();    // every DirectStorage request lands before its destinations die
	m_impl->SavePSOCache(true);   // pipelines built this session -> next start creates them warm
	// Drain the GPU trash AFTER the queue settles — parked objects must not outlive the device.
	if (m_impl->context && m_impl->device)
	{
		m_impl->context->Flush();
		m_impl->device->IdleGPU();
	}
	m_impl->PurgeTrash(true);
#ifdef _WIN32
	// DComp release order: visual -> target -> device, all before the swap chain they reference.
	if (m_impl->dcompVisual) { m_impl->dcompVisual->Release(); m_impl->dcompVisual = nullptr; }
	if (m_impl->dcompTarget) { m_impl->dcompTarget->Release(); m_impl->dcompTarget = nullptr; }
	if (m_impl->dcompDevice) { m_impl->dcompDevice->Release(); m_impl->dcompDevice = nullptr; }
#endif
	m_impl->swapChain.Release();
	m_impl->context.Release();
	m_impl->device.Release();
	if (m_window) { glfwDestroyWindow(m_window); m_window = nullptr; }
	glfwTerminate();
}

void NukeDiligent::update() {}

// Renderer + active backend name (editor status bar).
char* NukeDiligent::getEngine()
{
	if (m_impl->useVulkan) return (char*)"Diligent - Vulkan";
	if (m_impl->useD3D12)  return (char*)"Diligent - D3D12";
	return (char*)"Diligent - D3D11";
}
char* NukeDiligent::getVersion() { return (char*)"0.1.0"; }

void NukeDiligent::setOnGUI(bst::function<void(void)> cb)    { m_impl->onGUI.push_back(cb); }
void NukeDiligent::setOnRender(bst::function<void(void)> cb) { m_impl->onRender.push_back(cb); }
void NukeDiligent::setOnClose(bst::function<void()> cb)      { m_impl->onClose.push_back(cb); }

// Input is routed via iRender callbacks in a later milestone.
void NukeDiligent::keyboard(int, int, int, int) {}
void NukeDiligent::mouseMove(double, double) {}
void NukeDiligent::mouseClick(int, int, int) {}
void NukeDiligent::rawMouse(double, double) {}
void NukeDiligent::mouseEnterLeave(int) {}
void NukeDiligent::setWindowTitle(const char* title) { if (m_window && title) glfwSetWindowTitle(m_window, title); }
bool NukeDiligent::isWindowFocused() { return m_window && glfwGetWindowAttrib(m_window, GLFW_FOCUSED) != 0; }
bool NukeDiligent::isWindowMaximized() { return m_window && glfwGetWindowAttrib(m_window, GLFW_MAXIMIZED) != 0; }
void NukeDiligent::setWindowMaximized(bool m) { if (!m_window) return; if (m) glfwMaximizeWindow(m_window); else glfwRestoreWindow(m_window); }

// Applies a runtime window change (mode/size/decoration/opacity). Only the WINDOW is driven —
// the swap chain follows the new framebuffer size in the render loop.
void NukeDiligent::applyWindow(const WindowDesc& d)
{
	if (!m_window) return;
	GLFWmonitor* mon = glfwGetPrimaryMonitor();
	const GLFWvidmode* vm = mon ? glfwGetVideoMode(mon) : nullptr;

	// Leaving windowed: remember the rect so a later return restores it.
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
		// Undecorated window over the monitor at desktop resolution (no mode switch).
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
	glfwSetWindowAttrib(m_window, GLFW_FLOATING, d.floating ? GLFW_TRUE : GLFW_FALSE);
#ifdef GLFW_MOUSE_PASSTHROUGH
	glfwSetWindowAttrib(m_window, GLFW_MOUSE_PASSTHROUGH, d.clickThrough ? GLFW_TRUE : GLFW_FALSE);
#else
	if (d.clickThrough)
		cout << "[NukeDiligent]\tclickThrough needs GLFW 3.4+ — ignored" << endl;
#endif
	ApplyCaptureAffinity(m_window, d.hideFromCapture);
	m_windowMode = d.mode;
	if (d.transparent)
		cout << "[NukeDiligent]\tper-pixel transparency is a creation-time property — applies on next launch" << endl;
	cout << "[NukeDiligent]\tapplyWindow mode=" << d.mode << " " << d.w << "x" << d.h
	     << " decorated=" << d.decorated << " opacity=" << d.opacity
	     << " floating=" << d.floating << " clickThrough=" << d.clickThrough
	     << " hideFromCapture=" << d.hideFromCapture << endl;
}
void NukeDiligent::getCursorPos(double& x, double& y) { x = y = 0; if (m_window) glfwGetCursorPos(m_window, &x, &y); }

void NukeDiligent::setCursorMode(int mode)
{
	if (!m_window) return;
	int glfwMode = GLFW_CURSOR_NORMAL;
	switch (mode)
	{
		case 1: glfwMode = GLFW_CURSOR_HIDDEN;   break;   // invisible, free
		case 2: glfwMode = GLFW_CURSOR_DISABLED; break;   // invisible + centered, raw deltas
#ifdef GLFW_CURSOR_CAPTURED
		case 3: glfwMode = GLFW_CURSOR_CAPTURED; break;   // visible, clamped to the window
#endif
		default: break;
	}
	glfwSetInputMode(m_window, GLFW_CURSOR, glfwMode);
	// Locked mode: raw motion drops the OS cursor acceleration from the deltas.
	if (glfwRawMouseMotionSupported())
		glfwSetInputMode(m_window, GLFW_RAW_MOUSE_MOTION, mode == 2 ? GLFW_TRUE : GLFW_FALSE);
	m_cursorMode = mode;
}
int NukeDiligent::getCursorMode() { return m_cursorMode; }

void NukeDiligent::setDebugView(int mode)
{
	m_impl->debugView = mode;
}

void NukeDiligent::setScreenOverlay(Texture* tex)
{
	if (m_impl->overlayTex != tex) { m_impl->overlaySRB.Release(); m_impl->overlayLastSRV = nullptr; }
	m_impl->overlayTex = tex;
}

// Custom cursor (see irender.h): hardware = cached GLFW cursor per id; software = cached
// texture per id, drawn by DrawCursorPass. Does not override setCursorMode's capture/hide.
bool NukeDiligent::setCursorImage(uint64_t id, const unsigned char* rgba, int w, int h,
                                  int hotX, int hotY, int mode)
{
	if (!m_window) return false;
	m_impl->cursorWindow = m_window;
	if (mode == 0 || !rgba || w <= 0 || h <= 0)
	{
		glfwSetCursor(m_window, nullptr);
		if (m_impl->curCursorMode == 2 && m_cursorMode == 0)
			glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);   // undo the software hide
		m_impl->curCursorId = 0;
		m_impl->curCursorMode = 0;
		return true;
	}
	if (mode == 1)
	{
		GLFWcursor*& cur = m_impl->hwCursors[id];
		if (!cur)
		{
			GLFWimage img{ w, h, const_cast<unsigned char*>(rgba) };   // GLFW copies the pixels
			cur = glfwCreateCursor(&img, hotX, hotY);
			if (!cur) { m_impl->hwCursors.erase(id); return false; }
		}
		if (m_impl->curCursorMode == 2 && m_cursorMode == 0)
			glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		glfwSetCursor(m_window, cur);
		m_impl->curCursorId = id;
		m_impl->curCursorMode = 1;
		return true;
	}
	// mode 2: software — upload the frame once, hide the OS cursor, DrawCursorPass paints it.
	auto& sw = m_impl->swCursors[id];
	if (!sw.tex)
	{
		TextureDesc td;
		td.Name      = "cursor frame";
		td.Type      = RESOURCE_DIM_TEX_2D;
		td.Width     = (Uint32)w;
		td.Height    = (Uint32)h;
		td.Format    = TEX_FORMAT_RGBA8_UNORM;
		td.BindFlags = BIND_SHADER_RESOURCE;
		TextureSubResData sub{ rgba, (Uint64)w * 4 };
		TextureData data{ &sub, 1 };
		m_impl->device->CreateTexture(td, &data, &sw.tex);
		if (!sw.tex) { m_impl->swCursors.erase(id); return false; }
		sw.w = w; sw.h = h;
	}
	sw.hotX = hotX; sw.hotY = hotY;
	if (m_cursorMode == 0)
		glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
	glfwSetCursor(m_window, nullptr);
	m_impl->curCursorId = id;
	m_impl->curCursorMode = 2;
	return true;
}
bool NukeDiligent::isMouseButtonDown(int b) { return m_window && glfwGetMouseButton(m_window, b) == GLFW_PRESS; }

// Desktop file-drop -> editor import (one renderer instance, so a file-static callback is fine).
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
