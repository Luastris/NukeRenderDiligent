#include "NukeDiligentImpl.h"


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

	m_impl->useD3D12 = (desc.backend == 1);
	Win32NativeWindow Window{ hWnd };
	SwapChainDesc SCDesc;
	// Match the World PSO + offscreen render targets (RGBA8_UNORM). Diligent defaults the
	// backbuffer to *_SRGB, which mismatches the world PSO when rendering straight to the
	// window (the Player) — "RTV format does not match PSO" spam. Keep one format everywhere.
	// HDR10 display output (Player): a 10-bit backbuffer carries the PQ-encoded HDR signal. Plain RGBA8 otherwise.
	SCDesc.ColorBufferFormat = m_impl->hdrOutput ? TEX_FORMAT_RGB10A2_UNORM : TEX_FORMAT_RGBA8_UNORM;
	if (m_impl->useD3D12)
	{
		auto* pFactory = GetEngineFactoryD3D12();
		EngineD3D12CreateInfo EngineCI;
		pFactory->CreateDeviceAndContextsD3D12(EngineCI, &m_impl->device, &m_impl->context);
		if (!m_impl->device) { cout << "[NukeDiligent]\tD3D12 device creation failed" << endl; return 1; }
		pFactory->CreateSwapChainD3D12(m_impl->device, m_impl->context, SCDesc,
		                               FullScreenModeDesc{}, Window, &m_impl->swapChain);
	}
	else
	{
		auto* pFactory = GetEngineFactoryD3D11();
		EngineD3D11CreateInfo EngineCI;
		pFactory->CreateDeviceAndContextsD3D11(EngineCI, &m_impl->device, &m_impl->context);
		if (!m_impl->device) { cout << "[NukeDiligent]\tD3D11 device creation failed" << endl; return 1; }
		pFactory->CreateSwapChainD3D11(m_impl->device, m_impl->context, SCDesc,
		                               FullScreenModeDesc{}, Window, &m_impl->swapChain);
	}
	if (!m_impl->swapChain) { cout << "[NukeDiligent]\tswap chain creation failed" << endl; return 1; }
	// Ray tracing needs the D3D12 backend AND a capable GPU/driver (RTX / DXR1.1).
	m_impl->rtSupported = m_impl->useD3D12 && m_impl->device &&
	                      (m_impl->device->GetAdapterInfo().RayTracing.CapFlags & RAY_TRACING_CAP_FLAG_STANDALONE_SHADERS) != 0;
	cout << "[NukeDiligent]\tbackend=" << (m_impl->useD3D12 ? "D3D12" : "D3D11")
	     << " rayTracing=" << (m_impl->rtSupported ? "yes" : "no") << endl;

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

	m_impl->swapChain->Present();
	return 1;
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
