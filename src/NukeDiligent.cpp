// IMPORTANT include order: Windows-pulling headers (GLFW native + Diligent D3D)
// MUST come before the engine headers. Several engine headers do a global
// `using namespace std;`, which brings std::byte into scope; if the Windows SDK
// headers (objidl.h/oaidl.h, pulled by Diligent's D3D backend) are processed
// after that, `byte` becomes ambiguous (std::byte vs Windows ::byte). Processing
// the Windows headers first avoids the clash.

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
		RefCntAutoPtr<ITexture> color, depth;
		ITextureView* rtv = nullptr;
		ITextureView* dsv = nullptr;
		ITextureView* srv = nullptr;
		int w = 0, h = 0;
	};
	std::unordered_map<uint64_t, RT> rts;
	uint64_t rtCounter = 0;

	// --- 3D world pipelines (one per shader; all share the layout + CBs + white fallback) ---
	struct WorldPipe
	{
		RefCntAutoPtr<IPipelineState>         pso;
		RefCntAutoPtr<IShaderResourceBinding> srb;
		IShaderResourceVariable*              texVar = nullptr;   // PS "g_Tex" (dynamic)
	};
	std::unordered_map<uint64_t, WorldPipe> worldPipes;   // shader handle -> pipeline
	uint64_t                              defaultWorldHandle = 0;   // builtin "world" pipeline
	uint64_t                              nextShaderHandle   = 1;   // handles handed to the engine
	RefCntAutoPtr<IBuffer>                worldCB;     // VS: WVP + World   (shared)
	RefCntAutoPtr<IBuffer>                worldMatCB;  // PS: color + params + custom shader props (shared)
	static const uint32_t                 kMatCBBytes = 256;   // MatCB capacity (color/params + props)
	RefCntAutoPtr<ITexture>               whiteTex;    // 1x1 fallback when a material has no texture
	// Build a world-type PSO (fixed layout/CBs) from VS+PS source; store it under a handle.
	uint64_t MakeWorldPSO(const std::string& vsSrc, const std::string& psSrc, const char* dbg);
	struct MeshGPU { RefCntAutoPtr<IBuffer> pos, nrm, uv; int numVerts = 0; };
	std::unordered_map<Mesh*, MeshGPU>          meshCache;
	std::unordered_map<Texture*, RefCntAutoPtr<ITexture>> texCache;   // engine Texture -> GPU texture
	float4x4 curView, curProj;   // set in beginCamera, used in renderObject

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
	GP.DSVFormat         = dsFmt;
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

	// 1x1 white fallback texture (bound when a material has no texture).
	uint32_t white = 0xFFFFFFFFu;
	TextureDesc wd; wd.Type = RESOURCE_DIM_TEX_2D; wd.Width = 1; wd.Height = 1;
	wd.Format = TEX_FORMAT_RGBA8_UNORM; wd.BindFlags = BIND_SHADER_RESOURCE; wd.Usage = USAGE_IMMUTABLE;
	TextureSubResData wsr; wsr.pData = &white; wsr.Stride = 4;
	TextureData wdat; wdat.pSubResources = &wsr; wdat.NumSubresources = 1;
	device->CreateTexture(wd, &wdat, &whiteTex);

	// Built-in "world" pipeline from the engine shaders.
	defaultWorldHandle = MakeWorldPSO(shaderSource("world.vs"), shaderSource("world.ps"), "World");
}

uint64_t NukeDiligent::Impl::MakeWorldPSO(const std::string& vsSrc, const std::string& psSrc, const char* dbg)
{
	if (vsSrc.empty() || psSrc.empty()) return 0;
	ShaderCreateInfo sci;
	sci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
	RefCntAutoPtr<IShader> vs, ps;
	sci.Desc = {dbg, SHADER_TYPE_VERTEX, true}; sci.Source = vsSrc.c_str(); device->CreateShader(sci, &vs);
	sci.Desc = {dbg, SHADER_TYPE_PIXEL, true};  sci.Source = psSrc.c_str(); device->CreateShader(sci, &ps);
	if (!vs || !ps) return 0;

	GraphicsPipelineStateCreateInfo ci;
	ci.PSODesc.Name = dbg;
	auto& gp = ci.GraphicsPipeline;
	gp.NumRenderTargets             = 1;
	gp.RTVFormats[0]                = TEX_FORMAT_RGBA8_UNORM;
	gp.DSVFormat                    = TEX_FORMAT_D32_FLOAT;
	gp.PrimitiveTopology            = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	gp.RasterizerDesc.CullMode      = CULL_MODE_NONE;
	gp.DepthStencilDesc.DepthEnable = True;
	// Per-pixel transparency: straight-alpha blend (PS outputs alpha = material.a * texture.a).
	auto& rt0 = gp.BlendDesc.RenderTargets[0];
	rt0.BlendEnable    = True;
	rt0.SrcBlend       = BLEND_FACTOR_SRC_ALPHA;
	rt0.DestBlend      = BLEND_FACTOR_INV_SRC_ALPHA;
	rt0.BlendOp        = BLEND_OPERATION_ADD;
	rt0.SrcBlendAlpha  = BLEND_FACTOR_ONE;
	rt0.DestBlendAlpha = BLEND_FACTOR_INV_SRC_ALPHA;
	rt0.BlendOpAlpha   = BLEND_OPERATION_ADD;
	LayoutElement layout[] = {
		{0, 0, 3, VT_FLOAT32}, // position
		{1, 1, 3, VT_FLOAT32}, // normal
		{2, 2, 2, VT_FLOAT32}, // uv
	};
	gp.InputLayout.NumElements    = 3;
	gp.InputLayout.LayoutElements = layout;

	ShaderResourceVariableDesc vars[] = {{SHADER_TYPE_PIXEL, "g_Tex", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC}};
	ci.PSODesc.ResourceLayout.Variables    = vars;
	ci.PSODesc.ResourceLayout.NumVariables = 1;
	SamplerDesc samp; samp.MinFilter = FILTER_TYPE_LINEAR; samp.MagFilter = FILTER_TYPE_LINEAR; samp.MipFilter = FILTER_TYPE_LINEAR;
	samp.AddressU = TEXTURE_ADDRESS_WRAP; samp.AddressV = TEXTURE_ADDRESS_WRAP; samp.AddressW = TEXTURE_ADDRESS_WRAP;
	ImmutableSamplerDesc immSamp[] = {{SHADER_TYPE_PIXEL, "g_Tex", samp}};
	ci.PSODesc.ResourceLayout.ImmutableSamplers    = immSamp;
	ci.PSODesc.ResourceLayout.NumImmutableSamplers = 1;
	ci.pVS = vs; ci.pPS = ps;

	WorldPipe wp;
	device->CreateGraphicsPipelineState(ci, &wp.pso);
	if (!wp.pso) { cout << "[NukeDiligent]\tPSO build failed for shader '" << dbg << "'" << endl; return 0; }
	if (auto* v = wp.pso->GetStaticVariableByName(SHADER_TYPE_VERTEX, "CB"))    v->Set(worldCB);
	if (auto* m = wp.pso->GetStaticVariableByName(SHADER_TYPE_PIXEL,  "MatCB")) m->Set(worldMatCB);
	wp.pso->CreateShaderResourceBinding(&wp.srb, true);
	wp.texVar = wp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Tex");

	uint64_t h = nextShaderHandle++;
	worldPipes[h] = std::move(wp);
	return h;
}

ITextureView* NukeDiligent::Impl::GetTexSRV(Texture* t)
{
	if (!t || t->pixels.empty() || t->width <= 0 || t->height <= 0) return nullptr;
	auto it = texCache.find(t);
	if (it == texCache.end())
	{
		TextureDesc td; td.Type = RESOURCE_DIM_TEX_2D; td.Width = t->width; td.Height = t->height;
		td.Format = TEX_FORMAT_RGBA8_UNORM; td.BindFlags = BIND_SHADER_RESOURCE; td.Usage = USAGE_IMMUTABLE;
		TextureSubResData sub; sub.pData = t->pixels.data(); sub.Stride = (Uint64)t->width * 4;
		TextureData data; data.pSubResources = &sub; data.NumSubresources = 1;
		RefCntAutoPtr<ITexture> tex;
		device->CreateTexture(td, &data, &tex);
		it = texCache.emplace(t, tex).first;
	}
	return it->second ? it->second->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE) : nullptr;
}

NukeDiligent::Impl::RT NukeDiligent::Impl::MakeRT(int w, int h)
{
	RT rt; rt.w = w; rt.h = h;
	TextureDesc cd;
	cd.Name = "RT Color"; cd.Type = RESOURCE_DIM_TEX_2D; cd.Width = (Uint32)w; cd.Height = (Uint32)h;
	cd.Format = TEX_FORMAT_RGBA8_UNORM; cd.BindFlags = BIND_RENDER_TARGET | BIND_SHADER_RESOURCE;
	device->CreateTexture(cd, nullptr, &rt.color);
	if (rt.color)
	{
		rt.rtv = rt.color->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET);
		rt.srv = rt.color->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
	}
	TextureDesc dd;
	dd.Name = "RT Depth"; dd.Type = RESOURCE_DIM_TEX_2D; dd.Width = (Uint32)w; dd.Height = (Uint32)h;
	dd.Format = TEX_FORMAT_D32_FLOAT; dd.BindFlags = BIND_DEPTH_STENCIL;
	device->CreateTexture(dd, nullptr, &rt.depth);
	if (rt.depth)
		rt.dsv = rt.depth->GetDefaultView(TEXTURE_VIEW_DEPTH_STENCIL);
	return rt;
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
	SCDesc.ColorBufferFormat = TEX_FORMAT_RGBA8_UNORM;
	pFactory->CreateSwapChainD3D11(m_impl->device, m_impl->context, SCDesc,
	                               FullScreenModeDesc{}, Window, &m_impl->swapChain);
	if (!m_impl->swapChain) { cout << "[NukeDiligent]\tswap chain creation failed" << endl; return 1; }

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

void NukeDiligent::renderObject(Mesh* mesh, Material* mat,
                                const float pos[3], const float quat[4], const float scale[3])
{
	if (!mesh || mesh->numVerts <= 0 || m_impl->worldPipes.empty()) return;
	if (!mesh->vertexArray || !mesh->normalArray)   // immutable buffers need init data
	{
		std::cout << "[NukeDiligent]\tmesh '" << mesh->name << "' has null vertex/normal data (numVerts="
		          << mesh->numVerts << ") — skipping" << std::endl;
		return;
	}

	// Build (and cache) GPU vertex buffers for this mesh: positions + normals + uv.
	auto it = m_impl->meshCache.find(mesh);
	if (it == m_impl->meshCache.end())
	{
		Impl::MeshGPU g; g.numVerts = mesh->numVerts;
		const Uint64 sz3 = (Uint64)mesh->numVerts * 3 * sizeof(float);
		const Uint64 sz2 = (Uint64)mesh->numVerts * 2 * sizeof(float);
		BufferDesc bd; bd.BindFlags = BIND_VERTEX_BUFFER; bd.Usage = USAGE_IMMUTABLE;
		bd.Size = sz3; bd.Name = "mesh pos"; BufferData pdat{mesh->vertexArray, sz3}; m_impl->device->CreateBuffer(bd, &pdat, &g.pos);
		bd.Size = sz3; bd.Name = "mesh nrm"; BufferData ndat{mesh->normalArray, sz3}; m_impl->device->CreateBuffer(bd, &ndat, &g.nrm);
		std::vector<float> zeroUV;
		const float* uvSrc = mesh->uvArray;
		if (!uvSrc) { zeroUV.assign((size_t)mesh->numVerts * 2, 0.0f); uvSrc = zeroUV.data(); }   // mesh has no UVs
		bd.Size = sz2; bd.Name = "mesh uv"; BufferData udat{uvSrc, sz2}; m_impl->device->CreateBuffer(bd, &udat, &g.uv);
		it = m_impl->meshCache.emplace(mesh, std::move(g)).first;
	}
	Impl::MeshGPU& g = it->second;
	if (!g.pos || !g.nrm || !g.uv) return;

	float4x4 world = float4x4::Scale(scale[0], scale[1], scale[2])
	               * Diligent::Quaternion<float>(quat[0], quat[1], quat[2], quat[3]).ToMatrix()
	               * float4x4::Translation(pos[0], pos[1], pos[2]);
	float4x4 wvp = world * m_impl->curView * m_impl->curProj;

	struct CBData { float4x4 wvp; float4x4 world; };
	{
		MapHelper<CBData> cb(m_impl->context, m_impl->worldCB, MAP_WRITE, MAP_FLAG_DISCARD);
		cb->wvp = wvp; cb->world = world;
	}

	// Material: base color + diffuse texture (white fallback when none).
	float col[4] = { 1, 1, 1, 1 };
	ITextureView* srv = nullptr;
	if (mat)
	{
		for (int i = 0; i < 4; ++i) col[i] = mat->color[i];
		if (mat->diff) srv = m_impl->GetTexSRV(mat->diff);
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
		float prm[4] = { srv ? 1.0f : 0.0f, 0, 0, 0 };
		memcpy(p + 16, prm, sizeof(float) * 4);
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

	IDeviceContext* ctx = m_impl->context;
	IBuffer* vbs[]    = { g.pos, g.nrm, g.uv };
	Uint64   offs[]   = { 0, 0, 0 };
	ctx->SetVertexBuffers(0, 3, vbs, offs, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, SET_VERTEX_BUFFERS_FLAG_RESET);
	ctx->SetPipelineState(wp.pso);
	ctx->CommitShaderResources(wp.srb, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	DrawAttribs da{(Uint32)g.numVerts, DRAW_FLAG_VERIFY_STATES};
	ctx->Draw(da);
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
	ITextureView* rtv = nullptr;
	ITextureView* dsv = nullptr;
	int w = 0, h = 0;
	if (cam.target == 0)
	{
		rtv = m_impl->swapChain->GetCurrentBackBufferRTV();
		dsv = m_impl->swapChain->GetDepthBufferDSV();
		w = (int)m_impl->swapChain->GetDesc().Width;
		h = (int)m_impl->swapChain->GetDesc().Height;
	}
	else
	{
		auto it = m_impl->rts.find(cam.target);
		if (it == m_impl->rts.end()) return;
		rtv = it->second.rtv; dsv = it->second.dsv; w = it->second.w; h = it->second.h;
	}
	if (!rtv) return;

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
}

void NukeDiligent::endCamera() {}

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

	const Uint32 surfW = m_impl->swapChain->GetDesc().Width;
	const Uint32 surfH = m_impl->swapChain->GetDesc().Height;

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
