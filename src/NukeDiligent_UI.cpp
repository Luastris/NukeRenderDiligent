#include "NukeDiligentImpl.h"


// Get-or-create the cached SRB for a UI texture view. The texture is a MUTABLE variable set
// once at creation, so committing it allocates no dynamic GPU descriptors.
IShaderResourceBinding* NukeDiligent::Impl::UISRBFor(ITextureView* view)
{
	UISRBEntry& e = uiSRBCache[view];
	if (!e.srb)
	{
		uiPSO->CreateShaderResourceBinding(&e.srb, true);
		if (!e.srb) { uiSRBCache.erase(view); return nullptr; }
		if (IShaderResourceVariable* v = e.srb->GetVariableByName(SHADER_TYPE_PIXEL, "Texture"))
			v->Set(view);
	}
	e.lastUse = uiFrame;
	return e.srb;
}

// Upload the UI draw lists and draw them into the given render target view.
void NukeDiligent::Impl::DrawUILists(ITextureView* uirtv, Uint32 surfW, Uint32 surfH, const NukeUIDrawData& data)
{
	if (!uiPSO || !uirtv || data.listCount == 0) return;
	if (data.dispSize[0] <= 0.f || data.dispSize[1] <= 0.f) return;

	int totalVtx = 0, totalIdx = 0;
	for (int i = 0; i < data.listCount; ++i) { totalVtx += data.lists[i].vtxCount; totalIdx += data.lists[i].idxCount; }
	if (totalVtx == 0 || totalIdx == 0) return;

	IRenderDevice*  dev = device;
	IDeviceContext* ctx = context;

	if (!uiVB || uiVBSize < totalVtx)
	{
		Trash(uiVB);   // may grow mid-frame; never release inline
		uiVB.Release();
		while (uiVBSize < totalVtx) uiVBSize = uiVBSize ? uiVBSize * 2 : 4096;
		BufferDesc bd;
		bd.Name = "UI VB"; bd.BindFlags = BIND_VERTEX_BUFFER; bd.Usage = USAGE_DYNAMIC; bd.CPUAccessFlags = CPU_ACCESS_WRITE;
		bd.Size = (Uint64)uiVBSize * sizeof(NukeUIVert);
		dev->CreateBuffer(bd, nullptr, &uiVB);
	}
	if (!uiIB || uiIBSize < totalIdx)
	{
		Trash(uiIB);
		uiIB.Release();
		while (uiIBSize < totalIdx) uiIBSize = uiIBSize ? uiIBSize * 2 : 8192;
		BufferDesc bd;
		bd.Name = "UI IB"; bd.BindFlags = BIND_INDEX_BUFFER; bd.Usage = USAGE_DYNAMIC; bd.CPUAccessFlags = CPU_ACCESS_WRITE;
		bd.Size = (Uint64)uiIBSize * sizeof(uint16_t);
		dev->CreateBuffer(bd, nullptr, &uiIB);
	}

	{
		MapHelper<NukeUIVert> vtx(ctx, uiVB, MAP_WRITE, MAP_FLAG_DISCARD);
		MapHelper<uint16_t>   idx(ctx, uiIB, MAP_WRITE, MAP_FLAG_DISCARD);
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
		MapHelper<float4x4> cb(ctx, uiCB, MAP_WRITE, MAP_FLAG_DISCARD);
		if (cb) *cb = proj;
	}

	ctx->SetRenderTargets(1, &uirtv, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

	IBuffer* vbs[] = {uiVB};
	ctx->SetVertexBuffers(0, 1, vbs, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, SET_VERTEX_BUFFERS_FLAG_RESET);
	ctx->SetIndexBuffer(uiIB, 0, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	ctx->SetPipelineState(uiPSO);
	const float bf[4] = {0, 0, 0, 0};
	ctx->SetBlendFactors(bf);
	Viewport vp; vp.TopLeftX = 0; vp.TopLeftY = 0; vp.Width = (float)surfW; vp.Height = (float)surfH; vp.MinDepth = 0; vp.MaxDepth = 1;
	ctx->SetViewports(1, &vp, surfW, surfH);

	++uiFrame;
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
			if (!view) continue;   // no texture (failed/pending upload): drop the cmd, not the app
			if (view != lastView)
			{
				lastView = view;
				IShaderResourceBinding* srb = UISRBFor(view);
				if (!srb) continue;
				ctx->CommitShaderResources(srb, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
			}

			DrawIndexedAttribs da{cmd.elemCount, VT_UINT16, DRAW_FLAG_VERIFY_STATES};
			da.FirstIndexLocation = cmd.idxOffset + globalIdx;
			if (baseVertexSupported)
				da.BaseVertex = cmd.vtxOffset + globalVtx;
			ctx->DrawIndexed(da);
		}
		globalIdx += l.idxCount;
		globalVtx += l.vtxCount;
	}

	// LRU purge: an SRB's strong ref would otherwise keep stale views of resized RTs alive.
	if ((uiFrame & 511) == 0)
		for (auto it = uiSRBCache.begin(); it != uiSRBCache.end(); )
		{
			if (uiFrame - it->second.lastUse > 512) { Trash(it->second.srb); it = uiSRBCache.erase(it); }
			else ++it;
		}
}

void NukeDiligent::renderDrawLists(const NukeUIDrawData& data)
{
	ITextureView* uirtv = m_impl->uiRTV ? m_impl->uiRTV : m_impl->swapChain->GetCurrentBackBufferRTV();
	const Uint32 surfW = (m_impl->uiRTV && m_impl->uiTW) ? m_impl->uiTW : m_impl->swapChain->GetDesc().Width;
	const Uint32 surfH = (m_impl->uiRTV && m_impl->uiTH) ? m_impl->uiTH : m_impl->swapChain->GetDesc().Height;
	m_impl->DrawUILists(uirtv, surfW, surfH, data);
}

// --- UI multi-viewport: one swap chain per detached OS window ----------------------

void* NukeDiligent::nativeWindow()
{
	return m_window;
}

// Create queued secondary swap chains and resize mismatched ones. MUST run at the top of
// render(), before anything is recorded — never mid-frame.
void NukeDiligent::Impl::ApplyPendingViewportOps()
{
	++uiVpFrameNo;   // multi-window interleave clock; ticks every frame
	if (uiVpPending.empty() || !device) return;
#if !defined(_WIN32) && !defined(__APPLE__)
	// TODO(Deuterium/linux): X11/Wayland handles for per-viewport swap chains.
	uiVpPending.clear();
	return;
#else
	// At most ONE swap-chain create/resize per frame: back-to-back DXGI ops return ACCESS_DENIED
	// device removal. Skipped windows re-queue next frame.
	bool heavyOpDone = false;
	for (auto& kv : uiVpPending)
	{
		if (heavyOpDone) break;
		void* handle = kv.first;
		const int w = kv.second.first, h = kv.second.second;
		if (w < 8 || h < 8) continue;
#ifdef _WIN32
		// A queued op can outlive its HWND; any DXGI call on a dead window is ACCESS_DENIED + device removal.
		if (!::IsWindow((HWND)handle)) { uiVpSC.erase(handle); uiVpStable.erase(handle); continue; }
#endif   // macOS: imgui's Renderer_DestroyWindow tears the chain down before the NSWindow dies
		// Cool down after a failed creation: repeated DXGI failures escalate to device removal.
		{
			auto cd = uiVpCooldown.find(handle);
			if (cd != uiVpCooldown.end())
			{
				if (--cd->second > 0) continue;
				uiVpCooldown.erase(cd);
			}
		}
		// NUKE_VP_NORESIZE=1: never resize secondary chains, present stretched instead.
		static const bool noResize = []{ const char* e = std::getenv("NUKE_VP_NORESIZE"); return e && *e == '1'; }();
		RefCntAutoPtr<ISwapChain>& sc = uiVpSC[handle];
		if (sc && noResize) continue;
		if (!sc)
		{
			// Color format must match the main swap chain — the UI PSO was built for it. No depth.
			SwapChainDesc scd;
			scd.ColorBufferFormat = swapChain->GetDesc().ColorBufferFormat;
			scd.DepthBufferFormat = TEX_FORMAT_UNKNOWN;
			scd.Width = (Uint32)w; scd.Height = (Uint32)h;
			// Secondary chains must NOT be primary: a primary Present() runs FinishFrame() +
			// ReleaseStaleResources(), which must happen exactly once per frame.
			scd.IsPrimary = False;
#ifdef _WIN32
			Win32NativeWindow win{ handle };
			if (useVulkan)
				GetEngineFactoryVk()->CreateSwapChainVk(device, context, scd, win, &sc);
			else if (useD3D12)
				GetEngineFactoryD3D12()->CreateSwapChainD3D12(device, context, scd, FullScreenModeDesc{}, win, &sc);
			else
				GetEngineFactoryD3D11()->CreateSwapChainD3D11(device, context, scd, FullScreenModeDesc{}, win, &sc);
#elif defined(__APPLE__)
			// imgui_impl_glfw hands over the NSWindow: attach a CAMetalLayer to its content
			// view (same shim as the main window) and let MoltenVK own the surface.
			MacOSNativeWindow win{ NukeCocoaMetalViewForNSWindow(handle) };
			GetEngineFactoryVk()->CreateSwapChainVk(device, context, scd, win, &sc);   // Vulkan-only off Windows
#endif
			std::cout << "[NukeDiligent]	vp chain CREATE " << handle << " " << w << "x" << h
			          << (sc ? " ok" : " FAILED") << std::endl;
			if (!sc) { uiVpSC.erase(handle); uiVpCooldown[handle] = 120; }   // back off ~2s, don't hammer DXGI
			else uiVpGrace[handle] = 3;   // skip draw+present while imgui still adjusts the new OS window
			heavyOpDone = true;
		}
		else if ((int)sc->GetDesc().Width != w || (int)sc->GetDesc().Height != h)
		{
			// A target the driver already refused: asking again every frame is a live-lock.
			auto ref = uiVpRefused.find(handle);
			if (ref != uiVpRefused.end() && ref->second.first == w && ref->second.second == h) continue;
			// Debounced: a per-frame resize storm during a live drag starves the main swap chain's latency wait.
			auto& st = uiVpStable[handle];
			if (st.first.first != w || st.first.second != h) { st.first = { w, h }; st.second = 1; continue; }
			if (++st.second < 5) continue;   // ~5 frames unchanged = the drag settled
			st.second = 0;
			// Must resize via ISwapChain::Resize (unbinds back buffers, idles the GPU, resizes the
			// DXGI chain); recreating the chain from the factory instead fails with ACCESS_DENIED.
			context->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE);
			std::cout << "[NukeDiligent]	vp chain RESIZE " << handle << " " << sc->GetDesc().Width << "x"
			          << sc->GetDesc().Height << " -> " << w << "x" << h << std::endl;
			sc->Resize((Uint32)w, (Uint32)h);
			if ((int)sc->GetDesc().Width != w || (int)sc->GetDesc().Height != h)
			{
				uiVpRefused[handle] = { w, h };   // driver kept its own size: stop asking
				std::cout << "[NukeDiligent]	vp chain RESIZE REFUSED (kept " << sc->GetDesc().Width
				          << "x" << sc->GetDesc().Height << ")" << std::endl;
			}
			else uiVpRefused.erase(handle);
			std::cout << "[NukeDiligent]	vp chain RESIZE done" << std::endl;
			uiVpGrace[handle] = 2;   // settle frames after a resize (same DXGI race as creation)
			heavyOpDone = true;
		}
	}
	uiVpPending.clear();
#endif   // per-viewport swap chains (win32 + macOS)
}

// Render a detached window's UI into an offscreen texture and copy it to a staging ring;
// BlitHostWindows pushes the pixels to the window via GDI after the main present.
void NukeDiligent::uiViewportRender(void* nativeHandle, int w, int h, const NukeUIDrawData& data)
{
	if (!nativeHandle || w < 8 || h < 8 || !m_impl->device) return;
	// Vulkan: native per-window swapchains (imgui multi-viewport). D3D: GDI blit below.
	if (m_impl->useVulkan) { m_impl->ViewportRenderSwapchain(nativeHandle, w, h, data); return; }
	Impl::HostBlit& hb = m_impl->uiHostBlits[nativeHandle];
	if (!hb.rt || hb.w != w || hb.h != h)
	{
		if (hb.rt) m_impl->Trash(hb.rt);
		hb.rt.Release();
		for (auto& s : hb.staging) { if (s) m_impl->Trash(s); s.Release(); }
		hb.w = w; hb.h = h; hb.cur = 0;
		hb.valid[0] = hb.valid[1] = hb.valid[2] = false;
		const TEXTURE_FORMAT fmt = m_impl->swapChain ? m_impl->swapChain->GetDesc().ColorBufferFormat
		                                             : TEX_FORMAT_RGBA8_UNORM;
		TextureDesc td;
		td.Name = "host ui rt"; td.Type = RESOURCE_DIM_TEX_2D;
		td.Width = (Uint32)w; td.Height = (Uint32)h; td.Format = fmt; td.MipLevels = 1;
		td.BindFlags = BIND_RENDER_TARGET | BIND_SHADER_RESOURCE;
		m_impl->device->CreateTexture(td, nullptr, &hb.rt);
		TextureDesc st;
		st.Name = "host ui staging"; st.Type = RESOURCE_DIM_TEX_2D;
		st.Width = (Uint32)w; st.Height = (Uint32)h; st.Format = fmt; st.MipLevels = 1;
		st.Usage = USAGE_STAGING; st.CPUAccessFlags = CPU_ACCESS_READ; st.BindFlags = BIND_NONE;
		for (auto& s : hb.staging) m_impl->device->CreateTexture(st, nullptr, &s);
		if (!hb.rt || !hb.staging[0] || !hb.staging[1] || !hb.staging[2])
		{
			m_impl->uiHostBlits.erase(nativeHandle);
			return;
		}
		std::cout << "[NukeDiligent]\thost blit RT " << nativeHandle << " " << w << "x" << h << std::endl;
	}

	IDeviceContext* ctx = m_impl->context;
	ITextureView* rtv = hb.rt->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET);
	if (!rtv) return;
	ctx->SetRenderTargets(1, &rtv, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	const float clear[4] = { 0.06f, 0.06f, 0.07f, 1.0f };
	ctx->ClearRenderTarget(rtv, clear, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	m_impl->DrawUILists(rtv, (Uint32)w, (Uint32)h, data);
	ctx->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE);

	CopyTextureAttribs cp(hb.rt, RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
	                      hb.staging[hb.cur], RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	ctx->CopyTexture(cp);
	hb.valid[hb.cur] = true;
	m_impl->uiHostBlitQueue.push_back(nativeHandle);
}

// Vulkan native viewports: render a window's UI into its own swapchain. Creation/resize is
// deferred to ApplyPendingViewportOps; the present is queued after the main present.
void NukeDiligent::Impl::ViewportRenderSwapchain(void* nativeHandle, int w, int h, const NukeUIDrawData& data)
{
	auto it = uiVpSC.find(nativeHandle);
	ISwapChain* sc = (it != uiVpSC.end()) ? it->second.RawPtr() : nullptr;
	if (!sc || (int)sc->GetDesc().Width != w || (int)sc->GetDesc().Height != h)
		uiVpPending[nativeHandle] = { w, h };   // create/resize at the NEXT frame's top
	if (!sc) return;                            // first frame after opening: nothing to draw into yet

	{	// post-create/resize grace: sit out the settle frames
		auto g = uiVpGrace.find(nativeHandle);
		if (g != uiVpGrace.end())
		{
			if (--g->second > 0) return;
			uiVpGrace.erase(g);
		}
	}
	ITextureView* rtv = sc->GetCurrentBackBufferRTV();
	if (!rtv) return;
	context->SetRenderTargets(1, &rtv, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	const float clear[4] = { 0.06f, 0.06f, 0.07f, 1.0f };
	context->ClearRenderTarget(rtv, clear, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	DrawUILists(rtv, sc->GetDesc().Width, sc->GetDesc().Height, data);
	context->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE);
	vpPresentQueue.push_back(nativeHandle);     // presented AFTER the main Present
}

// Map the freshest GPU-completed staging of each host window and push its pixels via GDI.
// Must run after the main Present; maps with DO_NOT_WAIT so it never stalls the frame.
void NukeDiligent::Impl::BlitHostWindows()
{
	if (uiHostBlitQueue.empty()) return;
#ifndef _WIN32
	uiHostBlitQueue.clear();   // GDI host blit is Windows-only (D3D detached-window fallback)
	return;
#else
	for (void* hwnd : uiHostBlitQueue)
	{
		auto it = uiHostBlits.find(hwnd);
		if (it == uiHostBlits.end()) continue;
		HostBlit& hb = it->second;

		// Newest-first over the ring; nothing ready = the window keeps last frame's image.
		int mappedSlot = -1;
		MappedTextureSubresource msr{};
		for (int back = 0; back < 3 && mappedSlot < 0; ++back)
		{
			const int s = (hb.cur - back + 3) % 3;
			if (!hb.valid[s] || !hb.staging[s]) continue;
			msr = MappedTextureSubresource{};
			context->MapTextureSubresource(hb.staging[s], 0, 0, MAP_READ, MAP_FLAG_DO_NOT_WAIT, nullptr, msr);
			if (msr.pData) mappedSlot = s;
		}
		hb.cur = (hb.cur + 1) % 3;
		if (mappedSlot < 0) continue;

		// GDI wants BGRX top-down rows; the RT is RGBA8[_SRGB] or BGRA8[_SRGB].
		const TEXTURE_FORMAT fmt = hb.staging[mappedSlot]->GetDesc().Format;
		const bool needSwizzle = (fmt == TEX_FORMAT_RGBA8_UNORM || fmt == TEX_FORMAT_RGBA8_UNORM_SRGB);
		hb.scratch.resize((size_t)hb.w * hb.h * 4);
		const uint8_t* srcRows = (const uint8_t*)msr.pData;
		for (int y = 0; y < hb.h; ++y)
		{
			const uint8_t* srow = srcRows + (size_t)y * msr.Stride;
			uint8_t* drow = hb.scratch.data() + (size_t)y * hb.w * 4;
			if (!needSwizzle)
				memcpy(drow, srow, (size_t)hb.w * 4);
			else
				for (int x = 0; x < hb.w; ++x)
				{
					drow[x * 4 + 0] = srow[x * 4 + 2];
					drow[x * 4 + 1] = srow[x * 4 + 1];
					drow[x * 4 + 2] = srow[x * 4 + 0];
					drow[x * 4 + 3] = 255;
				}
		}
		context->UnmapTextureSubresource(hb.staging[mappedSlot], 0, 0);

		if (!::IsWindow((HWND)hwnd)) continue;
		HDC dc = GetDC((HWND)hwnd);
		if (!dc) continue;
		BITMAPINFO bi{};
		bi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
		bi.bmiHeader.biWidth = hb.w;
		bi.bmiHeader.biHeight = -hb.h;   // negative = top-down rows
		bi.bmiHeader.biPlanes = 1;
		bi.bmiHeader.biBitCount = 32;
		bi.bmiHeader.biCompression = BI_RGB;
		SetDIBitsToDevice(dc, 0, 0, hb.w, hb.h, 0, 0, 0, (UINT)hb.h, hb.scratch.data(), &bi, DIB_RGB_COLORS);
		ReleaseDC((HWND)hwnd, dc);
	}
	uiHostBlitQueue.clear();
#endif   // _WIN32
}
void NukeDiligent::uiViewportDestroy(void* nativeHandle)
{
	{
		auto hb = m_impl->uiHostBlits.find(nativeHandle);
		if (hb != m_impl->uiHostBlits.end())
		{
			if (hb->second.rt) m_impl->Trash(hb->second.rt);
			for (auto& s : hb->second.staging) if (s) m_impl->Trash(s);
			m_impl->uiHostBlits.erase(hb);
		}
	}
	auto it = m_impl->uiVpSC.find(nativeHandle);
	if (it == m_impl->uiVpSC.end()) return;
	// Park the swap chain in the GPU trash — the GPU may still be reading its back buffers.
	std::cout << "[NukeDiligent]	vp chain DESTROY " << nativeHandle << std::endl;
	m_impl->Trash(it->second);
	m_impl->uiVpSC.erase(it);
	m_impl->uiVpStable.erase(nativeHandle);
	m_impl->uiVpCooldown.erase(nativeHandle);
	m_impl->uiVpPending.erase(nativeHandle);
}

void NukeDiligent::getFrameStats(int& drawCalls, int& triangles)
{
	drawCalls = m_impl->statDrawsOut;
	triangles = m_impl->statTrisOut;
}

// Plugin export: a NUKEModule whose provides()="render" + phase()=PHASE_BOOT make the loader
// register queryService() (the iRender*) in the service registry during bootstrap.
class NukeDiligentModule : public NUKEModule
{
public:
	NukeDiligentModule()
	{
		std::strcpy(title, "Diligent Renderer");
		std::strcpy(description, "NukeEngine renderer backed by Diligent Engine (D3D11/D3D12).");
		std::strcpy(author, "Luastris");
		std::strcpy(site, "https://luastris.com");
		std::strcpy(version, "0.1.0");
		tags = { "render", "diligent", "d3d11", "d3d12" };
	}
	const char* provides() override { return "render"; }
	int         phase()    override { return PHASE_BOOT; }
	void*       queryService() override
	{
		if (!renderer) renderer = new NukeDiligent();
		return static_cast<iRender*>(renderer);
	}
	void OnLoad() override {}          // no component types to register
	void Run(AppInstance*) override {} // host-driven (main loop calls render()); no worker thread
	bool HasSettings() override { return false; }
	void Settings() override {}
	void Shutdown() override
	{
		// Teardown belongs here: boot providers are unloaded last, after every plugin that could touch the renderer.
		if (renderer) renderer->deinit();
		delete renderer;
		renderer = nullptr;
		stopped  = true;
	}

private:
	NukeDiligent* renderer = nullptr;
};

extern "C" BOOST_SYMBOL_EXPORT NukeDiligentModule plugin;
NukeDiligentModule plugin;
