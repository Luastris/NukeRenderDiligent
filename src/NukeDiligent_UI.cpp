#include "NukeDiligentImpl.h"


// Per-texture SRB (MUTABLE variable, set ONCE at creation): committing it costs no
// dynamic GPU descriptors, unlike a DYNAMIC variable which allocated fresh ones on
// every CommitShaderResources — dozens per frame per window, bleeding the heap dry.
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

// Shared UI draw body: upload the lists, then draw them into the given target.
// Used by renderDrawLists (main window / bound RT) and uiViewportRender (a detached
// OS window's own swap chain).
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
		Trash(uiVB);   // grows MID-FRAME (main window drew, a bigger secondary window follows)
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

	// LRU purge: drop SRBs of textures not drawn for a while (resized RTs leave stale
	// views behind — the SRB's strong ref must not keep them alive forever). Parked, not
	// freed inline (centralized lifetime rule).
	if ((uiFrame & 511) == 0)
		for (auto it = uiSRBCache.begin(); it != uiSRBCache.end(); )
		{
			if (uiFrame - it->second.lastUse > 512) { Trash(it->second.srb); it = uiSRBCache.erase(it); }
			else ++it;
		}
}

void NukeDiligent::renderDrawLists(const NukeUIDrawData& data)
{
	// Target: an explicit RT (bindRenderTarget -> runtime UI into the viewport/camera RT) or the
	// backbuffer (editor UI). No clear — the UI composites over whatever's already there.
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

// Applied at the TOP of render(), before anything is recorded: create queued swap chains and
// resize mismatched ones. Never mid-frame — see the Impl field comment.
void NukeDiligent::Impl::ApplyPendingViewportOps()
{
	++uiVpFrameNo;   // multi-window interleave clock (uiViewportRender) — ticks EVERY frame
	if (uiVpPending.empty() || !device) return;
	// AT MOST ONE heavy DXGI op (secondary-chain create/resize) per frame: opening several
	// detached windows in the same frame (e.g. session restore) created chains back-to-back
	// and DXGI answered with ACCESS_DENIED device removal. Skipped windows simply re-queue
	// through uiViewportRender's size-mismatch check next frame.
	bool heavyOpDone = false;
	for (auto& kv : uiVpPending)
	{
		if (heavyOpDone) break;
		void* handle = kv.first;
		const int w = kv.second.first, h = kv.second.second;
		if (w < 8 || h < 8) continue;
		// imgui DESTROYS/RECREATES platform windows (viewport merge, DPI) — a queued op can
		// outlive its HWND, and any DXGI call on a dead window is ACCESS_DENIED + device
		// removal. Validate first; purge state for windows that are gone.
		if (!::IsWindow((HWND)handle)) { uiVpSC.erase(handle); uiVpStable.erase(handle); continue; }
		// A failed creation must NOT retry every frame (repeated DXGI failures escalate to
		// device removal) — cool down before trying that window again.
		{
			auto cd = uiVpCooldown.find(handle);
			if (cd != uiVpCooldown.end())
			{
				if (--cd->second > 0) continue;
				uiVpCooldown.erase(cd);
			}
		}
		// Experiment gate (NUKE_VP_NORESIZE=1): never resize secondary chains — present
		// stretched forever. Used to isolate the ACCESS_DENIED device removal to the
		// secondary-resize path.
		static const bool noResize = []{ const char* e = std::getenv("NUKE_VP_NORESIZE"); return e && *e == '1'; }();
		RefCntAutoPtr<ISwapChain>& sc = uiVpSC[handle];
		if (sc && noResize) continue;
		if (!sc)
		{
			// Same color format as the main swap chain (the UI PSO was built for it);
			// no depth — the UI never depth-tests.
			SwapChainDesc scd;
			scd.ColorBufferFormat = swapChain->GetDesc().ColorBufferFormat;
			scd.DepthBufferFormat = TEX_FORMAT_UNKNOWN;
			scd.Width = (Uint32)w; scd.Height = (Uint32)h;
			// THE ROOT of two weeks of "device removed" asserts: SwapChainDesc defaults to
			// IsPrimary = true, and a PRIMARY swap chain's Present() runs FinishFrame() +
			// ReleaseStaleResources() — with a secondary window open that happened TWICE per
			// frame, corrupting the frame-resource lifetime bookkeeping. Secondary windows
			// are NOT primary.
			scd.IsPrimary = False;
			Win32NativeWindow win{ handle };
			if (useVulkan)
				GetEngineFactoryVk()->CreateSwapChainVk(device, context, scd, win, &sc);
			else if (useD3D12)
				GetEngineFactoryD3D12()->CreateSwapChainD3D12(device, context, scd, FullScreenModeDesc{}, win, &sc);
			else
				GetEngineFactoryD3D11()->CreateSwapChainD3D11(device, context, scd, FullScreenModeDesc{}, win, &sc);
			std::cout << "[NukeDiligent]	vp chain CREATE " << handle << " " << w << "x" << h
			          << (sc ? " ok" : " FAILED") << std::endl;
			if (!sc) { uiVpSC.erase(handle); uiVpCooldown[handle] = 120; }   // back off ~2s, don't hammer DXGI
			// GRACE: skip this window's draw+present for a few frames — imgui is still
			// adjusting the freshly created OS window (pos/style/DPI), and presenting into
			// it mid-adjustment races DXGI into ACCESS_DENIED (the open-time flake).
			else uiVpGrace[handle] = 3;
			heavyOpDone = true;
		}
		else if ((int)sc->GetDesc().Width != w || (int)sc->GetDesc().Height != h)
		{
			// DEBOUNCED (see uiVpStable): a live drag asks for a new size every frame, and a
			// per-frame recreate/resize storm starves the main swap chain's frame-latency
			// wait. Act only after the size holds still for a few frames.
			auto& st = uiVpStable[handle];
			if (st.first.first != w || st.first.second != h) { st.first = { w, h }; st.second = 1; continue; }
			if (++st.second < 5) continue;   // ~5 frames unchanged = the drag settled
			st.second = 0;
			// RESIZE via Diligent's own path: SwapChainD3D12::Resize internally unbinds the
			// back buffers from the framebuffer, clears its RTVs, idles the GPU and resizes
			// (or recreates) the DXGI chain — the ONE sequence DXGI accepts. Recreating the
			// whole ISwapChain from the factory instead leaves the old chain's deferred
			// buffers alive on the HWND and CreateSwapChainForHwnd dies with ACCESS_DENIED
			// (device removal) — the failure mode this replaced.
			context->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE);
			std::cout << "[NukeDiligent]	vp chain RESIZE " << handle << " " << sc->GetDesc().Width << "x"
			          << sc->GetDesc().Height << " -> " << w << "x" << h << std::endl;
			sc->Resize((Uint32)w, (Uint32)h);
			std::cout << "[NukeDiligent]	vp chain RESIZE done" << std::endl;
			uiVpGrace[handle] = 2;   // settle frames after a resize (same DXGI race as creation)
			heavyOpDone = true;
		}
	}
	uiVpPending.clear();
}

// A detached window's frame: render its UI into an OFFSCREEN texture and copy it to a
// staging ring; the pixels reach the window via GDI AFTER the main present
// (Impl::BlitHostWindows). NO swap chain is ever created for the window, so the whole
// class of secondary-swapchain DXGI races (create/resize/present vs a heavy frame -
// a month of ACCESS_DENIED device removals) is gone BY CONSTRUCTION.
void NukeDiligent::uiViewportRender(void* nativeHandle, int w, int h, const NukeUIDrawData& data)
{
	if (!nativeHandle || w < 8 || h < 8 || !m_impl->device) return;
	// Vulkan: native per-window swapchains (imgui multi-viewport). D3D: GDI blit below.
	if (m_impl->useVulkan) { m_impl->ViewportRenderSwapchain(nativeHandle, w, h, data); return; }
	Impl::HostBlit& hb = m_impl->uiHostBlits[nativeHandle];
	if (!hb.rt || hb.w != w || hb.h != h)
	{
		// Plain textures: a resize is create-new/park-old through the central GPU trash -
		// no DXGI, no debounce, no grace frames.
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

// VULKAN native viewports: render this window's UI into ITS OWN swapchain. Creation and
// resizes are deferred to the next frame's top (ApplyPendingViewportOps) with the same
// debounce/grace defenses; the present is queued after the main present.
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

// After the main Present: map the freshest GPU-COMPLETED staging of each host window
// (DO_NOT_WAIT - never stalls the frame) and push the pixels to the window with GDI.
void NukeDiligent::Impl::BlitHostWindows()
{
	if (uiHostBlitQueue.empty()) return;
	for (void* hwnd : uiHostBlitQueue)
	{
		auto it = uiHostBlits.find(hwnd);
		if (it == uiHostBlits.end()) continue;
		HostBlit& hb = it->second;

		// Newest-first, fall back to older ring slots - whichever the GPU has finished.
		// Nothing ready = the window keeps last frame's image.
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
}
void NukeDiligent::uiViewportDestroy(void* nativeHandle)
{
	// GDI-blit host resources (the current host path): park textures in the GPU trash.
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
	// The GPU may still be reading this swap chain's back buffers (frames in flight) —
	// PARK the swap chain in the centralized GPU trash instead of a mid-frame IdleGPU:
	// it stays alive for kTrashFrames, by which point every present that referenced it
	// has completed. NOTE imgui also RECREATES platform windows on viewport merge/DPI
	// changes, not just on user close.
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

// ---- Plugin export (boost::dll, unified plugin model) ----
// An ordinary NUKEModule exported as "plugin", like every other plugin. What makes it a
// renderer is metadata: provides()="render" + phase()=PHASE_BOOT. The loader enables it
// during bootstrap and registers queryService() (the iRender*) in the service registry;
// the host then grabs it via GetService<iRender>() and drives init/window itself.
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
		// Full renderer teardown lives HERE (not in the host): the loader revoked the
		// "render" service already, and UnloadModules shuts boot providers down LAST,
		// after every runtime plugin that might still touch the renderer is gone.
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
