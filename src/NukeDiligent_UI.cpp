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
		uiVB.Release();
		while (uiVBSize < totalVtx) uiVBSize = uiVBSize ? uiVBSize * 2 : 4096;
		BufferDesc bd;
		bd.Name = "UI VB"; bd.BindFlags = BIND_VERTEX_BUFFER; bd.Usage = USAGE_DYNAMIC; bd.CPUAccessFlags = CPU_ACCESS_WRITE;
		bd.Size = (Uint64)uiVBSize * sizeof(NukeUIVert);
		dev->CreateBuffer(bd, nullptr, &uiVB);
	}
	if (!uiIB || uiIBSize < totalIdx)
	{
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
	// views behind — the SRB's strong ref must not keep them alive forever).
	if ((uiFrame & 511) == 0)
		for (auto it = uiSRBCache.begin(); it != uiSRBCache.end(); )
		{
			if (uiFrame - it->second.lastUse > 512) it = uiSRBCache.erase(it);
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

void NukeDiligent::uiViewportRender(void* nativeHandle, int w, int h, const NukeUIDrawData& data)
{
	// Degenerate sizes come through while a window is minimizing/restoring — presenting
	// or resizing then can remove the D3D12 device. Sit those frames out.
	if (!nativeHandle || w < 8 || h < 8 || !m_impl->device) return;

	RefCntAutoPtr<ISwapChain>& sc = m_impl->uiVpSC[nativeHandle];
	if (!sc)
	{
		// Same color format as the main swap chain (the UI PSO was built for it);
		// no depth — the UI never depth-tests.
		SwapChainDesc scd;
		scd.ColorBufferFormat = m_impl->swapChain->GetDesc().ColorBufferFormat;
		scd.DepthBufferFormat = TEX_FORMAT_UNKNOWN;
		scd.Width = (Uint32)w; scd.Height = (Uint32)h;
		// THE ROOT of two weeks of "device removed" asserts: SwapChainDesc defaults to
		// IsPrimary = true, and a PRIMARY swap chain's Present() runs FinishFrame() +
		// ReleaseStaleResources() — with a secondary window open that happened TWICE per
		// frame, corrupting the frame-resource lifetime bookkeeping (the GPU kept reading
		// memory the second FinishFrame released) -> random device removals. It also gave
		// every secondary window its own frame-latency waitable (500 ms stalls when the
		// window is occluded -> the stutters). Secondary windows are NOT primary.
		scd.IsPrimary = False;
		Win32NativeWindow win{ nativeHandle };
		if (m_impl->useD3D12)
			GetEngineFactoryD3D12()->CreateSwapChainD3D12(m_impl->device, m_impl->context, scd, FullScreenModeDesc{}, win, &sc);
		else
			GetEngineFactoryD3D11()->CreateSwapChainD3D11(m_impl->device, m_impl->context, scd, FullScreenModeDesc{}, win, &sc);
		if (!sc) { m_impl->uiVpSC.erase(nativeHandle); return; }
	}
	IDeviceContext* ctx = m_impl->context;
	const SwapChainDesc& scd = sc->GetDesc();
	if ((int)scd.Width != w || (int)scd.Height != h)
	{
		// Resize needs the back buffers UNBOUND and NOT referenced by in-flight GPU
		// work, or D3D12 removes the device. A window-resize event is rare — a full
		// GPU idle here is cheap insurance.
		ctx->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE);
		ctx->Flush();
		m_impl->device->IdleGPU();
		sc->Resize((Uint32)w, (Uint32)h);
	}

	ITextureView* rtv = sc->GetCurrentBackBufferRTV();
	if (!rtv) return;
	ctx->SetRenderTargets(1, &rtv, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	const float clear[4] = { 0.06f, 0.06f, 0.07f, 1.0f };
	ctx->ClearRenderTarget(rtv, clear, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	m_impl->DrawUILists(rtv, sc->GetDesc().Width, sc->GetDesc().Height, data);
	// Unbind the secondary back buffer BEFORE presenting it — the state cache must not
	// hold a presented buffer when the next frame's main passes start binding targets.
	ctx->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE);
	sc->Present(0);   // no vsync: the main window's present already paces the frame
}

void NukeDiligent::uiViewportDestroy(void* nativeHandle)
{
	auto it = m_impl->uiVpSC.find(nativeHandle);
	if (it == m_impl->uiVpSC.end()) return;
	// The GPU may still be reading this swap chain's back buffers (frames in flight);
	// releasing them mid-use REMOVES THE DEVICE — it surfaces later as the
	// GetCompletedFenceValue == UINT64_MAX assert storm. Settle the queue first: window
	// destruction is rare, a full idle is cheap insurance. NOTE imgui also RECREATES
	// platform windows on viewport merge/DPI changes, not just on user close.
	if (m_impl->context && m_impl->device)
	{
		m_impl->context->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE);
		m_impl->context->Flush();
		m_impl->device->IdleGPU();
	}
	m_impl->uiVpSC.erase(it);
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
