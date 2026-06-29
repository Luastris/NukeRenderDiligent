#include "NukeDiligentImpl.h"


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

	// Target: an explicit RT (bindRenderTarget -> runtime UI into the viewport/camera RT) or the
	// backbuffer (editor UI). Bound here, no clear, so the UI composites over whatever's already there.
	ITextureView* uirtv = m_impl->uiRTV ? m_impl->uiRTV : m_impl->swapChain->GetCurrentBackBufferRTV();
	const Uint32 surfW = (m_impl->uiRTV && m_impl->uiTW) ? m_impl->uiTW : m_impl->swapChain->GetDesc().Width;
	const Uint32 surfH = (m_impl->uiRTV && m_impl->uiTH) ? m_impl->uiTH : m_impl->swapChain->GetDesc().Height;
	ctx->SetRenderTargets(1, &uirtv, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

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
