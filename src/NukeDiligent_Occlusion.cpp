#include "NukeDiligentImpl.h"
#include <cstring>
#include <cmath>

// Hi-Z occlusion culling: per-target visibility history, MAX depth pyramid, GPU box test,
// deferred indirect replay and the latency-hidden readback. The draw-path hooks (tag scope,
// deferral, indirect draws) live in NukeDiligent_Scene.cpp.

// ---- resources ---------------------------------------------------------------------------------

void NukeDiligent::Impl::CreateOcclResources()
{
	occlCSPSO.Release(); hizCopyPSO.Release(); hizCopyMSPSO.Release(); hizDownPSO.Release();
	occlCSSRB.Release(); hizCopySRB.Release(); hizCopyMSSRB.Release(); hizDownSRB.Release();
	hizCopyVar = hizCopyMSVar = hizDownVar = nullptr;
	const std::string vs = shaderSource("post.vs");
	const std::string cp = shaderSource("hiz_copy.ps"), dn = shaderSource("hiz_down.ps"), cs = shaderSource("occl.cs");
	if (vs.empty() || cp.empty() || dn.empty() || cs.empty())
	{ cout << "[NukeDiligent]\tHi-Z occlusion shaders missing — occlusion culling off" << endl; return; }

	if (!occlCB)
	{
		BufferDesc d; d.Name = "OcclCB"; d.Size = sizeof(float) * 16 + sizeof(float) * 4 + sizeof(Uint32) * 4;
		d.Usage = USAGE_DYNAMIC; d.BindFlags = BIND_UNIFORM_BUFFER; d.CPUAccessFlags = CPU_ACCESS_WRITE;
		device->CreateBuffer(d, nullptr, &occlCB);
	}

	// Fullscreen pyramid passes (post.vs + pixel reduce), R32F targets, Load-only (no sampler).
	auto fsPSO = [&](const std::string& ps, const char* dbg, const char* var, bool msaa,
	                 RefCntAutoPtr<IPipelineState>& pso, RefCntAutoPtr<IShaderResourceBinding>& srb, IShaderResourceVariable*& v)
	{
		ShaderCreateInfo s; s.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
		ShaderMacro m[] = {{"HIZ_MSAA", "1"}};
		if (msaa) s.Macros = ShaderMacroArray{m, 1};
		RefCntAutoPtr<IShader> vv, pp;
		s.Desc = {dbg, SHADER_TYPE_VERTEX, true}; s.Source = vs.c_str(); CreateShaderCached(s, &vv);
		s.Desc = {dbg, SHADER_TYPE_PIXEL, true};  s.Source = ps.c_str(); CreateShaderCached(s, &pp);
		if (!vv || !pp) return;
		GraphicsPipelineStateCreateInfo ci; ci.PSODesc.Name = dbg;
		auto& gp = ci.GraphicsPipeline;
		gp.NumRenderTargets = 1; gp.RTVFormats[0] = TEX_FORMAT_R32_FLOAT; gp.DSVFormat = TEX_FORMAT_UNKNOWN;
		gp.PrimitiveTopology = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; gp.RasterizerDesc.CullMode = CULL_MODE_NONE;
		gp.DepthStencilDesc.DepthEnable = False; gp.InputLayout.NumElements = 0;
		ShaderResourceVariableDesc vars[] = {{SHADER_TYPE_PIXEL, var, SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC}};
		ci.PSODesc.ResourceLayout.Variables = vars; ci.PSODesc.ResourceLayout.NumVariables = 1;
		ci.pVS = vv; ci.pPS = pp;
		CreateGraphicsPipelineStateCached(ci, &pso);
		if (!pso) { cout << "[NukeDiligent]\tHi-Z PSO failed: " << dbg << endl; return; }
		pso->CreateShaderResourceBinding(&srb, true);
		v = srb ? srb->GetVariableByName(SHADER_TYPE_PIXEL, var) : nullptr;
	};
	fsPSO(cp, "HiZ Copy",    "g_Depth", false, hizCopyPSO,   hizCopySRB,   hizCopyVar);
	fsPSO(cp, "HiZ Copy MS", "g_Depth", true,  hizCopyMSPSO, hizCopyMSSRB, hizCopyMSVar);
	fsPSO(dn, "HiZ Down",    "g_Src",   false, hizDownPSO,   hizDownSRB,   hizDownVar);

	// Box test + indirect-argument compute.
	{
		ShaderCreateInfo s; s.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
		s.Desc = {"Occlusion CS", SHADER_TYPE_COMPUTE, true}; s.Source = cs.c_str();
		RefCntAutoPtr<IShader> csh; CreateShaderCached(s, &csh);
		if (csh)
		{
			ComputePipelineStateCreateInfo cci; cci.PSODesc.Name = "Occlusion PSO";
			ShaderResourceVariableDesc cvars[] = {
				{SHADER_TYPE_COMPUTE, "g_Aabbs", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
				{SHADER_TYPE_COMPUTE, "g_Recs",  SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
				{SHADER_TYPE_COMPUTE, "g_Vis",   SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
				{SHADER_TYPE_COMPUTE, "g_Args",  SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
				{SHADER_TYPE_COMPUTE, "g_HiZ",   SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
			};
			cci.PSODesc.ResourceLayout.Variables = cvars; cci.PSODesc.ResourceLayout.NumVariables = 5;
			cci.pCS = csh;
			CreateComputePipelineStateCached(cci, &occlCSPSO);
			if (occlCSPSO)
			{
				if (auto* v = occlCSPSO->GetStaticVariableByName(SHADER_TYPE_COMPUTE, "OcclCB")) v->Set(occlCB);
				occlCSPSO->CreateShaderResourceBinding(&occlCSSRB, true);
			}
		}
	}
	cout << "[NukeDiligent]\tHi-Z occlusion " << (occlCSPSO && occlCSSRB && hizCopyPSO && hizDownPSO ? "ready" : "FAILED") << endl;
}

// Grow the tag/record/verdict/argument buffers to hold `n` entries (power-of-two capacity).
void NukeDiligent::Impl::OcclEnsureBuffers(Uint32 n)
{
	if (n <= occlCap && occlAabbBuf) return;
	Uint32 cap = occlCap ? occlCap : 256;
	while (cap < n) cap *= 2;
	Trash(occlAabbBuf); Trash(occlRecBuf); Trash(occlVisBuf); Trash(occlArgsBuf);
	occlAabbBuf.Release(); occlRecBuf.Release(); occlVisBuf.Release(); occlArgsBuf.Release();
	auto make = [&](const char* name, Uint32 stride, Uint64 size, Diligent::BIND_FLAGS bind, BUFFER_MODE mode, RefCntAutoPtr<IBuffer>& out)
	{
		BufferDesc d; d.Name = name; d.Size = size; d.Usage = USAGE_DEFAULT; d.BindFlags = bind;
		d.Mode = mode; d.ElementByteStride = stride;
		device->CreateBuffer(d, nullptr, &out);
	};
	make("Occl AABBs", 32, (Uint64)cap * 32, BIND_SHADER_RESOURCE, BUFFER_MODE_STRUCTURED, occlAabbBuf);
	make("Occl Recs",  32, (Uint64)cap * 32, BIND_SHADER_RESOURCE, BUFFER_MODE_STRUCTURED, occlRecBuf);
	make("Occl Vis",    4, (Uint64)cap * 4,  BIND_UNORDERED_ACCESS | BIND_SHADER_RESOURCE, BUFFER_MODE_STRUCTURED, occlVisBuf);
	make("Occl Args",   4, (Uint64)cap * 20, BIND_UNORDERED_ACCESS | BIND_INDIRECT_DRAW_ARGS, BUFFER_MODE_RAW, occlArgsBuf);
	occlCap = cap;
}

// ---- per-camera flow ---------------------------------------------------------------------------

bool NukeDiligent::Impl::OcclDecide(uint64_t id)
{
	OcclView& v = occlViews[curCamKey];
	auto it = v.hist.find(id);
	if (it == v.hist.end()) return true;   // never tested: draw (conservative)
	if (!occlFreeze) it->second.frame = occlFrame;
	return it->second.visible;
}

bool NukeDiligent::Impl::OcclArm(int& slot, bool& defer)
{
	if (occlPendingSlot < 0)
	{
		occlPendingSlot = (int)occlTags.size();
		occlTags.push_back(occlPendingTag);
		occlPendingDefer = !OcclDecide(occlPendingTag.id);
	}
	slot = occlPendingSlot; defer = occlPendingDefer;
	return defer;
}

void NukeDiligent::Impl::OcclBeginCamera()
{
	occlTags.clear(); occlDeferred.clear();
	occlPending = false; occlDrawTag = false; occlPendingSlot = -1; occlPendingDefer = false; occlReplay = -1;
	occlPassActive = occlEnabled && occlCSPSO && occlCSSRB && hizCopyPSO && hizDownPSO;
	if (!occlPassActive) { statOcclTracked = statOcclCulled = 0; return; }
	OcclView& v = occlViews[curCamKey];
	v.lastUsed = occlFrame;
	// Bound the per-target state (previews, camera-to-texture churn): drop the coldest view.
	if (occlViews.size() > 8)
	{
		uint64_t oldest = ~0ull; uint64_t key = 0; bool found = false;
		for (auto& kv : occlViews)
			if (kv.first != curCamKey && kv.second.lastUsed < oldest) { oldest = kv.second.lastUsed; key = kv.first; found = true; }
		if (found)
		{
			OcclView& o = occlViews[key];
			Trash(o.hiz); Trash(o.hizScratch); for (auto& r : o.ring) Trash(r.staging);
			occlViews.erase(key);
		}
	}
	if (occlFreeze) return;   // verdicts stay as they were
	// Matured readbacks -> history. DO_NOT_WAIT: a slot that is not ready waits another frame.
	for (OcclView::Ring& r : v.ring)
	{
		if (r.pending < 0) continue;
		if (r.pending > 0) { --r.pending; continue; }
		void* p = nullptr;
		context->MapBuffer(r.staging, MAP_READ, MAP_FLAG_DO_NOT_WAIT, p);
		if (!p)
		{
			// D3D11 answers WAS_STILL_DRAWING (null), but the debug context already booked the
			// buffer as mapped: release the booking or the next map asserts "already been mapped".
			context->UnmapBuffer(r.staging, MAP_READ);
			continue;
		}
		const Uint32* vis = (const Uint32*)p;
		for (size_t i = 0; i < r.ids.size(); ++i)
		{
			OcclHist& h = v.hist[r.ids[i]];
			h.visible = vis[i] != 0; h.frame = occlFrame;
		}
		context->UnmapBuffer(r.staging, MAP_READ);
		r.pending = -1; r.ids.clear();
	}
	// Age out ids that stopped being submitted (unloaded/culled by the frustum for a while).
	if ((occlFrame & 127) == 0)
		for (auto it = v.hist.begin(); it != v.hist.end();)
			if (it->second.frame + 600 < occlFrame) it = v.hist.erase(it); else ++it;
}

// Build the MAX pyramid of the camera depth into v.hiz. Source level m-1 is copied to a scratch
// chain first: Diligent tracks resource state per texture, so one texture can't be both the
// RTV of mip m and the SRV of mip m-1 in the same pass.
void NukeDiligent::Impl::OcclBuildHiZ(OcclView& v, ITexture* depth, int w, int h)
{
	int mips = 1; while ((w >> mips) > 0 || (h >> mips) > 0) ++mips;
	if (!v.hiz || v.hizW != w || v.hizH != h)
	{
		Trash(v.hiz); Trash(v.hizScratch);
		v.hiz.Release(); v.hizScratch.Release(); v.hizRTV.clear(); v.hizSrcSRV.clear();
		TextureDesc td; td.Name = "HiZ"; td.Type = RESOURCE_DIM_TEX_2D; td.Width = (Uint32)w; td.Height = (Uint32)h;
		td.MipLevels = (Uint32)mips; td.Format = TEX_FORMAT_R32_FLOAT; td.BindFlags = BIND_RENDER_TARGET | BIND_SHADER_RESOURCE;
		device->CreateTexture(td, nullptr, &v.hiz);
		td.Name = "HiZ Scratch"; td.BindFlags = BIND_SHADER_RESOURCE;
		device->CreateTexture(td, nullptr, &v.hizScratch);
		if (!v.hiz || !v.hizScratch) { v.hiz.Release(); v.hizScratch.Release(); return; }
		for (int m = 0; m < mips; ++m)
		{
			TextureViewDesc vd; vd.ViewType = TEXTURE_VIEW_RENDER_TARGET; vd.MostDetailedMip = (Uint32)m; vd.NumMipLevels = 1;
			RefCntAutoPtr<ITextureView> rtv; v.hiz->CreateView(vd, &rtv); v.hizRTV.push_back(rtv);
			vd.ViewType = TEXTURE_VIEW_SHADER_RESOURCE;
			RefCntAutoPtr<ITextureView> srv; v.hizScratch->CreateView(vd, &srv); v.hizSrcSRV.push_back(srv);
		}
		v.hizW = w; v.hizH = h; v.hizMips = mips;
	}
	auto pass = [&](IPipelineState* pso, IShaderResourceBinding* srb, ITextureView* rtv, int vw, int vh)
	{
		context->SetRenderTargets(1, &rtv, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		Viewport vp; vp.TopLeftX = 0; vp.TopLeftY = 0; vp.Width = (float)vw; vp.Height = (float)vh; vp.MinDepth = 0; vp.MaxDepth = 1;
		context->SetViewports(1, &vp, vw, vh);
		context->SetPipelineState(pso);
		context->CommitShaderResources(srb, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		DrawAttribs da{3, DRAW_FLAG_VERIFY_STATES}; context->Draw(da);
	};
	const bool ms = depth->GetDesc().SampleCount > 1;
	IShaderResourceVariable* cv = ms ? hizCopyMSVar : hizCopyVar;
	if (cv) cv->Set(depth->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
	pass(ms ? hizCopyMSPSO : hizCopyPSO, ms ? hizCopyMSSRB : hizCopySRB, v.hizRTV[0], w, h);
	for (int m = 1; m < mips; ++m)
	{
		// Unbind first: mip m-1 is still the bound render target, and copying out of a bound
		// target makes Vulkan unbind it with a warning (a log line per level per frame).
		context->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE);
		CopyTextureAttribs ca(v.hiz, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, v.hizScratch, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		ca.SrcMipLevel = (Uint32)(m - 1); ca.DstMipLevel = (Uint32)(m - 1);
		context->CopyTexture(ca);
		if (hizDownVar) hizDownVar->Set(v.hizSrcSRV[m - 1]);
		pass(hizDownPSO, hizDownSRB, v.hizRTV[m], std::max(1, w >> m), std::max(1, h >> m));
	}
}

// Frozen view: the draws the history holds back, as red wire boxes (what the culling removed).
void NukeDiligent::Impl::OcclDebugBoxes()
{
	std::lock_guard<std::mutex> lock(debugMutex);
	for (const OcclDeferred& d : occlDeferred)
	{
		if (d.tag < 0 || d.tag >= (int)occlTags.size()) continue;
		const OcclTag& t = occlTags[d.tag];
		auto P = [&](int c, float* o) { o[0] = (c & 1) ? t.mx[0] : t.mn[0]; o[1] = (c & 2) ? t.mx[1] : t.mn[1]; o[2] = (c & 4) ? t.mx[2] : t.mn[2]; };
		static const int E[12][2] = {{0,1},{2,3},{4,5},{6,7},{0,2},{1,3},{4,6},{5,7},{0,4},{1,5},{2,6},{3,7}};
		for (auto& e : E)
		{
			float a[3], b[3]; P(e[0], a); P(e[1], b);
			debugVerts.insert(debugVerts.end(), { a[0], a[1], a[2], 1.0f, 0.2f, 0.15f, 1.0f,
			                                      b[0], b[1], b[2], 1.0f, 0.2f, 0.15f, 1.0f });
		}
	}
}

// Pyramid + test + readback copy. Returns 0 = draw nothing deferred (frozen), 1 = replay the
// deferred draws indirect (arguments written), 2 = replay them plain (no usable depth).
int NukeDiligent::Impl::OcclEndOpaque()
{
	occlPassActive = false; occlPending = false; occlDrawTag = false;
	statOcclTracked = (int)occlTags.size(); statOcclCulled = (int)occlDeferred.size();
	if (occlTags.empty()) return 2;
	if (occlFreeze) { OcclDebugBoxes(); return 0; }
	OcclView& v = occlViews[curCamKey];
	ITextureView* rtv = curRTV; ITextureView* dsv = curDSV;
	if (!dsv || curRTW <= 0 || curRTH <= 0) return 2;
	ITexture* depth = dsv->GetTexture();
	const int w = curRTW, h = curRTH;

	GpuPass("hiz");
	context->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE);   // depth becomes a source
	OcclBuildHiZ(v, depth, w, h);
	if (!v.hiz)
	{
		context->SetRenderTargets(1, &rtv, dsv, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		GpuPass("scene");
		return 2;
	}

	const Uint32 nTags = (Uint32)occlTags.size(), nRecs = (Uint32)occlDeferred.size();
	OcclEnsureBuffers(std::max(nTags, nRecs));
	{
		std::vector<float> ab((size_t)nTags * 8);
		for (Uint32 i = 0; i < nTags; ++i)
		{
			const OcclTag& t = occlTags[i];
			float* o = &ab[(size_t)i * 8];
			o[0] = t.mn[0]; o[1] = t.mn[1]; o[2] = t.mn[2]; o[3] = 0.0f;
			o[4] = t.mx[0]; o[5] = t.mx[1]; o[6] = t.mx[2]; o[7] = 0.0f;
		}
		context->UpdateBuffer(occlAabbBuf, 0, (Uint64)nTags * 32, ab.data(), RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		if (nRecs)
		{
			std::vector<Uint32> rc((size_t)nRecs * 8, 0u);
			for (Uint32 i = 0; i < nRecs; ++i)
			{
				const OcclDeferred& d = occlDeferred[i];
				Uint32* o = &rc[(size_t)i * 8];
				o[0] = (Uint32)d.tag; o[1] = d.recCount; o[2] = d.recFirst; o[3] = d.recInst; o[4] = d.recFirstInst; o[5] = d.recIndexed ? 1u : 0u;
			}
			context->UpdateBuffer(occlRecBuf, 0, (Uint64)nRecs * 32, rc.data(), RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		}
	}
	struct OcclCBData { float4x4 vp; float dims[4]; Uint32 counts[4]; };
	auto writeCB = [&](Uint32 mode)
	{
		MapHelper<OcclCBData> cb(context, occlCB, MAP_WRITE, MAP_FLAG_DISCARD);
		if (cb == nullptr) return false;
		cb->vp = curView * curProjNoJitter;
		cb->dims[0] = (float)w; cb->dims[1] = (float)h; cb->dims[2] = (float)v.hizMips; cb->dims[3] = 0.0f;
		cb->counts[0] = nTags; cb->counts[1] = nRecs; cb->counts[2] = mode; cb->counts[3] = 0;
		return true;
	};
	auto set = [&](const char* n, IDeviceObject* o)
	{ if (auto* var = occlCSSRB->GetVariableByName(SHADER_TYPE_COMPUTE, n)) var->Set(o); };
	set("g_Aabbs", occlAabbBuf->GetDefaultView(BUFFER_VIEW_SHADER_RESOURCE));
	set("g_Recs",  occlRecBuf->GetDefaultView(BUFFER_VIEW_SHADER_RESOURCE));
	set("g_Vis",   occlVisBuf->GetDefaultView(BUFFER_VIEW_UNORDERED_ACCESS));
	set("g_Args",  occlArgsBuf->GetDefaultView(BUFFER_VIEW_UNORDERED_ACCESS));
	set("g_HiZ",   v.hiz->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
	context->SetPipelineState(occlCSPSO);
	context->CommitShaderResources(occlCSSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	if (!writeCB(0)) return 2;
	context->DispatchCompute(DispatchComputeAttribs((nTags + 63) / 64, 1, 1));
	if (nRecs)
	{
		// The second dispatch reads g_Vis the first one wrote: make the UAV writes visible.
		StateTransitionDesc b(occlVisBuf, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_UNORDERED_ACCESS, STATE_TRANSITION_FLAG_UPDATE_STATE);
		context->TransitionResourceStates(1, &b);
		if (writeCB(1)) context->DispatchCompute(DispatchComputeAttribs((nRecs + 63) / 64, 1, 1));
	}
	// Verdicts -> staging ring (read back by OcclBeginCamera a few frames on).
	OcclView::Ring& r = v.ring[v.ringHead]; v.ringHead = (v.ringHead + 1) % 3;
	if (!r.staging || r.staging->GetDesc().Size < (Uint64)occlCap * 4)
	{
		Trash(r.staging); r.staging.Release();
		BufferDesc sd; sd.Name = "Occl Readback"; sd.Size = (Uint64)occlCap * 4; sd.Usage = USAGE_STAGING;
		sd.BindFlags = BIND_NONE; sd.CPUAccessFlags = CPU_ACCESS_READ;
		device->CreateBuffer(sd, nullptr, &r.staging);
		r.pending = -1;
	}
	if (r.staging)
	{
		context->CopyBuffer(occlVisBuf, 0, RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
		                    r.staging, 0, (Uint64)nTags * 4, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		r.ids.resize(nTags);
		for (Uint32 i = 0; i < nTags; ++i) r.ids[i] = occlTags[i].id;
		r.pending = 2;
	}
	// Back to the camera targets for the deferred replay + the rest of the pass.
	context->SetRenderTargets(1, &rtv, dsv, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	Viewport vp; vp.TopLeftX = 0; vp.TopLeftY = 0; vp.Width = (float)w; vp.Height = (float)h; vp.MinDepth = 0; vp.MaxDepth = 1;
	context->SetViewports(1, &vp, w, h);
	GpuPass("scene");
	return 1;
}

// ---- iRender seams -----------------------------------------------------------------------------

void NukeDiligent::setOcclusionId(uint64_t id, const float mn[3], const float mx[3])
{
	if (!m_impl->occlPassActive || !id) { m_impl->occlPending = false; return; }
	m_impl->occlPending = true;
	m_impl->occlPendingTag.id = id;
	memcpy(m_impl->occlPendingTag.mn, mn, sizeof(float) * 3);
	memcpy(m_impl->occlPendingTag.mx, mx, sizeof(float) * 3);
}

void NukeDiligent::endOpaque()
{
	Impl& im = *m_impl;
	if (!im.occlPassActive) return;
	const int mode = im.OcclEndOpaque();
	if (mode != 0)
	{
		// The pyramid compute pass just rebound the context: every "already committed" cache is
		// stale, and a replayed draw whose material EQUALS the last drawn one would skip its
		// commit entirely (SRB-less DrawIndirect asserts).
		im.sceneCommitSrb = nullptr;
		im.matCBFor = nullptr;
		im.tessBindMat = nullptr;
		im.lastInstBind.pso = nullptr;
		// Replay the deferred draws — indirect (the test wrote the arguments) or plain when the
		// test could not run. Each record restores the overlay context its draw was tagged with.
		std::vector<Impl::OcclDeferred> defer; defer.swap(im.occlDeferred);
		for (size_t i = 0; i < defer.size(); ++i)
		{
			const Impl::OcclDeferred& d = defer[i];
			if (d.mat && d.liveSet)
			{
				d.mat->liveDrawSet = true;
				memcpy(d.mat->liveDrawValue, d.liveVal, sizeof(d.liveVal)); memcpy(d.mat->liveDrawMaskChan, d.liveChan, sizeof(d.liveChan));
				memcpy(d.mat->liveDrawMaskXform, d.liveXf, sizeof(d.liveXf)); d.mat->liveDrawMaskRes = d.liveRes; d.mat->liveDrawMask3D = d.liveMask;
			}
			im.occlReplay = mode == 1 ? (int)i : -1;
			if (d.instanced) renderObjectInstanced(d.mesh, d.mat, d.instBuf, d.first, d.count);
			else             RenderObjectRange(d.mesh, d.mat, d.pos, d.quat, d.scale, d.firstIndex, d.indexCount);
			im.occlReplay = -1;
		}
	}
	im.occlDeferred.clear(); im.occlTags.clear();
}

void NukeDiligent::setOcclusionCulling(bool enable, bool freeze)
{
	// Per-world flag: the editor flips it every frame (previews off, live world on), so the
	// history must survive a disable — views age out by lastUsed / frame on their own.
	Impl& im = *m_impl;
	im.occlEnabled = enable; im.occlFreeze = freeze;
}

void NukeDiligent::getOcclusionStats(int& tracked, int& culled)
{
	tracked = m_impl->statOcclTracked; culled = m_impl->statOcclCulled;
}
