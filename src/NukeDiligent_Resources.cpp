#include "NukeDiligentImpl.h"


// ---- centralized GPU-resource lifetime manager (see the header block for THE rule) ----------------
void NukeDiligent::Impl::Trash(IObject* o)
{
	if (!o) return;
	std::lock_guard<std::mutex> lk(trashMutex);
	gpuTrash.emplace_back(RefCntAutoPtr<IObject>(o), frameId);
}
void NukeDiligent::Impl::TrashRT(RT& rt)
{
	Trash(rt.color); Trash(rt.colorMS); Trash(rt.depth); Trash(rt.depthMS); Trash(rt.post);
}
void NukeDiligent::Impl::PurgeTrash(bool everything)
{
	std::lock_guard<std::mutex> lk(trashMutex);
	if (everything) { gpuTrash.clear(); return; }
	gpuTrash.erase(std::remove_if(gpuTrash.begin(), gpuTrash.end(),
		[&](const std::pair<RefCntAutoPtr<IObject>, uint64_t>& e) { return frameId - e.second > kTrashFrames; }),
		gpuTrash.end());
}
void NukeDiligent::Impl::EvictSized(std::unordered_map<uint64_t, SizedTexSet>& cache, uint64_t curKey)
{
	const size_t CAP = 6;   // main + previews + drag-resize transient slack
	while (cache.size() > CAP)
	{
		uint64_t lruKey = 0, lru = ~0ull; bool found = false;
		for (auto& kv : cache)
		{
			if (kv.first == curKey) continue;   // never evict the set in use this frame
			if (kv.second.lastUsed < lru) { lru = kv.second.lastUsed; lruKey = kv.first; found = true; }
		}
		if (!found) break;
		auto it = cache.find(lruKey);
		Trash(it->second.a); Trash(it->second.b);   // may have been used < kTrashFrames ago (drag-resize)
		cache.erase(it);
	}
}

ITextureView* NukeDiligent::Impl::GetTexSRV(Texture* t)
{
	if (!t) return nullptr;
	if (t->renderTexture)   // sample a RenderTexture = the camera's render-target color view
	{
		// Feedback guard: never sample the RT we're currently rendering INTO (an object that displays the
		// RT, caught in that camera's own view, would bind it as SRV while it's the RTV -> Diligent drops
		// the render target and the whole pass renders nothing but the clear color).
		if (t->rtId != 0 && t->rtId == curTarget) return nullptr;
		auto rit = rts.find(t->rtId);
		return rit != rts.end() ? rit->second.srv : nullptr;
	}
	if (t->pixels.empty() || t->width <= 0 || t->height <= 0) return nullptr;

	if (t->frameCount > 1)   // animated (GIF): a separate Texture2D per frame; return the current frame's SRV
	{
		auto av = animTex.find(t);
		if (av == animTex.end())
		{
			const int w = t->width, h = t->height, n = t->frameCount;
			const bool bc = (t->format == Texture::FMT_BC1 || t->format == Texture::FMT_BC3);
			const int  bb = (t->format == Texture::FMT_BC1) ? 8 : 16;
			const TEXTURE_FORMAT fmt = bc ? (t->format == Texture::FMT_BC1 ? TEX_FORMAT_BC1_UNORM : TEX_FORMAT_BC3_UNORM) : TEX_FORMAT_RGBA8_UNORM;
			const size_t frameBytes = bc ? (size_t)((w + 3) / 4) * ((h + 3) / 4) * bb : (size_t)w * h * 4;
			const Uint64 stride     = bc ? (Uint64)((w + 3) / 4) * bb : (Uint64)w * 4;
			if (t->pixels.size() < frameBytes * n) return nullptr;
			std::vector<RefCntAutoPtr<ITexture>> frames(n);
			for (int k = 0; k < n; ++k)
			{
				TextureDesc td; td.Type = RESOURCE_DIM_TEX_2D; td.Width = w; td.Height = h; td.MipLevels = 1;
				td.Format = fmt; td.BindFlags = BIND_SHADER_RESOURCE; td.Usage = USAGE_IMMUTABLE;
				TextureSubResData sub; sub.pData = t->pixels.data() + (size_t)k * frameBytes; sub.Stride = stride;
				TextureData data; data.pSubResources = &sub; data.NumSubresources = 1;
				device->CreateTexture(td, &data, &frames[k]);
			}
			av = animTex.emplace(t, std::move(frames)).first;
		}
		auto& fs = av->second;
		if (fs.empty()) return nullptr;
		int f = t->curFrame; if (f < 0) f = 0; if (f >= (int)fs.size()) f %= (int)fs.size();
		return fs[f] ? fs[f]->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE) : nullptr;
	}

	auto it = texCache.find(t);
	if (it != texCache.end())
		return it->second ? it->second->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE) : nullptr;
	{
		RefCntAutoPtr<ITexture> tex;
		if (t->format == Texture::FMT_BC1 || t->format == Texture::FMT_BC3 || t->format == Texture::FMT_BC5)
		{
			// Pre-compressed BC with a stored mip chain — upload every level (no GenerateMips for BC).
			const int  blockBytes = (t->format == Texture::FMT_BC1) ? 8 : 16;   // BC3/BC5 = 16
			const int  mips = t->mipCount < 1 ? 1 : t->mipCount;
			TextureDesc td; td.Type = RESOURCE_DIM_TEX_2D; td.Width = t->width; td.Height = t->height;
			td.MipLevels = mips; td.BindFlags = BIND_SHADER_RESOURCE; td.Usage = USAGE_IMMUTABLE;
			td.Format = (t->format == Texture::FMT_BC1) ? TEX_FORMAT_BC1_UNORM
			          : (t->format == Texture::FMT_BC5) ? TEX_FORMAT_BC5_UNORM : TEX_FORMAT_BC3_UNORM;
			std::vector<TextureSubResData> subs(mips);
			size_t off = 0; int mw = t->width, mh = t->height;
			for (int m = 0; m < mips; ++m)
			{
				int bx = (mw + 3) / 4, by = (mh + 3) / 4;
				subs[m].pData  = t->pixels.data() + off;
				subs[m].Stride = (Uint64)bx * blockBytes;
				off += (size_t)bx * by * blockBytes;
				mw = mw > 1 ? mw / 2 : 1; mh = mh > 1 ? mh / 2 : 1;
			}
			TextureData data; data.pSubResources = subs.data(); data.NumSubresources = (Uint32)mips;
			device->CreateTexture(td, &data, &tex);
		}
		else   // RGBA8 (non-BC fallback, e.g. odd sizes / GIF frames) — single level, no GPU mip-gen
		{
			if (t->pixels.size() < (size_t)t->width * t->height * 4) return nullptr;   // guard malformed data
			TextureDesc td; td.Type = RESOURCE_DIM_TEX_2D; td.Width = t->width; td.Height = t->height;
			td.MipLevels = 1; td.Format = TEX_FORMAT_RGBA8_UNORM;
			td.BindFlags = BIND_SHADER_RESOURCE; td.Usage = USAGE_IMMUTABLE;
			TextureSubResData sub; sub.pData = t->pixels.data(); sub.Stride = (Uint64)t->width * 4;
			TextureData data; data.pSubResources = &sub; data.NumSubresources = 1;
			device->CreateTexture(td, &data, &tex);
		}
		// NEVER cache a failed upload. A transient CreateTexture failure (e.g. GPU-memory pressure while a
		// huge inspector preview is resident) would otherwise blank this texture PERMANENTLY on every
		// consumer until the editor restarts. Return null now and retry next frame — the cache only ever
		// holds a live SRV, so a valid texture always self-heals.
		if (!tex)
		{
			std::cout << "[NukeDiligent]\tGetTexSRV upload FAILED (" << t->width << "x" << t->height
			          << " fmt " << t->format << " mips " << t->mipCount << ", " << t->pixels.size()
			          << " bytes) — GPU resource pressure; retrying next frame." << std::endl;
			return nullptr;
		}
		it = texCache.emplace(t, tex).first;
	}
	return it->second ? it->second->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE) : nullptr;
}

NukeDiligent::Impl::RT NukeDiligent::Impl::MakeRT(int w, int h)
{
	RT rt; rt.w = w; rt.h = h;
	const bool ms = samples > 1;

	// HDR (RGBA16F) color: geometry target (no MSAA) / resolve destination (MSAA). The post pass reads it.
	TextureDesc cd;
	cd.Name = "RT Color HDR"; cd.Type = RESOURCE_DIM_TEX_2D; cd.Width = (Uint32)w; cd.Height = (Uint32)h;
	cd.Format = SceneFmt(); cd.BindFlags = BIND_RENDER_TARGET | BIND_SHADER_RESOURCE;   // RGBA16F (HDR) or RGBA8 (off)
	device->CreateTexture(cd, nullptr, &rt.color);
	if (rt.color) rt.hdrSRV = rt.color->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);

	// LDR (RGBA8) post output — the tonemapped result the UI shows / a material samples.
	TextureDesc pd;
	pd.Name = "RT Color Post"; pd.Type = RESOURCE_DIM_TEX_2D; pd.Width = (Uint32)w; pd.Height = (Uint32)h;
	pd.Format = TEX_FORMAT_RGBA8_UNORM; pd.BindFlags = BIND_RENDER_TARGET | BIND_SHADER_RESOURCE;
	device->CreateTexture(pd, nullptr, &rt.post);
	if (rt.post) { rt.postRTV = rt.post->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET); rt.srv = rt.post->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE); }

	if (ms)
	{
		TextureDesc cm = cd; cm.Name = "RT Color HDR MS"; cm.SampleCount = samples; cm.BindFlags = BIND_RENDER_TARGET;
		device->CreateTexture(cm, nullptr, &rt.colorMS);
		if (rt.colorMS) rt.rtv = rt.colorMS->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET);
		TextureDesc dm; dm.Name = "RT Depth MS"; dm.Type = RESOURCE_DIM_TEX_2D; dm.Width = (Uint32)w; dm.Height = (Uint32)h;
		dm.Format = TEX_FORMAT_D32_FLOAT; dm.BindFlags = BIND_DEPTH_STENCIL; dm.SampleCount = samples;
		device->CreateTexture(dm, nullptr, &rt.depthMS);
		if (rt.depthMS) rt.dsv = rt.depthMS->GetDefaultView(TEXTURE_VIEW_DEPTH_STENCIL);
	}
	else
	{
		if (rt.color) rt.rtv = rt.color->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET);
		TextureDesc dd; dd.Name = "RT Depth"; dd.Type = RESOURCE_DIM_TEX_2D; dd.Width = (Uint32)w; dd.Height = (Uint32)h;
		dd.Format = TEX_FORMAT_D32_FLOAT; dd.BindFlags = BIND_DEPTH_STENCIL;
		device->CreateTexture(dd, nullptr, &rt.depth);
		if (rt.depth) rt.dsv = rt.depth->GetDefaultView(TEXTURE_VIEW_DEPTH_STENCIL);
	}
	return rt;
}

// (Re)create the HDR intermediate used when a camera renders to target 0 (the Player's backbuffer path):
// geometry -> this HDR target (MS if enabled) -> resolve -> HDR single -> post pass -> the swap-chain backbuffer.
// Reuses MakeRT (its unused `post` LDR texture is harmless; the post pass writes the real backbuffer instead).
void NukeDiligent::Impl::EnsureBackbufferMS(int w, int h)
{
	if (w <= 0 || h <= 0) return;
	if (backbufferMS.color && backbufferMS.w == w && backbufferMS.h == h) return;
	TrashRT(backbufferMS);   // a window-resize replaces it mid-loop; the old targets may be in flight
	backbufferMS = MakeRT(w, h);
}

NukeDiligent::Impl::MeshGPU* NukeDiligent::Impl::GetMeshGPU(Mesh* mesh)
{
	if (!mesh || mesh->numVerts <= 0) return nullptr;
	auto it = meshCache.find(mesh);
	if (it == meshCache.end())
	{
		if (!mesh->vertexArray || !mesh->normalArray)   // immutable buffers need init data
		{
			std::cout << "[NukeDiligent]\tmesh '" << mesh->name << "' has null vertex/normal data (numVerts="
			          << mesh->numVerts << ") — skipping" << std::endl;
			return nullptr;
		}
		// Build (and cache) GPU vertex buffers for this mesh: positions + normals + uv.
		// DEFAULT usage (not immutable): dynamic meshes (skinned instances, procedural)
		// re-upload in place below when Mesh::version changes.
		MeshGPU g; g.numVerts = mesh->numVerts; g.version = mesh->version;
		const Uint64 sz3 = (Uint64)mesh->numVerts * 3 * sizeof(float);
		const Uint64 sz2 = (Uint64)mesh->numVerts * 2 * sizeof(float);
		BufferDesc bd; bd.BindFlags = BIND_VERTEX_BUFFER; bd.Usage = USAGE_DEFAULT;
		// Positions double as BLAS geometry under D3D12 ray tracing -> they need BIND_RAY_TRACING too.
		BufferDesc pbd = bd; if (rtSupported) pbd.BindFlags = BIND_VERTEX_BUFFER | BIND_RAY_TRACING;
		pbd.Size = sz3; pbd.Name = "mesh pos"; BufferData pdat{mesh->vertexArray, sz3}; device->CreateBuffer(pbd, &pdat, &g.pos);
		bd.Size = sz3; bd.Name = "mesh nrm"; BufferData ndat{mesh->normalArray, sz3}; device->CreateBuffer(bd, &ndat, &g.nrm);
		std::vector<float> zeroUV;
		const float* uvSrc = mesh->uvArray;
		if (!uvSrc) { zeroUV.assign((size_t)mesh->numVerts * 2, 0.0f); uvSrc = zeroUV.data(); }   // mesh has no UVs
		bd.Size = sz2; bd.Name = "mesh uv"; BufferData udat{uvSrc, sz2}; device->CreateBuffer(bd, &udat, &g.uv);
		it = meshCache.emplace(mesh, std::move(g)).first;
	}
	MeshGPU& g = it->second;
	if (!g.pos || !g.nrm || !g.uv) return nullptr;
	if (g.version != mesh->version)
	{
		if (g.numVerts != mesh->numVerts)   // topology changed: rebuild from scratch
		{
			Trash(g.pos); Trash(g.nrm); Trash(g.uv);   // this frame's earlier draws may reference them
			meshCache.erase(it);
			auto bit = blasCache.find(mesh);           // BLAS built over the OLD pos buffer -> stale + dangling
			if (bit != blasCache.end()) { Trash(bit->second); blasCache.erase(bit); }
			return GetMeshGPU(mesh);
		}
		const Uint64 sz3 = (Uint64)mesh->numVerts * 3 * sizeof(float);
		if (mesh->vertexArray) context->UpdateBuffer(g.pos, 0, sz3, mesh->vertexArray, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		if (mesh->normalArray) context->UpdateBuffer(g.nrm, 0, sz3, mesh->normalArray, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		g.version = mesh->version;
	}
	return &g;
}
void NukeDiligent::bindRenderTarget(uint64_t id)
{
	if (id == 0) { m_impl->uiRTV = nullptr; m_impl->uiTW = m_impl->uiTH = 0; return; }
	auto it = m_impl->rts.find(id);
	if (it == m_impl->rts.end()) { m_impl->uiRTV = nullptr; m_impl->uiTW = m_impl->uiTH = 0; return; }
	m_impl->uiRTV = it->second.rtv; m_impl->uiTW = (Uint32)it->second.w; m_impl->uiTH = (Uint32)it->second.h;
}
void NukeDiligent::invalidateTexture(Texture* t)   // re-uploaded on next GetTexSRV
{
	if (!t) return;
	// The old SRV pointer may still sit in UI draw data recorded this frame (thumbnails, GUI images)
	// and in per-frame bindless tables — park, don't free inline.
	auto it = m_impl->texCache.find(t);
	if (it != m_impl->texCache.end()) { m_impl->Trash(it->second); m_impl->texCache.erase(it); }
	auto at = m_impl->animTex.find(t);
	if (at != m_impl->animTex.end())
	{
		for (auto& f : at->second) m_impl->Trash(f);
		m_impl->animTex.erase(at);
	}
}

void NukeDiligent::invalidateMesh(Mesh* m)
{
	if (!m) return;
	auto it = m_impl->meshCache.find(m);
	if (it != m_impl->meshCache.end())
	{
		m_impl->Trash(it->second.pos); m_impl->Trash(it->second.nrm); m_impl->Trash(it->second.uv);
		m_impl->meshCache.erase(it);
	}
	// The BLAS references the OLD pos buffer's GPU memory — after the buffers go, a cached BLAS would
	// make the TLAS trace freed memory (device removed). Rebuilt lazily from the new buffers.
	auto bit = m_impl->blasCache.find(m);
	if (bit != m_impl->blasCache.end()) { m_impl->Trash(bit->second); m_impl->blasCache.erase(bit); }
}

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
	if (!tex)
	{
		// Transient GPU-memory pressure (boot-time uploads, parallel tooling): callers
		// treat 0 as "retry later", so say WHY a frame degraded instead of dying silently.
		std::cout << "[NukeDiligent]\tcreateTexture2D FAILED (" << width << "x" << height
		          << ") — GPU resource pressure; UI retries next frame." << std::endl;
		return 0;
	}
	ITextureView* view = tex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
	uint64_t handle = reinterpret_cast<uint64_t>(view);
	m_impl->textures[handle] = tex; // keep alive
	return handle;
}

void NukeDiligent::destroyTexture2D(uint64_t handle)
{
	// Centralized lifetime: the view pointer may still sit in UI draw data recorded this frame (any
	// window, incl. detached OS viewports) — park the texture + its cached SRB, never free inline.
	auto sit = m_impl->uiSRBCache.find(reinterpret_cast<ITextureView*>(handle));
	if (sit != m_impl->uiSRBCache.end()) { m_impl->Trash(sit->second.srb); m_impl->uiSRBCache.erase(sit); }
	auto it = m_impl->textures.find(handle);
	if (it != m_impl->textures.end()) { m_impl->Trash(it->second); m_impl->textures.erase(it); }
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
	// Resize happens MID-FRAME (a panel resizing during the UI pass): the old post SRV may already be
	// recorded in this frame's draw lists, and this frame's world pass wrote into the old RTV. Park
	// everything; drop the old SRV's cached UI SRB (keyed by the now-parked view pointer).
	Impl::RT old = it->second;
	auto sit = m_impl->uiSRBCache.find(old.srv);
	if (sit != m_impl->uiSRBCache.end()) { m_impl->Trash(sit->second.srb); m_impl->uiSRBCache.erase(sit); }
	it->second = m_impl->MakeRT(w, h);
	m_impl->TrashRT(old);
}

uint64_t NukeDiligent::getRenderTargetTexture(uint64_t id)
{
	auto it = m_impl->rts.find(id);
	return (it == m_impl->rts.end()) ? 0 : reinterpret_cast<uint64_t>(it->second.srv);
}
