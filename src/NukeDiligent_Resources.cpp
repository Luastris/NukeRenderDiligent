#include "NukeDiligentImpl.h"


// Centralized GPU-resource lifetime manager: park objects here, never Release() them inline.
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
		// Feedback guard: binding the RT we render into as an SRV makes Diligent drop the render target.
		if (t->rtId != 0 && t->rtId == curTarget) return nullptr;
		auto rit = rts.find(t->rtId);
		return rit != rts.end() ? rit->second.srv : nullptr;
	}
	if (!t->HasPixelData() || t->width <= 0 || t->height <= 0) return nullptr;

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
		// T3 streaming: a world-drawn (touched) BC texture is born at its DISTANCE-desired mip
		// range; the pump re-targets it later. Untouched textures (UI, previews) stay full.
		int base = 0;
		if (streamBudget > 0 && StreamEligible(t))
		{
			auto st = streamTex.find(t);
			if (st != streamTex.end())
				base = st->second.residentBase = StreamDesiredBase(t, st->second.lastDist);
		}
		// Pak-resident pixels: DirectStorage streams them into VRAM — this draw goes without
		// the texture, the adopter publishes it when it lands. No provider / a failed pak read:
		// pull the pixels on the CPU and upload as always.
		if (t->pixels.empty() && t->pakSource)
		{
			if (dstor && !storFailed.count(t))
			{
				if (!storPendingTex.count(t) && StorageRequestTex(t, base, false)) storPendingTex.insert(t);
				return nullptr;
			}
			if (!t->EnsurePixels()) return nullptr;
		}
		RefCntAutoPtr<ITexture> tex = CreateEngineTex(t, base);
		// Never cache a failed upload: the cache must only ever hold a live SRV, so it self-heals on retry.
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

// One engine texture -> one GPU texture, uploading mips [baseMip..last] (BC) or the single
// RGBA8 level. baseMip shrinks the OBJECT — normalized UVs keep shaders oblivious.
RefCntAutoPtr<ITexture> NukeDiligent::Impl::CreateEngineTex(Texture* t, int baseMip)
{
	RefCntAutoPtr<ITexture> tex;
	if (t->format == Texture::FMT_BC1 || t->format == Texture::FMT_BC3 || t->format == Texture::FMT_BC5)
	{
		// Pre-compressed BC with a stored mip chain — upload the resident levels (no GenerateMips for BC).
		const int  blockBytes = (t->format == Texture::FMT_BC1) ? 8 : 16;   // BC3/BC5 = 16
		const int  mips = t->mipCount < 1 ? 1 : t->mipCount;
		if (baseMip < 0) baseMip = 0;
		if (baseMip > mips - 1) baseMip = mips - 1;
		baseMip = AlignedBase(t, baseMip);
		TextureDesc td; td.Type = RESOURCE_DIM_TEX_2D;
		td.Width  = (Uint32)std::max(1, t->width  >> baseMip);
		td.Height = (Uint32)std::max(1, t->height >> baseMip);
		td.MipLevels = mips - baseMip; td.BindFlags = BIND_SHADER_RESOURCE; td.Usage = USAGE_IMMUTABLE;
		td.Format = (t->format == Texture::FMT_BC1) ? TEX_FORMAT_BC1_UNORM
		          : (t->format == Texture::FMT_BC5) ? TEX_FORMAT_BC5_UNORM : TEX_FORMAT_BC3_UNORM;
		std::vector<TextureSubResData> subs((size_t)mips - baseMip);
		size_t off = 0; int mw = t->width, mh = t->height;
		for (int m = 0; m < mips; ++m)
		{
			int bx = (mw + 3) / 4, by = (mh + 3) / 4;
			if (m >= baseMip)
			{
				subs[m - baseMip].pData  = t->pixels.data() + off;
				subs[m - baseMip].Stride = (Uint64)bx * blockBytes;
			}
			off += (size_t)bx * by * blockBytes;
			mw = mw > 1 ? mw / 2 : 1; mh = mh > 1 ? mh / 2 : 1;
		}
		if (off > t->pixels.size()) return tex;   // malformed chain
		TextureData data; data.pSubResources = subs.data(); data.NumSubresources = (Uint32)subs.size();
		device->CreateTexture(td, &data, &tex);
	}
	else   // RGBA8 (non-BC fallback, e.g. odd sizes / GIF frames) — single level, no GPU mip-gen
	{
		if (t->pixels.size() < (size_t)t->width * t->height * 4) return tex;   // guard malformed data
		TextureDesc td; td.Type = RESOURCE_DIM_TEX_2D; td.Width = t->width; td.Height = t->height;
		td.MipLevels = 1; td.Format = TEX_FORMAT_RGBA8_UNORM;
		td.BindFlags = BIND_SHADER_RESOURCE; td.Usage = USAGE_IMMUTABLE;
		TextureSubResData sub; sub.pData = t->pixels.data(); sub.Stride = (Uint64)t->width * 4;
		TextureData data; data.pSubResources = &sub; data.NumSubresources = 1;
		device->CreateTexture(td, &data, &tex);
	}
	return tex;
}

// ---- T3 texture streaming -----------------------------------------------------------------------

// Streamable: BC with a real mip chain and enough levels that dropping some actually saves
// memory; animated frames, render textures and tiny maps never stream.
bool NukeDiligent::Impl::StreamEligible(Texture* t) const
{
	if (!t || t->renderTexture || t->frameCount > 1) return false;
	if (t->format != Texture::FMT_BC1 && t->format != Texture::FMT_BC3 && t->format != Texture::FMT_BC5) return false;
	return t->mipCount >= 4 && std::max(t->width, t->height) >= 256;
}

// A BC resource's top level must be a multiple of 4 on D3D12/Vulkan; an NPOT chain's lower
// levels (4000 -> 250 -> 125 -> 62) cannot head a resource. Walk back to a level that can.
int NukeDiligent::Impl::AlignedBase(Texture* t, int base)
{
	if (t->format != Texture::FMT_BC1 && t->format != Texture::FMT_BC3 && t->format != Texture::FMT_BC5) return base;
	while (base > 0 && (((t->width >> base) & 3) || ((t->height >> base) & 3))) --base;
	return base;
}

// The always-resident tail: first mip whose larger dimension is <= 64.
int NukeDiligent::Impl::StreamTailBase(Texture* t)
{
	int base = 0, dim = std::max(t->width, t->height);
	while (dim > 64 && base < t->mipCount - 1) { dim >>= 1; ++base; }
	return AlignedBase(t, base);
}

long long NukeDiligent::Impl::StreamBytes(Texture* t, int base)
{
	const int blockBytes = (t->format == Texture::FMT_BC1) ? 8 : 16;
	long long bytes = 0; int mw = t->width, mh = t->height;
	for (int m = 0; m < t->mipCount; ++m)
	{
		if (m >= base) bytes += (long long)((mw + 3) / 4) * ((mh + 3) / 4) * blockBytes;
		mw = mw > 1 ? mw / 2 : 1; mh = mh > 1 ? mh / 2 : 1;
	}
	return bytes;
}

// Distance -> first-resident mip: full res inside kFullResDist, one level per doubling after.
int NukeDiligent::Impl::StreamDesiredBase(Texture* t, float dist) const
{
	static constexpr float kFullResDist = 24.0f;
	int b = 0; float d = dist;
	while (d > kFullResDist && b < 30) { d *= 0.5f; ++b; }
	const int tail = StreamTailBase(t);
	return AlignedBase(t, b > tail ? tail : b);
}

// Per-draw feedback from the world passes: the nearest use this frame drives residency.
void NukeDiligent::Impl::StreamTouch(Texture* t, float dist)
{
	if (streamBudget <= 0 || !StreamEligible(t)) return;
	StreamTex& s = streamTex[t];
	if (dist < s.minDist) s.minDist = dist;
	if (dist < s.lastDist) s.lastDist = dist;   // a fresh entry gets a real distance BEFORE the first pump
	s.lastTouch = frameId;
}

void NukeDiligent::Impl::StreamPump()
{
	if (streamTex.empty()) return;

	// Merge this frame's feedback; long-untouched textures decay to the tail.
	static constexpr uint64_t kDecayFrames = 600;
	struct Cand { Texture* t; StreamTex* s; int target; };
	std::vector<Cand> cands;
	cands.reserve(streamTex.size());
	long long wantBytes = 0;
	for (auto& kv : streamTex)
	{
		StreamTex& s = kv.second;
		if (s.minDist < 1e29f) { s.lastDist = s.minDist; s.minDist = 1e30f; }
		int target;
		if (streamBudget <= 0)
			target = 0;   // streaming turned off live: walk everything back to full res
		else if (frameId - s.lastTouch > kDecayFrames)
			target = StreamTailBase(kv.first);
		else
			target = StreamDesiredBase(kv.first, s.lastDist);
		cands.push_back(Cand{ kv.first, &s, target });
		wantBytes += StreamBytes(kv.first, target);
	}

	// Over budget: push the FARTHEST textures to their tails until the target set fits.
	if (streamBudget > 0 && wantBytes > streamBudget)
	{
		std::sort(cands.begin(), cands.end(),
		          [](const Cand& a, const Cand& b) { return a.s->lastDist > b.s->lastDist; });
		for (Cand& c : cands)
		{
			if (wantBytes <= streamBudget) break;
			const int tail = StreamTailBase(c.t);
			if (c.target >= tail) continue;
			wantBytes -= StreamBytes(c.t, c.target) - StreamBytes(c.t, tail);
			c.target = tail;
		}
	}

	// Rebuilds, bounded per frame: upgrades (more detail) go immediately, nearest first;
	// downgrades wait ~2s of stable lower need (no flapping at ring boundaries).
	static constexpr int kMaxRebuilds = 6;
	static constexpr long long kMaxUploadBytes = 24ll << 20;
	std::sort(cands.begin(), cands.end(),
	          [](const Cand& a, const Cand& b) { return a.s->lastDist < b.s->lastDist; });
	int rebuilds = 0; long long uploaded = 0;
	for (Cand& c : cands)
	{
		StreamTex& s = *c.s;
		if (c.target == s.residentBase) { s.wantLowerFrames = 0; continue; }
		if (c.target > s.residentBase)   // dropping detail
		{
			if (++s.wantLowerFrames < 120) continue;
		}
		else s.wantLowerFrames = 0;
		if (rebuilds >= kMaxRebuilds || uploaded >= kMaxUploadBytes) continue;
		auto tc = texCache.find(c.t);
		if (tc == texCache.end()) { s.residentBase = c.target; continue; }   // not on GPU yet: born at target
		if (c.t->pixels.empty() && c.t->pakSource && dstor && !storFailed.count(c.t))
		{
			// Re-target through DirectStorage: the old object stays bound until the new one lands.
			if (storPendingTex.count(c.t)) continue;
			if (!StorageRequestTex(c.t, c.target, false)) continue;
			storPendingTex.insert(c.t);
			uploaded += StreamBytes(c.t, c.target);
			++rebuilds;
			s.residentBase = c.target;
			s.wantLowerFrames = 0;
			continue;
		}
		if (c.t->pixels.empty() && !c.t->EnsurePixels()) continue;
		RefCntAutoPtr<ITexture> fresh = CreateEngineTex(c.t, c.target);
		if (!fresh) continue;
		Trash(tc->second);
		tc->second = fresh;
		uploaded += StreamBytes(c.t, c.target);
		++rebuilds;
		s.residentBase = c.target;
		s.wantLowerFrames = 0;
	}

	// Stats by recomputation (incremental deltas drift across invalidate/reload cycles).
	streamResident = 0; streamFullBytes = 0;
	for (auto& kv : streamTex)
	{
		streamResident  += StreamBytes(kv.first, kv.second.residentBase);
		streamFullBytes += StreamBytes(kv.first, 0);
	}
}

void NukeDiligent::setTextureStreaming(long long budgetBytes)
{
	m_impl->streamBudget = budgetBytes < 0 ? 0 : budgetBytes;
	std::cout << "[NukeDiligent]\ttexture streaming budget = "
	          << (m_impl->streamBudget >> 20) << " MB" << std::endl;
}

void NukeDiligent::textureStreamInfo(long long& residentBytes, long long& savedBytes, int& streamedCount)
{
	residentBytes = m_impl->streamResident;
	savedBytes = m_impl->streamFullBytes - m_impl->streamResident;
	if (savedBytes < 0) savedBytes = 0;
	streamedCount = (int)m_impl->streamTex.size();
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
		// Sampleable: the Hi-Z occlusion pyramid reads the farthest sample per pixel at endOpaque.
		dm.Format = TEX_FORMAT_D32_FLOAT; dm.BindFlags = BIND_DEPTH_STENCIL | BIND_SHADER_RESOURCE; dm.SampleCount = samples;
		device->CreateTexture(dm, nullptr, &rt.depthMS);
		if (rt.depthMS) rt.dsv = rt.depthMS->GetDefaultView(TEXTURE_VIEW_DEPTH_STENCIL);
	}
	else
	{
		if (rt.color) rt.rtv = rt.color->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET);
		TextureDesc dd; dd.Name = "RT Depth"; dd.Type = RESOURCE_DIM_TEX_2D; dd.Width = (Uint32)w; dd.Height = (Uint32)h;
		dd.Format = TEX_FORMAT_D32_FLOAT; dd.BindFlags = BIND_DEPTH_STENCIL | BIND_SHADER_RESOURCE;   // Hi-Z source
		device->CreateTexture(dd, nullptr, &rt.depth);
		if (rt.depth) rt.dsv = rt.depth->GetDefaultView(TEXTURE_VIEW_DEPTH_STENCIL);
	}
	return rt;
}

// (Re)create the HDR intermediate for cameras rendering to target 0: geometry -> HDR target
// (MS if enabled) -> resolve -> post pass -> swap-chain backbuffer.
void NukeDiligent::Impl::EnsureBackbufferMS(int w, int h)
{
	if (w <= 0 || h <= 0) return;
	if (backbufferMS.color && backbufferMS.w == w && backbufferMS.h == h) return;
	TrashRT(backbufferMS);   // a window-resize replaces it mid-loop; the old targets may be in flight
	backbufferMS = MakeRT(w, h);
}

// ---- pooled mesh streams (TB-5) -----------------------------------------------------------------

bool NukeDiligent::Impl::PoolAlloc(std::map<uint32_t, uint32_t>& fm, uint32_t count, uint32_t& off)
{
	for (auto it = fm.begin(); it != fm.end(); ++it)
		if (it->second >= count)
		{
			off = it->first;
			const uint32_t rem = it->second - count;
			const uint32_t rOff = it->first + count;
			fm.erase(it);
			if (rem) fm[rOff] = rem;
			return true;
		}
	return false;
}

void NukeDiligent::Impl::PoolFree(std::map<uint32_t, uint32_t>& fm, uint32_t off, uint32_t count)
{
	if (!count) return;
	auto next = fm.lower_bound(off);
	// Coalesce with the previous block...
	if (next != fm.begin())
	{
		auto prev = std::prev(next);
		if (prev->first + prev->second == off) { off = prev->first; count += prev->second; fm.erase(prev); }
	}
	// ...and with the following one.
	if (next != fm.end() && off + count == next->first) { count += next->second; fm.erase(next); }
	fm[off] = count;
}

NukeDiligent::Impl::PoolArena* NukeDiligent::Impl::PoolAllocMesh(uint32_t verts, uint32_t inds,
                                                                 uint32_t& vOff, uint32_t& iOff)
{
	static constexpr uint32_t kArenaVerts = 1u << 20;   // 48 MB of streams per arena
	static constexpr uint32_t kArenaIdx   = 4u << 20;   // 16 MB of indices
	if (verts > kArenaVerts || inds > kArenaIdx) return nullptr;   // absurd mesh: dedicated path
	for (auto& a : meshPool)
	{
		uint32_t v, i;
		if (!PoolAlloc(a->vFree, verts, v)) continue;
		if (!PoolAlloc(a->iFree, inds, i)) { PoolFree(a->vFree, v, verts); continue; }
		vOff = v; iOff = i;
		return a.get();
	}
	// Grow: one more arena.
	auto a = std::make_unique<PoolArena>();
	a->vertCap = kArenaVerts;
	a->idxCap = kArenaIdx;
	BufferDesc bd; bd.Usage = USAGE_DEFAULT; bd.BindFlags = BIND_VERTEX_BUFFER;
	BufferDesc pbd = bd; if (rtSupported) pbd.BindFlags = BIND_VERTEX_BUFFER | BIND_RAY_TRACING;
	pbd.Size = (Uint64)kArenaVerts * 12; pbd.Name = "mesh pool pos"; device->CreateBuffer(pbd, nullptr, &a->pos);
	bd.Size = (Uint64)kArenaVerts * 12; bd.Name = "mesh pool nrm"; device->CreateBuffer(bd, nullptr, &a->nrm);
	{
		// The uv stream must READ as zeros (pooled meshes carry no uvs) — init it explicitly.
		std::vector<float> zero((size_t)kArenaVerts * 2, 0.0f);
		bd.Size = (Uint64)kArenaVerts * 8; bd.Name = "mesh pool uv";
		BufferData zd{ zero.data(), bd.Size };
		device->CreateBuffer(bd, &zd, &a->uv);
	}
	bd.Size = (Uint64)kArenaVerts * 16; bd.Name = "mesh pool col"; device->CreateBuffer(bd, nullptr, &a->col);
	BufferDesc ib; ib.Usage = USAGE_DEFAULT; ib.Name = "mesh pool idx";
	ib.BindFlags = rtSupported ? (BIND_INDEX_BUFFER | BIND_RAY_TRACING) : BIND_INDEX_BUFFER;
	ib.Size = (Uint64)kArenaIdx * 4;
	device->CreateBuffer(ib, nullptr, &a->idx);
	if (!a->pos || !a->nrm || !a->uv || !a->col || !a->idx) return nullptr;
	a->vFree[0] = kArenaVerts;
	a->iFree[0] = kArenaIdx;
	meshPool.push_back(std::move(a));
	PoolArena* pa = meshPool.back().get();
	uint32_t v, i;
	if (!PoolAlloc(pa->vFree, verts, v) || !PoolAlloc(pa->iFree, inds, i)) return nullptr;
	vOff = v; iOff = i;
	std::cout << "[NukeDiligent]\tmesh pool arena #" << meshPool.size() << " (64 MB)" << std::endl;
	return pa;
}

NukeDiligent::Impl::MeshGPU* NukeDiligent::Impl::GetMeshGPU(Mesh* mesh)
{
	if (!mesh || mesh->numVerts <= 0) return nullptr;
	auto it = meshCache.find(mesh);
	// Pooled residency (Mesh::pooled): plain static indexed geometry lives as arena ranges —
	// the whole (re)serve cycle allocates and uploads, never creates GPU objects.
	const bool wantPool = mesh->pooled && mesh->indexArray && mesh->numIndices > 0
	                   && mesh->vertexArray && mesh->normalArray && mesh->colorArray
	                   && !skinRecs.count(mesh) && !mesh->rtBendArray;
	if (it == meshCache.end() && wantPool)
	{
		MeshGPU g;
		g.numVerts = mesh->numVerts;
		g.numIndices = mesh->numIndices;
		g.version = mesh->version;
		g.arena = PoolAllocMesh((uint32_t)mesh->numVerts, (uint32_t)mesh->numIndices, g.vOff, g.iOff);
		if (g.arena)
		{
			const Uint64 sz3 = (Uint64)mesh->numVerts * 3 * sizeof(float);
			context->UpdateBuffer(g.arena->pos, g.PosOfs(), sz3, mesh->vertexArray, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
			context->UpdateBuffer(g.arena->nrm, g.NrmOfs(), sz3, mesh->normalArray, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
			context->UpdateBuffer(g.arena->col, g.ColOfs(), (Uint64)mesh->numVerts * 16, mesh->colorArray, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
			context->UpdateBuffer(g.arena->idx, g.IdxOfs(), (Uint64)mesh->numIndices * 4, mesh->indexArray, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
			it = meshCache.emplace(mesh, std::move(g)).first;
			return &it->second;
		}
		// Pool refused (absurd size / device OOM): the dedicated path below still works.
	}
	if (it == meshCache.end())
	{
		if (!mesh->vertexArray || !mesh->normalArray)   // immutable buffers need init data
		{
			std::cout << "[NukeDiligent]\tmesh '" << mesh->name << "' has null vertex/normal data (numVerts="
			          << mesh->numVerts << ") — skipping" << std::endl;
			return nullptr;
		}
		// USAGE_DEFAULT, not immutable: dynamic meshes re-upload in place when Mesh::version changes.
		MeshGPU g; g.numVerts = mesh->numVerts; g.version = mesh->version;
		const Uint64 sz3 = (Uint64)mesh->numVerts * 3 * sizeof(float);
		const Uint64 sz2 = (Uint64)mesh->numVerts * 2 * sizeof(float);
		BufferDesc bd; bd.BindFlags = BIND_VERTEX_BUFFER; bd.Usage = USAGE_DEFAULT;
		// Positions double as BLAS geometry under D3D12 ray tracing -> they need BIND_RAY_TRACING too.
		BufferDesc pbd = bd; if (rtSupported) pbd.BindFlags = BIND_VERTEX_BUFFER | BIND_RAY_TRACING;
		// GPU-skinned INSTANCE: pos/nrm are compute OUTPUTS (structured UAV that still binds as
		// the draw VB / BLAS input), plus the previous-frame positions for TAA velocity.
		g.skinned = skinRecs.count(mesh) != 0;
		if (g.skinned)
		{
			pbd.Mode = BUFFER_MODE_STRUCTURED; pbd.ElementByteStride = sizeof(float);
			pbd.BindFlags = BIND_VERTEX_BUFFER | BIND_UNORDERED_ACCESS | BIND_SHADER_RESOURCE
			              | (rtSupported ? BIND_RAY_TRACING : BIND_NONE);
			bd.Mode = BUFFER_MODE_STRUCTURED; bd.ElementByteStride = sizeof(float);
			bd.BindFlags = BIND_VERTEX_BUFFER | BIND_UNORDERED_ACCESS | BIND_SHADER_RESOURCE;
		}
		pbd.Size = sz3; pbd.Name = "mesh pos"; BufferData pdat{mesh->vertexArray, sz3}; device->CreateBuffer(pbd, &pdat, &g.pos);
		bd.Size = sz3; bd.Name = "mesh nrm"; BufferData ndat{mesh->normalArray, sz3}; device->CreateBuffer(bd, &ndat, &g.nrm);
		if (g.skinned)
		{
			BufferDesc ppd = bd; ppd.Name = "skin prev pos";
			BufferData ppdat{mesh->vertexArray, sz3};   // first frame: prev == bind
			device->CreateBuffer(ppd, &ppdat, &g.skinPosPrev);
			bd.Mode = BUFFER_MODE_UNDEFINED; bd.ElementByteStride = 0; bd.BindFlags = BIND_VERTEX_BUFFER;   // uv below stays a plain VB
		}
		std::vector<float> zeroUV;
		const float* uvSrc = mesh->uvArray;
		if (!uvSrc) { zeroUV.assign((size_t)mesh->numVerts * 2, 0.0f); uvSrc = zeroUV.data(); }   // mesh has no UVs
		bd.Size = sz2; bd.Name = "mesh uv"; BufferData udat{uvSrc, sz2}; device->CreateBuffer(bd, &udat, &g.uv);
		if (mesh->colorArray)   // optional RGBA vertex colors (terrain splat weights etc.), slot 3
		{
			const Uint64 sz4c = (Uint64)mesh->numVerts * 4 * sizeof(float);
			bd.Size = sz4c; bd.Name = "mesh col"; BufferData cdat{mesh->colorArray, sz4c};
			device->CreateBuffer(bd, &cdat, &g.col);
		}
		if (mesh->indexArray && mesh->numIndices > 0)   // v4 indexed mesh: index buffer (BLAS reads it too)
		{
			BufferDesc ib; ib.Usage = USAGE_DEFAULT; ib.Name = "mesh idx";
			ib.BindFlags = rtSupported ? (BIND_INDEX_BUFFER | BIND_RAY_TRACING) : BIND_INDEX_BUFFER;
			ib.Size = (Uint64)mesh->numIndices * sizeof(uint32_t);
			BufferData idat{mesh->indexArray, ib.Size};
			device->CreateBuffer(ib, &idat, &g.idx);
			g.numIndices = mesh->numIndices;
		}
		// RT wind bend: compute inputs + the bent position buffer the BLAS builds over. The separate
		// structured copy of the positions keeps the vertex buffer's bind flags untouched.
		if (rtSupported && mesh->rtBendArray && mesh->rtPivotArray)
		{
			const Uint64 sz4 = (Uint64)mesh->numVerts * 4 * sizeof(float);
			BufferDesc sb; sb.Usage = USAGE_DEFAULT; sb.Mode = BUFFER_MODE_STRUCTURED;
			sb.BindFlags = BIND_SHADER_RESOURCE; sb.ElementByteStride = sizeof(float);
			sb.Size = sz3; sb.Name = "bend src";
			BufferData sdat{mesh->vertexArray, sz3}; device->CreateBuffer(sb, &sdat, &g.bendSrc);
			sb.ElementByteStride = sizeof(float) * 4;
			sb.Size = sz4; sb.Name = "bend data";
			BufferData bdat2{mesh->rtBendArray, sz4}; device->CreateBuffer(sb, &bdat2, &g.bendData);
			sb.Name = "bend pivot";
			BufferData pdat2{mesh->rtPivotArray, sz4}; device->CreateBuffer(sb, &pdat2, &g.bendPivot);
			BufferDesc db; db.Usage = USAGE_DEFAULT; db.Mode = BUFFER_MODE_STRUCTURED;
			db.BindFlags = BIND_UNORDERED_ACCESS | BIND_RAY_TRACING; db.ElementByteStride = sizeof(float);
			db.Size = sz3; db.Name = "bend pos out";
			BufferData ddat{mesh->vertexArray, sz3};   // starts unbent -> BLAS valid before the first dispatch
			device->CreateBuffer(db, &ddat, &g.posBent);
		}
		it = meshCache.emplace(mesh, std::move(g)).first;
	}
	MeshGPU& g = it->second;
	if (!g.PosBuf() || !g.NrmBuf() || !g.UVBuf()) return nullptr;
	if (g.version != mesh->version)
	{
		if (g.numVerts != mesh->numVerts || g.numIndices != mesh->numIndices)   // topology changed: rebuild from scratch
		{
			// Pooled: the ranges return to the arena, the buffers themselves stay put.
			if (g.arena)
			{
				PoolFree(g.arena->vFree, g.vOff, (uint32_t)g.numVerts);
				PoolFree(g.arena->iFree, g.iOff, (uint32_t)g.numIndices);
			}
			// Park EVERYTHING the erase would inline-release — this frame's earlier draws
			// (and in-flight BLAS builds) may still reference the buffers.
			Trash(g.pos); Trash(g.nrm); Trash(g.uv); Trash(g.col); Trash(g.idx);
			Trash(g.bendSrc); Trash(g.bendData); Trash(g.bendPivot); Trash(g.posBent); Trash(g.blasScratch);
			Trash(g.skinPosPrev); Trash(g.skinSrcPos); Trash(g.skinSrcNrm);
			Trash(g.skinIdxBuf); Trash(g.skinWgtBuf); Trash(g.skinMorph);
			meshCache.erase(it);
			auto bit = blasCache.find(mesh);           // BLAS built over the OLD pos buffer -> stale + dangling
			if (bit != blasCache.end()) { Trash(bit->second); blasCache.erase(bit); }
			for (auto sit = blasSectionCache.lower_bound({mesh, 0ull});
			     sit != blasSectionCache.end() && sit->first.first == mesh; )
			{ Trash(sit->second); sit = blasSectionCache.erase(sit); }
			// RT attribute-pool slices were unrolled for the OLD topology.
			meshNrmByteOffset.erase(mesh); meshUVByteOffset.erase(mesh); meshPosByteOffset.erase(mesh);
			return GetMeshGPU(mesh);
		}
		const Uint64 sz3 = (Uint64)mesh->numVerts * 3 * sizeof(float);
		if (g.arena)
		{
			// Same-count re-serve of a pooled mesh: rewrite the ranges in place — including the
			// INDICES (a baked node's face-mask reselect can keep the count and change the list).
			if (mesh->vertexArray) context->UpdateBuffer(g.arena->pos, g.PosOfs(), sz3, mesh->vertexArray, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
			if (mesh->normalArray) context->UpdateBuffer(g.arena->nrm, g.NrmOfs(), sz3, mesh->normalArray, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
			if (mesh->colorArray)  context->UpdateBuffer(g.arena->col, g.ColOfs(), (Uint64)mesh->numVerts * 16, mesh->colorArray, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
			if (mesh->indexArray)  context->UpdateBuffer(g.arena->idx, g.IdxOfs(), (Uint64)mesh->numIndices * 4, mesh->indexArray, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
			// The BLAS geometry references the range contents: drop BOTH caches (indexed meshes
			// live in the SECTION cache) or RT keeps tracing the pre-rewrite triangles.
			auto bit = blasCache.find(mesh);
			if (bit != blasCache.end()) { Trash(bit->second); blasCache.erase(bit); }
			for (auto sit = blasSectionCache.lower_bound({mesh, 0ull});
			     sit != blasSectionCache.end() && sit->first.first == mesh; )
			{ Trash(sit->second); sit = blasSectionCache.erase(sit); }
			g.version = mesh->version;
			return &g;
		}
		if (mesh->vertexArray) context->UpdateBuffer(g.pos, 0, sz3, mesh->vertexArray, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		if (mesh->normalArray) context->UpdateBuffer(g.nrm, 0, sz3, mesh->normalArray, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		if (mesh->colorArray && g.col)
			context->UpdateBuffer(g.col, 0, (Uint64)mesh->numVerts * 4 * sizeof(float), mesh->colorArray, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		if (g.bendSrc && mesh->vertexArray)
		{
			context->UpdateBuffer(g.bendSrc, 0, sz3, mesh->vertexArray, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
			const Uint64 sz4 = (Uint64)mesh->numVerts * 4 * sizeof(float);
			if (g.bendData && mesh->rtBendArray)   context->UpdateBuffer(g.bendData, 0, sz4, mesh->rtBendArray, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
			if (g.bendPivot && mesh->rtPivotArray) context->UpdateBuffer(g.bendPivot, 0, sz4, mesh->rtPivotArray, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
			if (g.posBent) context->UpdateBuffer(g.posBent, 0, sz3, mesh->vertexArray, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		}
		g.version = mesh->version;
	}
	return &g;
}

// Lazy static compute inputs for a skin SOURCE mesh: structured bind streams + packed bone
// indices/weights + concatenated morph deltas ([pos 3n][nrm 3n] per target).
void NukeDiligent::Impl::EnsureSkinInputs(Mesh* source, MeshGPU& gs)
{
	if (gs.skinSrcPos || !source->boneIndex || !source->boneWeight || source->numVerts <= 0) return;
	const int n = source->numVerts;
	const Uint64 sz3 = (Uint64)n * 3 * sizeof(float);
	BufferDesc sb; sb.Usage = USAGE_DEFAULT; sb.Mode = BUFFER_MODE_STRUCTURED;
	sb.BindFlags = BIND_SHADER_RESOURCE;
	sb.ElementByteStride = sizeof(float);
	sb.Size = sz3; sb.Name = "skin src pos"; BufferData pd{source->vertexArray, sz3}; device->CreateBuffer(sb, &pd, &gs.skinSrcPos);
	sb.Name = "skin src nrm"; BufferData nd2{source->normalArray, sz3}; device->CreateBuffer(sb, &nd2, &gs.skinSrcNrm);
	std::vector<uint32_t> packed((size_t)n * 2);
	for (int v = 0; v < n; ++v)
	{
		const unsigned short* b = source->boneIndex + (size_t)v * 4;
		packed[(size_t)v * 2 + 0] = (uint32_t)b[0] | ((uint32_t)b[1] << 16);
		packed[(size_t)v * 2 + 1] = (uint32_t)b[2] | ((uint32_t)b[3] << 16);
	}
	sb.ElementByteStride = sizeof(uint32_t) * 2; sb.Size = (Uint64)n * 2 * sizeof(uint32_t);
	sb.Name = "skin bone idx"; BufferData id2{packed.data(), sb.Size}; device->CreateBuffer(sb, &id2, &gs.skinIdxBuf);
	sb.ElementByteStride = sizeof(float) * 4; sb.Size = (Uint64)n * 4 * sizeof(float);
	sb.Name = "skin bone wgt"; BufferData wd2{source->boneWeight, sb.Size}; device->CreateBuffer(sb, &wd2, &gs.skinWgtBuf);
	gs.skinMorphCount = (int)source->morphs.size();
	{
		const size_t per = (size_t)n * 6;
		std::vector<float> md(per * source->morphs.size(), 0.0f);
		if (md.empty()) md.resize(6, 0.0f);   // the SRB always binds a valid buffer
		for (size_t t = 0; t < source->morphs.size(); ++t)
		{
			const Mesh::MorphTarget& mt = source->morphs[t];
			if (mt.posDelta.size() == (size_t)n * 3) memcpy(md.data() + t * per, mt.posDelta.data(), sizeof(float) * 3 * n);
			if (mt.nrmDelta.size() == (size_t)n * 3) memcpy(md.data() + t * per + (size_t)n * 3, mt.nrmDelta.data(), sizeof(float) * 3 * n);
		}
		sb.ElementByteStride = sizeof(float); sb.Size = (Uint64)md.size() * sizeof(float);
		sb.Name = "skin morph deltas"; BufferData mdd{md.data(), sb.Size}; device->CreateBuffer(sb, &mdd, &gs.skinMorph);
	}
}

// Rebuild every cached BLAS range of a skinned instance over its freshly posed positions.
void NukeDiligent::Impl::RebuildSkinBLAS(Mesh* instance, MeshGPU& gi)
{
	if (!rtSupported || !gi.idx) return;
	for (auto it = blasSectionCache.lower_bound({instance, 0ull});
	     it != blasSectionCache.end() && it->first.first == instance; ++it)
	{
		IBottomLevelAS* blas = it->second;
		const uint32_t first = (uint32_t)(it->first.second >> 32);
		const uint32_t count = (uint32_t)(it->first.second & 0xFFFFFFFFull);
		if (!blas || count < 3) continue;
		BLASBuildTriangleData td;
		td.GeometryName         = "geo";
		td.pVertexBuffer        = gi.pos;
		td.VertexStride         = 3 * sizeof(float);
		td.VertexCount          = (Uint32)gi.numVerts;
		td.VertexValueType      = VT_FLOAT32;
		td.VertexComponentCount = 3;
		td.pIndexBuffer         = gi.idx;
		td.IndexOffset          = (Uint64)first * sizeof(uint32_t);
		td.IndexType            = VT_UINT32;
		td.PrimitiveCount       = (Uint32)(count / 3);
		td.Flags                = instance->rtAlphaTested ? RAYTRACING_GEOMETRY_FLAG_NONE : RAYTRACING_GEOMETRY_FLAG_OPAQUE;
		RefCntAutoPtr<IBuffer> scratch;
		BufferDesc sbd; sbd.Name = "skin BLAS scratch"; sbd.Usage = USAGE_DEFAULT; sbd.BindFlags = BIND_RAY_TRACING;
		sbd.Size = blas->GetScratchBufferSizes().Build;
		device->CreateBuffer(sbd, nullptr, &scratch);
		if (!scratch) continue;
		BuildBLASAttribs ba;
		ba.pBLAS                  = blas;
		ba.pTriangleData          = &td;
		ba.TriangleDataCount      = 1;
		ba.pScratchBuffer         = scratch;
		ba.Update                 = false;   // full rebuild (pose deltas exceed refit guarantees)
		ba.BLASTransitionMode     = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
		ba.GeometryTransitionMode = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
		ba.ScratchBufferTransitionMode = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
		context->BuildBLAS(ba);
		Trash(scratch);   // the build may still be in flight — parked, not freed inline
	}
}

bool NukeDiligent::gpuSkin() { return m_impl->skinCSPSO && m_impl->skinCSSRB; }

void NukeDiligent::setSkinPalette(Mesh* instance, Mesh* source, const float* palette16, int boneCount,
                                  const float* morphWeights, int morphCount)
{
	Impl& im = *m_impl;
	if (!im.skinCSPSO || !im.skinCSSRB || !instance || !source || !palette16 || boneCount <= 0) return;
	auto recIt = im.skinRecs.find(instance);
	if (recIt == im.skinRecs.end())
	{
		// A pre-skin cache (edit-mode bind-pose draws) lacks the UAV outputs — rebuild it.
		if (im.meshCache.count(instance)) invalidateMesh(instance);
		recIt = im.skinRecs.emplace(instance, Impl::SkinRec{}).first;
	}
	Impl::SkinRec& rec = recIt->second;
	rec.src = source;
	Impl::MeshGPU* gi = im.GetMeshGPU(instance);
	Impl::MeshGPU* gs = im.GetMeshGPU(source);
	if (!gi || !gs || !gi->skinned || !gi->skinPosPrev) return;
	im.EnsureSkinInputs(source, *gs);
	if (!gs->skinSrcPos || !gs->skinIdxBuf || !gs->skinWgtBuf || !gs->skinMorph) return;

	if (boneCount > rec.boneCap || !rec.palette)
	{
		im.Trash(rec.palette);
		BufferDesc pb; pb.Name = "skin palette"; pb.Usage = USAGE_DYNAMIC; pb.CPUAccessFlags = CPU_ACCESS_WRITE;
		pb.Mode = BUFFER_MODE_STRUCTURED; pb.ElementByteStride = sizeof(float) * 4;
		pb.BindFlags = BIND_SHADER_RESOURCE;
		pb.Size = (Uint64)boneCount * 16 * sizeof(float);
		im.device->CreateBuffer(pb, nullptr, &rec.palette);
		rec.boneCap = boneCount;
	}
	if (!rec.palette) return;
	{
		MapHelper<float> mp(im.context, rec.palette, MAP_WRITE, MAP_FLAG_DISCARD);
		if (mp == nullptr) return;
		memcpy(mp, palette16, sizeof(float) * 16 * boneCount);
	}
	const int mw = morphCount > 0 ? morphCount : 1;
	if (mw > rec.morphCap || !rec.morphW)
	{
		im.Trash(rec.morphW);
		BufferDesc mb; mb.Name = "skin morph weights"; mb.Usage = USAGE_DYNAMIC; mb.CPUAccessFlags = CPU_ACCESS_WRITE;
		mb.Mode = BUFFER_MODE_STRUCTURED; mb.ElementByteStride = sizeof(float);
		mb.BindFlags = BIND_SHADER_RESOURCE;
		mb.Size = (Uint64)mw * sizeof(float);
		im.device->CreateBuffer(mb, nullptr, &rec.morphW);
		rec.morphCap = mw;
	}
	if (!rec.morphW) return;
	{
		MapHelper<float> mp(im.context, rec.morphW, MAP_WRITE, MAP_FLAG_DISCARD);
		if (mp == nullptr) return;
		if (morphWeights && morphCount > 0) memcpy(mp, morphWeights, sizeof(float) * morphCount);
		else mp[0] = 0.0f;
	}
	{
		MapHelper<Uint32> pc(im.context, im.skinCSParamsCB, MAP_WRITE, MAP_FLAG_DISCARD);
		if (pc == nullptr) return;
		pc[0] = (Uint32)source->numVerts;
		pc[1] = (Uint32)boneCount;
		pc[2] = (Uint32)(morphCount < gs->skinMorphCount ? morphCount : gs->skinMorphCount);
		pc[3] = 0;
	}
	auto bind = [&](const char* nm, IDeviceObject* o)
	{ if (auto* v = im.skinCSSRB->GetVariableByName(SHADER_TYPE_COMPUTE, nm)) v->Set(o); };
	bind("g_Palette",     rec.palette->GetDefaultView(BUFFER_VIEW_SHADER_RESOURCE));
	bind("g_BindPos",     gs->skinSrcPos->GetDefaultView(BUFFER_VIEW_SHADER_RESOURCE));
	bind("g_BindNrm",     gs->skinSrcNrm->GetDefaultView(BUFFER_VIEW_SHADER_RESOURCE));
	bind("g_BoneIdx",     gs->skinIdxBuf->GetDefaultView(BUFFER_VIEW_SHADER_RESOURCE));
	bind("g_BoneWgt",     gs->skinWgtBuf->GetDefaultView(BUFFER_VIEW_SHADER_RESOURCE));
	bind("g_MorphDelta",  gs->skinMorph->GetDefaultView(BUFFER_VIEW_SHADER_RESOURCE));
	bind("g_MorphWeight", rec.morphW->GetDefaultView(BUFFER_VIEW_SHADER_RESOURCE));
	bind("g_PosOut",      gi->pos->GetDefaultView(BUFFER_VIEW_UNORDERED_ACCESS));
	bind("g_NrmOut",      gi->nrm->GetDefaultView(BUFFER_VIEW_UNORDERED_ACCESS));
	bind("g_PosPrev",     gi->skinPosPrev->GetDefaultView(BUFFER_VIEW_UNORDERED_ACCESS));
	im.context->SetPipelineState(im.skinCSPSO);
	im.context->CommitShaderResources(im.skinCSSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	DispatchComputeAttribs da((Uint32)((source->numVerts + 63) / 64), 1, 1);
	im.context->DispatchCompute(da);
	im.RebuildSkinBLAS(instance, *gi);
}

int NukeDiligent::Impl::SelectLod(Mesh* mesh, const float pos[3], const float scale[3])
{
	if (!mesh || mesh->lods.size() < 2) return 0;
	mesh->EnsureBounds();
	if (!mesh->boundsValid) return 0;
	const float ex = mesh->aabbMax[0] - mesh->aabbMin[0];
	const float ey = mesh->aabbMax[1] - mesh->aabbMin[1];
	const float ez = mesh->aabbMax[2] - mesh->aabbMin[2];
	float smax = std::fabs(scale[0]);
	if (std::fabs(scale[1]) > smax) smax = std::fabs(scale[1]);
	if (std::fabs(scale[2]) > smax) smax = std::fabs(scale[2]);
	const float diam = std::sqrt(ex * ex + ey * ey + ez * ez) * smax;
	const float dx = pos[0] - lodCamPos[0], dy = pos[1] - lodCamPos[1], dz = pos[2] - lodCamPos[2];
	const float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
	const float cov = diam / (dist > 0.01f ? dist : 0.01f);   // ~ screen coverage fraction
	int l = 0;
	for (int k = 1; k < (int)mesh->lods.size(); ++k)
		if (cov < mesh->lods[k].screenSize) l = k;   // thresholds descend: deepest passing level wins
	return l;
}

void NukeDiligent::Impl::LodRange(Mesh* mesh, int lod, uint32_t& first, uint32_t& count)
{
	if (!mesh) { first = count = 0; return; }
	if (mesh->numIndices <= 0 || mesh->sections.empty())
	{
		first = 0;
		count = (uint32_t)(mesh->numIndices > 0 ? mesh->numIndices : mesh->numVerts);
		return;
	}
	MeshLOD L = mesh->Lod(lod);
	first = mesh->Section(L.firstSection).firstIndex;
	count = 0;
	for (int s = 0; s < L.sectionCount; ++s) count += mesh->Section(L.firstSection + s).indexCount;
}

void NukeDiligent::bindRenderTarget(uint64_t id)
{
	if (id == 0) { m_impl->uiRTV = nullptr; m_impl->uiTW = m_impl->uiTH = 0; return; }
	auto it = m_impl->rts.find(id);
	if (it == m_impl->rts.end()) { m_impl->uiRTV = nullptr; m_impl->uiTW = m_impl->uiTH = 0; return; }
	// UI must composite over the LDR post output: the RGBA16F HDR target mismatches the UI PSO's
	// RGBA8 format and D3D12 silently discards every draw.
	m_impl->uiRTV = it->second.postRTV ? it->second.postRTV : it->second.rtv;
	m_impl->uiTW = (Uint32)it->second.w; m_impl->uiTH = (Uint32)it->second.h;
}
void NukeDiligent::invalidateTexture(Texture* t)   // re-uploaded on next GetTexSRV
{
	if (!t) return;
	// The old SRV may still sit in this frame's recorded UI draw data — park, don't free inline.
	auto it = m_impl->texCache.find(t);
	if (it != m_impl->texCache.end()) { m_impl->Trash(it->second); m_impl->texCache.erase(it); }
	auto at = m_impl->animTex.find(t);
	if (at != m_impl->animTex.end())
	{
		for (auto& f : at->second) m_impl->Trash(f);
		m_impl->animTex.erase(at);
	}
	m_impl->streamTex.erase(t);   // the pointer may be about to die; feedback re-registers it
	m_impl->storPendingTex.erase(t);   // an in-flight DirectStorage job lands into the trash
	m_impl->storFailed.erase(t);
}

void NukeDiligent::invalidateMesh(Mesh* m)
{
	if (!m) return;
	auto it = m_impl->meshCache.find(m);
	if (it != m_impl->meshCache.end())
	{
		if (it->second.arena)
		{
			// Pooled mesh: give the ranges back — the arena buffers themselves never die here.
			// The GPU may still read them for kTrashFrames, but a reused range is only ever
			// REWRITTEN via UpdateBuffer, which serializes against prior draws.
			Impl::PoolFree(it->second.arena->vFree, it->second.vOff, (uint32_t)it->second.numVerts);
			Impl::PoolFree(it->second.arena->iFree, it->second.iOff, (uint32_t)it->second.numIndices);
		}
		m_impl->Trash(it->second.pos); m_impl->Trash(it->second.nrm); m_impl->Trash(it->second.uv);
		m_impl->Trash(it->second.col); m_impl->Trash(it->second.idx);
		m_impl->Trash(it->second.bendSrc); m_impl->Trash(it->second.bendData); m_impl->Trash(it->second.bendPivot);
		m_impl->Trash(it->second.posBent); m_impl->Trash(it->second.blasScratch);
		m_impl->Trash(it->second.skinPosPrev);
		m_impl->Trash(it->second.skinSrcPos); m_impl->Trash(it->second.skinSrcNrm);
		m_impl->Trash(it->second.skinIdxBuf); m_impl->Trash(it->second.skinWgtBuf); m_impl->Trash(it->second.skinMorph);
		m_impl->meshCache.erase(it);
	}
	// The BLAS references the old pos buffer's memory: keeping it would make the TLAS trace freed memory.
	auto bit = m_impl->blasCache.find(m);
	if (bit != m_impl->blasCache.end()) { m_impl->Trash(bit->second); m_impl->blasCache.erase(bit); }
	for (auto sit = m_impl->blasSectionCache.lower_bound({m, 0ull});
	     sit != m_impl->blasSectionCache.end() && sit->first.first == m; )
	{ m_impl->Trash(sit->second); sit = m_impl->blasSectionCache.erase(sit); }
	// RT attribute-pool slices go stale with the mesh (a freed Mesh* address can be reused).
	m_impl->meshNrmByteOffset.erase(m); m_impl->meshUVByteOffset.erase(m); m_impl->meshPosByteOffset.erase(m);
	// GPU-skin state (instance record + per-frame buffers).
	auto skIt = m_impl->skinRecs.find(m);
	if (skIt != m_impl->skinRecs.end())
	{
		m_impl->Trash(skIt->second.palette);
		m_impl->Trash(skIt->second.morphW);
		m_impl->skinRecs.erase(skIt);
	}
}

// Neutral UI seam: generic 2D draw, no ImGui types.

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
	// The view may still sit in this frame's UI draw data — park the texture + its cached SRB.
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
	// Resizes land mid-frame: park the old targets and drop the cached UI SRB keyed by the old SRV.
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

// Read back the LDR image of `rtId` (0 = backbuffer) into `rgba`, sized w*h. Handles RGBA8 and
// BGRA8 layouts including sRGB views. Returns false when the target is not readable.
bool NukeDiligent::captureTarget(uint64_t rtId, int& w, int& h, std::vector<uint8_t>& rgba)
{
	if (!m_impl->device || !m_impl->context) return false;
	ITexture* src = nullptr;
	if (rtId == 0)
	{
		if (!m_impl->swapChain) return false;
		ITextureView* bb = m_impl->swapChain->GetCurrentBackBufferRTV();
		if (!bb) return false;
		src = bb->GetTexture();
	}
	else
	{
		auto it = m_impl->rts.find(rtId);
		if (it == m_impl->rts.end() || !it->second.post) return false;
		src = it->second.post;
	}
	if (!src) return false;

	const TextureDesc& sd = src->GetDesc();
	const TEXTURE_FORMAT fmt = sd.Format;
	const bool isRGBA = fmt == TEX_FORMAT_RGBA8_UNORM || fmt == TEX_FORMAT_RGBA8_UNORM_SRGB;
	const bool isBGRA = fmt == TEX_FORMAT_BGRA8_UNORM || fmt == TEX_FORMAT_BGRA8_UNORM_SRGB;
	if (!isRGBA && !isBGRA) return false;   // LDR 8-bit only — HDR targets are not "what's shown"

	TextureDesc st;
	st.Name = "capture staging"; st.Type = RESOURCE_DIM_TEX_2D;
	st.Width = sd.Width; st.Height = sd.Height; st.Format = fmt;
	st.MipLevels = 1; st.Usage = USAGE_STAGING; st.CPUAccessFlags = CPU_ACCESS_READ;
	st.BindFlags = BIND_NONE;
	RefCntAutoPtr<ITexture> staging;
	m_impl->device->CreateTexture(st, nullptr, &staging);
	if (!staging) return false;

	CopyTextureAttribs cp(src, RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
	                      staging, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	m_impl->context->CopyTexture(cp);
	m_impl->context->Flush();
	m_impl->device->IdleGPU();   // the copy must be complete before the map

	MappedTextureSubresource m;
	m_impl->context->MapTextureSubresource(staging, 0, 0, MAP_READ, MAP_FLAG_NONE, nullptr, m);
	if (!m.pData) return false;
	w = (int)sd.Width; h = (int)sd.Height;
	rgba.resize((size_t)w * h * 4);
	const uint8_t* srcRow = (const uint8_t*)m.pData;
	for (int y = 0; y < h; ++y, srcRow += m.Stride)
	{
		uint8_t* dst = rgba.data() + (size_t)y * w * 4;
		if (isRGBA)
			std::memcpy(dst, srcRow, (size_t)w * 4);
		else
			for (int x = 0; x < w; ++x)   // BGRA -> RGBA
			{
				dst[x * 4 + 0] = srcRow[x * 4 + 2];
				dst[x * 4 + 1] = srcRow[x * 4 + 1];
				dst[x * 4 + 2] = srcRow[x * 4 + 0];
				dst[x * 4 + 3] = srcRow[x * 4 + 3];
			}
	}
	m_impl->context->UnmapTextureSubresource(staging, 0, 0);
	return true;
}
