#include "NukeDiligentImpl.h"


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
			meshCache.erase(it);
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
void NukeDiligent::invalidateTexture(Texture* t) { if (t) m_impl->texCache.erase(t); }   // re-uploaded on next GetTexSRV

void NukeDiligent::invalidateMesh(Mesh* m)
{
	if (!m) return;
	m_impl->meshCache.erase(m);   // buffers are ref-counted; Diligent releases them GPU-safely
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
	m_impl->uiSRBCache.erase(reinterpret_cast<ITextureView*>(handle));   // drop its cached SRB too
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
