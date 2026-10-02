// Module compute seam (abi 60): opaque handles over Diligent buffers / 3D textures / compute
// pipelines, dispatches bound by variable NAME, async readbacks, and GPU-resident pooled
// meshes a compute shader writes straight into the shared stream arena (terrain mesher).
#include "NukeDiligentImpl.h"
#include <cstring>

using namespace Diligent;
using std::cout; using std::endl;
using nuke::NukeGpuBind; using nuke::NukeGpuMeshRange; using nuke::Mesh; using nuke::Material;

bool NukeDiligent::gpuSupported() { return m_impl->device != nullptr && m_impl->context != nullptr; }

uint64_t NukeDiligent::gpuCreateBuffer(uint64_t bytes, int usage, uint32_t stride, const void* init)
{
	Impl& im = *m_impl;
	if (!im.device || bytes == 0) return 0;
	BufferDesc d; d.Name = "module gpu buffer"; d.Size = bytes; d.Usage = USAGE_DEFAULT;
	d.BindFlags = BIND_SHADER_RESOURCE | BIND_UNORDERED_ACCESS;
	if (usage == 2) d.BindFlags |= BIND_INDIRECT_DRAW_ARGS;
	if (usage == 0 && stride > 0) { d.Mode = BUFFER_MODE_STRUCTURED; d.ElementByteStride = stride; }
	else { d.Mode = BUFFER_MODE_RAW; d.ElementByteStride = 4; }
	Impl::GpuRes r;
	BufferData bd{ init, init ? bytes : 0 };
	im.device->CreateBuffer(d, init ? &bd : nullptr, &r.buf);
	if (!r.buf) return 0;
	const uint64_t id = im.gpuNext++;
	im.gpuRes[id] = r;
	return id;
}

void NukeDiligent::gpuUpdateBuffer(uint64_t buf, uint64_t offset, const void* data, uint64_t bytes)
{
	Impl& im = *m_impl;
	auto it = im.gpuRes.find(buf);
	if (it == im.gpuRes.end() || !it->second.buf || !data || !bytes) return;
	im.context->UpdateBuffer(it->second.buf, offset, bytes, data, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
}

void NukeDiligent::gpuCopyBuffer(uint64_t src, uint64_t srcOff, uint64_t dst, uint64_t dstOff, uint64_t bytes)
{
	Impl& im = *m_impl;
	IBuffer* s = im.GpuBuffer(src); IBuffer* d = im.GpuBuffer(dst);
	if (!s || !d || !bytes) return;
	im.context->CopyBuffer(s, srcOff, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, d, dstOff, bytes, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
}

uint64_t NukeDiligent::gpuCreateTexture3D(int w, int h, int d, int format, const void* init)
{
	Impl& im = *m_impl;
	if (!im.device || w <= 0 || h <= 0 || d <= 0) return 0;
	TextureDesc td; td.Name = "module gpu tex3d"; td.Type = RESOURCE_DIM_TEX_3D;
	td.Width = (Uint32)w; td.Height = (Uint32)h; td.Depth = (Uint32)d; td.MipLevels = 1;
	td.Usage = USAGE_DEFAULT; td.BindFlags = BIND_SHADER_RESOURCE | BIND_UNORDERED_ACCESS;
	switch (format)
	{
	case 0: td.Format = TEX_FORMAT_R8_SNORM; break;
	case 1: td.Format = TEX_FORMAT_R8_UINT; break;
	case 2: td.Format = TEX_FORMAT_R8_SINT; break;
	case 3: td.Format = TEX_FORMAT_R32_UINT; break;
	default: return 0;
	}
	const Uint32 bpp = format == 3 ? 4 : 1;
	TextureSubResData sub; TextureData tdat;
	if (init)
	{
		sub.pData = init; sub.Stride = (Uint64)w * bpp; sub.DepthStride = (Uint64)w * h * bpp;
		tdat.pSubResources = &sub; tdat.NumSubresources = 1;
	}
	Impl::GpuRes r;
	im.device->CreateTexture(td, init ? &tdat : nullptr, &r.tex);
	if (!r.tex) return 0;
	const uint64_t id = im.gpuNext++;
	im.gpuRes[id] = r;
	return id;
}

void NukeDiligent::gpuUpdateTexture3D(uint64_t tex, int x, int y, int z, int w, int h, int d,
                                      const void* data, uint32_t rowBytes, uint32_t sliceBytes)
{
	Impl& im = *m_impl;
	auto it = im.gpuRes.find(tex);
	if (it == im.gpuRes.end() || !it->second.tex || !data) return;
	Box box; box.MinX = (Uint32)x; box.MaxX = (Uint32)(x + w); box.MinY = (Uint32)y; box.MaxY = (Uint32)(y + h);
	box.MinZ = (Uint32)z; box.MaxZ = (Uint32)(z + d);
	TextureSubResData sub; sub.pData = data; sub.Stride = rowBytes; sub.DepthStride = sliceBytes;
	im.context->UpdateTexture(it->second.tex, 0, 0, box, sub, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
}

void NukeDiligent::gpuDestroy(uint64_t res)
{
	Impl& im = *m_impl;
	auto it = im.gpuRes.find(res);
	if (it != im.gpuRes.end())
	{
		im.Trash(it->second.buf); im.Trash(it->second.tex);
		im.gpuRes.erase(it);
		return;
	}
	auto pit = im.gpuPipes.find(res);
	if (pit != im.gpuPipes.end())
	{
		im.Trash(pit->second.srb); im.Trash(pit->second.pso); im.Trash(pit->second.cb);
		im.gpuPipes.erase(pit);
	}
}

uint64_t NukeDiligent::gpuCreateCompute(const char* name, const char* hlsl, const char* entry)
{
	Impl& im = *m_impl;
	if (!im.device || !name || !hlsl) return 0;
	ShaderCreateInfo sci; sci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
	sci.pShaderSourceStreamFactory = im.ShaderFactory();
	sci.Desc = { name, SHADER_TYPE_COMPUTE, true };
	sci.Source = hlsl; sci.EntryPoint = entry && entry[0] ? entry : "main";
	RefCntAutoPtr<IShader> cs;
	im.CreateShaderCached(sci, &cs);
	if (!cs) { cout << "[NukeDiligent]\tgpuCreateCompute '" << name << "': shader failed to compile" << endl; return 0; }
	ComputePipelineStateCreateInfo ci; ci.PSODesc.Name = name;
	ci.PSODesc.ResourceLayout.DefaultVariableType = SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC;
	ci.pCS = cs;
	Impl::GpuPipe p;
	im.CreateComputePipelineStateCached(ci, &p.pso);
	if (!p.pso) { cout << "[NukeDiligent]\tgpuCreateCompute '" << name << "': pipeline failed" << endl; return 0; }
	p.pso->CreateShaderResourceBinding(&p.srb, true);
	if (!p.srb) return 0;
	p.name = name;
	const uint64_t id = im.gpuNext++;
	im.gpuPipes[id] = std::move(p);
	return id;
}

// Bind by variable name: the shader's declaration decides SRV / UAV / constant buffer. UAV
// resources already in the UAV state get a UAV->UAV barrier (dependent dispatches).
bool NukeDiligent::Impl::GpuBind(GpuPipe& p, const NukeGpuBind* binds, int n, const void* params, uint32_t paramBytes)
{
	if (params && paramBytes)
	{
		if (!p.cb || p.cbCap < paramBytes)
		{
			Trash(p.cb); p.cb.Release();
			BufferDesc d; d.Name = "module gpu params"; d.Size = (paramBytes + 255) & ~255u; d.Usage = USAGE_DYNAMIC;
			d.BindFlags = BIND_UNIFORM_BUFFER; d.CPUAccessFlags = CPU_ACCESS_WRITE;
			device->CreateBuffer(d, nullptr, &p.cb);
			if (!p.cb) return false;
			p.cbCap = (uint32_t)d.Size;
		}
		void* dst = nullptr;
		context->MapBuffer(p.cb, MAP_WRITE, MAP_FLAG_DISCARD, dst);
		if (!dst) return false;
		std::memcpy(dst, params, paramBytes);
		context->UnmapBuffer(p.cb, MAP_WRITE);
		if (!p.paramsBound) { if (auto* v = p.srb->GetVariableByName(SHADER_TYPE_COMPUTE, "Params")) { v->Set(p.cb); p.paramsBound = true; } }
	}
	std::vector<StateTransitionDesc> barriers;
	for (int i = 0; i < n; ++i)
	{
		auto vit = p.vars.find(binds[i].name);
		if (vit == p.vars.end())
			vit = p.vars.emplace(binds[i].name, std::make_pair(p.srb->GetVariableByName(SHADER_TYPE_COMPUTE, binds[i].name), (IDeviceObject*)nullptr)).first;
		IShaderResourceVariable* v = vit->second.first;
		if (!v) continue;   // a binding the shader does not declare: harmless
		auto it = gpuRes.find(binds[i].res);
		if (it == gpuRes.end()) { cout << "[NukeDiligent]\t" << p.name << ": unknown gpu resource for '" << binds[i].name << "'" << endl; return false; }
		ShaderResourceDesc rd; v->GetResourceDesc(rd);
		const SHADER_RESOURCE_TYPE t = rd.Type;
		GpuRes& r = it->second;
		IDeviceObject* view = nullptr;
		bool uav = false;
		switch (t)
		{
		case SHADER_RESOURCE_TYPE_BUFFER_SRV: if (r.buf) view = r.buf->GetDefaultView(BUFFER_VIEW_SHADER_RESOURCE); break;
		case SHADER_RESOURCE_TYPE_BUFFER_UAV: if (r.buf) { view = r.buf->GetDefaultView(BUFFER_VIEW_UNORDERED_ACCESS); uav = true; } break;
		case SHADER_RESOURCE_TYPE_TEXTURE_SRV: if (r.tex) view = r.tex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE); break;
		case SHADER_RESOURCE_TYPE_TEXTURE_UAV: if (r.tex) { view = r.tex->GetDefaultView(TEXTURE_VIEW_UNORDERED_ACCESS); uav = true; } break;
		case SHADER_RESOURCE_TYPE_CONSTANT_BUFFER: if (r.buf) view = r.buf; break;
		default: break;
		}
		if (!view) { cout << "[NukeDiligent]\t" << p.name << ": '" << binds[i].name << "' resource kind mismatch" << endl; return false; }
		if (vit->second.second != view) { v->Set(view); vit->second.second = view; }
		if (uav)
		{
			if (r.buf && r.buf->GetState() == RESOURCE_STATE_UNORDERED_ACCESS)
				barriers.emplace_back(r.buf, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_UNORDERED_ACCESS, STATE_TRANSITION_FLAG_UPDATE_STATE);
			if (r.tex && r.tex->GetState() == RESOURCE_STATE_UNORDERED_ACCESS)
				barriers.emplace_back(r.tex, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_UNORDERED_ACCESS, STATE_TRANSITION_FLAG_UPDATE_STATE);
		}
	}
	if (!barriers.empty()) context->TransitionResourceStates((Uint32)barriers.size(), barriers.data());
	context->SetPipelineState(p.pso);
	context->CommitShaderResources(p.srb, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	// A compute pass rebinds the context: the scene's "already committed" caches are stale.
	sceneCommitSrb = nullptr; matCBFor = nullptr; tessBindMat = nullptr;
	return true;
}

void NukeDiligent::gpuDispatch(uint64_t pipe, const NukeGpuBind* binds, int bindCount,
                               const void* params, uint32_t paramBytes, uint32_t gx, uint32_t gy, uint32_t gz)
{
	Impl& im = *m_impl;
	auto it = im.gpuPipes.find(pipe);
	if (it == im.gpuPipes.end() || !gx || !gy || !gz) return;
	if (!im.GpuBind(it->second, binds, bindCount, params, paramBytes)) return;
	im.context->DispatchCompute(DispatchComputeAttribs(gx, gy, gz));
}

void NukeDiligent::gpuDispatchIndirect(uint64_t pipe, const NukeGpuBind* binds, int bindCount,
                                       const void* params, uint32_t paramBytes, uint64_t argsBuf, uint64_t argsOffset)
{
	Impl& im = *m_impl;
	auto it = im.gpuPipes.find(pipe);
	IBuffer* args = im.GpuBuffer(argsBuf);
	if (it == im.gpuPipes.end() || !args) return;
	if (!im.GpuBind(it->second, binds, bindCount, params, paramBytes)) return;
	DispatchComputeIndirectAttribs ia; ia.pAttribsBuffer = args; ia.DispatchArgsByteOffset = argsOffset;
	ia.AttribsBufferStateTransitionMode = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
	im.context->DispatchComputeIndirect(ia);
}

uint64_t NukeDiligent::gpuReadback(uint64_t buf, uint64_t offset, uint64_t bytes)
{
	Impl& im = *m_impl;
	IBuffer* src = im.GpuBuffer(buf);
	if (!src || !bytes) return 0;
	Impl::GpuRead r;
	// Reuse a retired staging buffer of a fitting size (a committed resource per readback is
	// the expensive part on D3D12); otherwise create one.
	for (size_t i = 0; i < im.gpuStagingPool.size(); ++i)
	{
		const Uint64 sz = im.gpuStagingPool[i]->GetDesc().Size;
		if (sz >= bytes && sz <= bytes * 4 + 4096) { r.staging = im.gpuStagingPool[i]; im.gpuStagingPool.erase(im.gpuStagingPool.begin() + i); break; }
	}
	if (!r.staging)
	{
		BufferDesc sd; sd.Name = "module gpu readback"; sd.Size = bytes; sd.Usage = USAGE_STAGING;
		sd.BindFlags = BIND_NONE; sd.CPUAccessFlags = CPU_ACCESS_READ;
		im.device->CreateBuffer(sd, nullptr, &r.staging);
		if (!r.staging) return 0;
	}
	im.context->CopyBuffer(src, offset, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, r.staging, 0, bytes, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	if (!im.gpuFence) { FenceDesc fd; fd.Name = "module gpu readback fence"; im.device->CreateFence(fd, &im.gpuFence); }
	if (im.gpuFence) { r.fence = ++im.gpuFenceValue; im.context->EnqueueSignal(im.gpuFence, r.fence); }
	r.bytes = bytes; r.frame = im.frameId;
	const uint64_t id = im.gpuNext++;
	im.gpuReads[id] = r;
	return id;
}

int NukeDiligent::gpuReadbackPoll(uint64_t ticket, void* dst, uint64_t bytes)
{
	Impl& im = *m_impl;
	auto it = im.gpuReads.find(ticket);
	if (it == im.gpuReads.end()) return -1;
	Impl::GpuRead& r = it->second;
	if (im.frameId == r.frame) return 0;   // the copy is in this frame's command stream
	if (im.gpuFence && im.gpuFence->GetCompletedValue() < r.fence) return 0;   // the GPU has not run the copy yet
	void* p = nullptr;
	im.context->MapBuffer(r.staging, MAP_READ, MAP_FLAG_DO_NOT_WAIT, p);
	if (!p)
	{
		im.context->UnmapBuffer(r.staging, MAP_READ);   // D3D11 books the map even when still drawing
		return 0;
	}
	if (dst && bytes) std::memcpy(dst, p, (size_t)std::min(bytes, r.bytes));
	im.context->UnmapBuffer(r.staging, MAP_READ);
	if (im.gpuStagingPool.size() < 64) im.gpuStagingPool.push_back(r.staging); else im.Trash(r.staging);
	im.gpuReads.erase(it);
	return 1;
}

// A pooled mesh with NO CPU streams: the arena ranges are the compute shader's output.
bool NukeDiligent::gpuMeshReserve(Mesh* mesh, uint32_t verts, uint32_t inds, NukeGpuMeshRange& out)
{
	Impl& im = *m_impl;
	std::memset(&out, 0, sizeof(out));
	if (!mesh || !im.device || !verts || !inds) return false;
	auto it = im.meshCache.find(mesh);
	if (it != im.meshCache.end())
	{
		// Re-reserve: hand the old ranges back (and the BLAS over them), then take new ones.
		Impl::MeshGPU& g = it->second;
		if (g.arena)
		{
			Impl::PoolFree(g.arena->vFree, g.vOff, (uint32_t)g.numVerts);
			Impl::PoolFree(g.arena->iFree, g.iOff, (uint32_t)g.numIndices);
		}
		im.Trash(g.blasScratch);
		im.meshCache.erase(it);
		auto bit = im.blasCache.find(mesh);
		if (bit != im.blasCache.end()) { im.Trash(bit->second); im.blasCache.erase(bit); }
		for (auto sit = im.blasSectionCache.lower_bound({mesh, 0ull});
		     sit != im.blasSectionCache.end() && sit->first.first == mesh; )
		{ im.Trash(sit->second); sit = im.blasSectionCache.erase(sit); }
		im.meshNrmByteOffset.erase(mesh); im.meshUVByteOffset.erase(mesh); im.meshPosByteOffset.erase(mesh);
	}
	Impl::MeshGPU g;
	g.resident = true;
	g.numVerts = (int)verts; g.numIndices = (int)inds; g.version = mesh->version;
	g.arena = im.PoolAllocMesh(verts, inds, g.vOff, g.iOff);
	if (!g.arena) return false;
	// The arena buffers double as compute outputs: expose them under stable handles.
	auto handleFor = [&](IBuffer* b) -> uint64_t
	{
		for (auto& kv : im.gpuRes) if (kv.second.buf.RawPtr() == b) return kv.first;
		Impl::GpuRes r; r.buf = b;
		const uint64_t id = im.gpuNext++;
		im.gpuRes[id] = r;
		return id;
	};
	out.pos = handleFor(g.arena->pos); out.nrm = handleFor(g.arena->nrm);
	out.col = handleFor(g.arena->col); out.idx = handleFor(g.arena->idx);
	out.vOff = g.vOff; out.iOff = g.iOff; out.vCap = verts; out.iCap = inds;
	im.meshCache.emplace(mesh, std::move(g));
	return true;
}

void NukeDiligent::renderObjectIndirect(Mesh* mesh, Material* mat, const float pos[3], const float quat[4], const float scale[3],
                                        uint64_t argsBuf, uint64_t argsOffset)
{
	Impl& im = *m_impl;
	IBuffer* args = im.GpuBuffer(argsBuf);
	if (!mesh || !args) return;
	Impl::TagScope tag(m_impl);
	im.gpuIndirectBuf = args; im.gpuIndirectOff = argsOffset;
	RenderObjectRange(mesh, mat, pos, quat, scale, 0, 1);   // the count comes from the args
	im.gpuIndirectBuf = nullptr;
}

void NukeDiligent::renderGBufferObjectIndirect(Mesh* mesh, Material* mat, const float pos[3], const float quat[4], const float scale[3],
                                               uint64_t argsBuf, uint64_t argsOffset)
{
	Impl& im = *m_impl;
	IBuffer* args = im.GpuBuffer(argsBuf);
	if (!mesh || !args) return;
	im.gpuIndirectBuf = args; im.gpuIndirectOff = argsOffset;
	RenderGBufferRange(mesh, mat, pos, quat, scale, pos, quat, scale, 0, 1);
	im.gpuIndirectBuf = nullptr;
}

