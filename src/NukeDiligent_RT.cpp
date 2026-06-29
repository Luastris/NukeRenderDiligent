#include "NukeDiligentImpl.h"

// Ray tracing foundation (D3D12): BLAS per mesh + a per-frame scene TLAS that the world shader ray-queries.
// Phase 2A — builds the acceleration structures only (no shader use yet). All gated on rtSupported.

// Get-or-build the bottom-level AS for a mesh, from its (ray-tracing-capable) position buffer. Non-indexed
// triangle soup (numVerts/3 triangles), matching how renderObject draws. Cached for the mesh's lifetime.
IBottomLevelAS* NukeDiligent::Impl::GetMeshBLAS(Mesh* mesh)
{
	auto it = blasCache.find(mesh);
	if (it != blasCache.end()) return it->second;
	MeshGPU* gp = GetMeshGPU(mesh);
	if (!gp || !gp->pos || gp->numVerts < 3) { blasCache[mesh] = {}; return nullptr; }

	BLASTriangleDesc tri;
	tri.GeometryName        = "geo";
	tri.MaxVertexCount      = (Uint32)gp->numVerts;
	tri.VertexValueType     = VT_FLOAT32;
	tri.VertexComponentCount= 3;
	tri.MaxPrimitiveCount   = (Uint32)(gp->numVerts / 3);
	tri.IndexType           = VT_UNDEFINED;   // non-indexed

	BottomLevelASDesc desc;
	desc.Name          = "Mesh BLAS";
	desc.pTriangles    = &tri;
	desc.TriangleCount = 1;
	desc.Flags         = RAYTRACING_BUILD_AS_PREFER_FAST_TRACE;
	RefCntAutoPtr<IBottomLevelAS> blas;
	device->CreateBLAS(desc, &blas);
	if (!blas) { blasCache[mesh] = {}; return nullptr; }

	RefCntAutoPtr<IBuffer> scratch;
	BufferDesc sbd; sbd.Name = "BLAS scratch"; sbd.Usage = USAGE_DEFAULT; sbd.BindFlags = BIND_RAY_TRACING;
	sbd.Size = blas->GetScratchBufferSizes().Build;
	device->CreateBuffer(sbd, nullptr, &scratch);

	BLASBuildTriangleData td;
	td.GeometryName         = "geo";
	td.pVertexBuffer        = gp->pos;
	td.VertexStride         = 3 * sizeof(float);
	td.VertexCount          = (Uint32)gp->numVerts;
	td.VertexValueType      = VT_FLOAT32;
	td.VertexComponentCount = 3;
	td.PrimitiveCount       = (Uint32)(gp->numVerts / 3);
	td.Flags                = RAYTRACING_GEOMETRY_FLAG_OPAQUE;

	BuildBLASAttribs ba;
	ba.pBLAS                  = blas;
	ba.pTriangleData          = &td;
	ba.TriangleDataCount      = 1;
	ba.pScratchBuffer         = scratch;
	ba.BLASTransitionMode     = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
	ba.GeometryTransitionMode = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
	ba.ScratchBufferTransitionMode = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
	context->BuildBLAS(ba);

	blasCache[mesh] = blas;
	return blas;
}

// Empty TLAS bound to g_TLAS whenever there is no scene TLAS (e.g. a scene with no opaque meshes) so the shader
// resource is always valid; ray queries against it simply miss (fully lit). Built once.
void NukeDiligent::Impl::EnsureRTFallback()
{
	if (fallbackTLAS || !rtSupported) return;
	TopLevelASDesc td; td.Name = "Fallback TLAS"; td.MaxInstanceCount = 1; td.Flags = RAYTRACING_BUILD_AS_PREFER_FAST_TRACE;
	device->CreateTLAS(td, &fallbackTLAS);
	if (!fallbackTLAS) return;
	BufferDesc sbd; sbd.Name = "FB TLAS scratch"; sbd.Usage = USAGE_DEFAULT; sbd.BindFlags = BIND_RAY_TRACING;
	sbd.Size = fallbackTLAS->GetScratchBufferSizes().Build; device->CreateBuffer(sbd, nullptr, &fbTlasScratch);
	BufferDesc ibd; ibd.Name = "FB TLAS inst"; ibd.Usage = USAGE_DEFAULT; ibd.BindFlags = BIND_RAY_TRACING;
	ibd.Size = Uint64{TLAS_INSTANCE_DATA_SIZE} * 1; device->CreateBuffer(ibd, nullptr, &fbTlasInst);
	TLASBuildInstanceData dummy{};   // Diligent requires pInstances != null even when InstanceCount == 0
	BuildTLASAttribs ba;
	ba.pTLAS = fallbackTLAS; ba.pInstances = &dummy; ba.InstanceCount = 0;   // empty -> all rays miss
	ba.pInstanceBuffer = fbTlasInst; ba.pScratchBuffer = fbTlasScratch;
	ba.BindingMode = HIT_GROUP_BINDING_MODE_USER_DEFINED; ba.HitGroupStride = 0;
	ba.TLASTransitionMode = ba.BLASTransitionMode = ba.InstanceBufferTransitionMode = ba.ScratchBufferTransitionMode
		= RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
	context->BuildTLAS(ba);
}

bool NukeDiligent::rtAvailable() { return m_impl->rtSupported; }

void NukeDiligent::beginRTScene()
{
	if (!m_impl->rtSupported) return;
	m_impl->EnsureRTFallback();
	m_impl->rtInstances.clear();
	m_impl->rtInstanceNames.clear();
	m_impl->rtSceneReady = false;
}

void NukeDiligent::addRTInstance(Mesh* mesh, const float pos[3], const float quat[4], const float scale[3])
{
	if (!m_impl->rtSupported) return;
	IBottomLevelAS* blas = m_impl->GetMeshBLAS(mesh);
	if (!blas) return;

	float4x4 world = float4x4::Scale(scale[0], scale[1], scale[2])
	               * Diligent::Quaternion<float>(quat[0], quat[1], quat[2], quat[3]).ToMatrix()
	               * float4x4::Translation(pos[0], pos[1], pos[2]);

	TLASBuildInstanceData inst;
	inst.pBLAS    = blas;
	inst.Mask     = 0xFF;
	inst.Flags    = RAYTRACING_INSTANCE_NONE;
	inst.CustomId = 0;
	inst.ContributionToHitGroupIndex = 0;   // USER_DEFINED binding: must be explicit (not TLAS_INSTANCE_OFFSET_AUTO)
	// InstanceMatrix is 3x4 row-major (rotation | translation). Our world matrix is row-vector (v*M), so the
	// instance row r / col c = world.m[c][r] (transpose of the upper 3x3, translation from row 3).
	for (int r = 0; r < 3; ++r)
		for (int c = 0; c < 4; ++c)
			inst.Transform.data[r][c] = world.m[c][r];

	m_impl->rtInstanceNames.push_back("i" + std::to_string(m_impl->rtInstances.size()));
	m_impl->rtInstances.push_back(inst);
}

void NukeDiligent::buildRTScene()
{
	auto* d = m_impl;
	if (!d->rtSupported || d->rtInstances.empty()) { d->rtSceneReady = false; return; }
	const Uint32 count = (Uint32)d->rtInstances.size();
	// Fix up InstanceName pointers (the names vector may have reallocated during accumulation).
	for (Uint32 i = 0; i < count; ++i) d->rtInstances[i].InstanceName = d->rtInstanceNames[i].c_str();

	if (!d->tlas || d->tlasMaxInstances < count)   // (re)create when capacity grows
	{
		d->tlas.Release(); d->tlasScratch.Release(); d->tlasInstanceBuf.Release();
		TopLevelASDesc td; td.Name = "Scene TLAS"; td.MaxInstanceCount = count;
		td.Flags = RAYTRACING_BUILD_AS_PREFER_FAST_TRACE;
		d->device->CreateTLAS(td, &d->tlas);
		if (!d->tlas) { d->rtSceneReady = false; return; }
		d->tlasMaxInstances = count;
		BufferDesc sbd; sbd.Name = "TLAS scratch"; sbd.Usage = USAGE_DEFAULT; sbd.BindFlags = BIND_RAY_TRACING;
		sbd.Size = d->tlas->GetScratchBufferSizes().Build;
		d->device->CreateBuffer(sbd, nullptr, &d->tlasScratch);
		BufferDesc ibd; ibd.Name = "TLAS instances"; ibd.Usage = USAGE_DEFAULT; ibd.BindFlags = BIND_RAY_TRACING;
		ibd.Size = Uint64{TLAS_INSTANCE_DATA_SIZE} * count;
		d->device->CreateBuffer(ibd, nullptr, &d->tlasInstanceBuf);
	}
	if (!d->tlasScratch || !d->tlasInstanceBuf) { d->rtSceneReady = false; return; }

	BuildTLASAttribs ba;
	ba.pTLAS                        = d->tlas;
	ba.pInstances                   = d->rtInstances.data();
	ba.InstanceCount                = count;
	ba.pInstanceBuffer              = d->tlasInstanceBuf;
	ba.pScratchBuffer               = d->tlasScratch;
	ba.BindingMode                  = HIT_GROUP_BINDING_MODE_USER_DEFINED;   // inline ray query: no SBT (stride 0 ok)
	ba.HitGroupStride               = 0;
	ba.TLASTransitionMode           = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
	ba.BLASTransitionMode           = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
	ba.InstanceBufferTransitionMode = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
	ba.ScratchBufferTransitionMode  = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
	d->context->BuildTLAS(ba);
	d->rtSceneReady = true;
}
