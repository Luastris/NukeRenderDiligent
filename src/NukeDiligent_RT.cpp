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

// RT reflections pass: ray-query the TLAS per pixel, shade the hit, blend onto the chain colour.
void NukeDiligent::Impl::RunRTReflect(PostPipe& pp, ITextureView* srcSRV, ITextureView* dstRTV, int w, int h, const std::vector<float>& params)
{
	// clip->view (invProj) + view->world (invView), two-step = numerically stable vs inverting view*proj.
	{ struct CB { float4x4 invProj, invView; }; MapHelper<CB> cb(context, rtRefCB, MAP_WRITE, MAP_FLAG_DISCARD);
	  cb->invProj = curProj.Inverse(); cb->invView = curView.Inverse(); }
	{
		MapHelper<float> p(context, postParamsCB, MAP_WRITE, MAP_FLAG_DISCARD);
		int n = (int)params.size(); if (n > 64) n = 64; for (int k = 0; k < 64; ++k) p[k] = (k < n) ? params[k] : 0.0f;
	}
	context->SetRenderTargets(1, &dstRTV, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	Viewport vp; vp.TopLeftX = 0; vp.TopLeftY = 0; vp.Width = (float)w; vp.Height = (float)h; vp.MinDepth = 0; vp.MaxDepth = 1;
	context->SetViewports(1, &vp, w, h);
	if (pp.srcVar)     pp.srcVar->Set(srcSRV);
	if (pp.gbufVar)    pp.gbufVar->Set(gbufSRV);
	if (pp.depthVar)   pp.depthVar->Set(gbufDepthSRV);
	if (pp.rtProbeVar) pp.rtProbeVar->Set((probeActive && probeCubeSRV) ? probeCubeSRV : fallbackCubeSRV);
	if (pp.tlasVar)    pp.tlasVar->Set((rtSceneReady && tlas) ? (IDeviceObject*)tlas.RawPtr() : (IDeviceObject*)fallbackTLAS.RawPtr());
	if (pp.instVar)    pp.instVar->Set(rtInstSRV);
	if (pp.nrmVar)     pp.nrmVar->Set(rtNrmSRV);
	if (pp.uvVar)      pp.uvVar->Set(rtUVSRV ? rtUVSRV : rtNrmSRV);   // (rtNrmSRV is a valid non-null fallback)
	if (pp.matTexVar)  // fixed bindless array: real albedo SRVs + white fallback in the unused slots
	{
		ITextureView* white = whiteTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
		IDeviceObject* arr[Impl::kMaxMatTex];
		for (uint32_t k = 0; k < kMaxMatTex; ++k)
		{
			if (k < matTexSRVs.size())
			{
				ITextureView* s = GetTexSRV(matTexPtr[k]);   // re-resolve each frame -> animated (GIF) textures update
				if (s) matTexSRVs[k] = s;
				arr[k] = matTexSRVs[k] ? matTexSRVs[k] : white;
			}
			else arr[k] = white;
		}
		pp.matTexVar->SetArray(arr, 0, kMaxMatTex);
	}
	context->SetPipelineState(pp.pso);
	context->CommitShaderResources(pp.srb, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	DrawAttribs da{3, DRAW_FLAG_VERIFY_STATES};
	context->Draw(da);
}

bool NukeDiligent::rtAvailable() { return m_impl->rtSupported; }

void NukeDiligent::beginRTScene()
{
	if (!m_impl->rtSupported) return;
	m_impl->EnsureRTFallback();
	m_impl->rtInstances.clear();
	m_impl->rtInstanceNames.clear();
	m_impl->rtInstData.clear();
	m_impl->rtSceneReady = false;
}

void NukeDiligent::addRTInstance(Mesh* mesh, Material* mat, const float pos[3], const float quat[4], const float scale[3])
{
	if (!m_impl->rtSupported) return;
	IBottomLevelAS* blas = m_impl->GetMeshBLAS(mesh);
	if (!blas || !mesh->normalArray || mesh->numVerts < 3) return;

	// Register this mesh's normals + uvs in the concatenated buffers (once per unique mesh).
	uint32_t nrmOff, uvOff;
	auto nit = m_impl->meshNrmByteOffset.find(mesh);
	if (nit == m_impl->meshNrmByteOffset.end())
	{
		nrmOff = (uint32_t)(m_impl->allNrmCPU.size() * sizeof(float));
		m_impl->allNrmCPU.insert(m_impl->allNrmCPU.end(), mesh->normalArray, mesh->normalArray + (size_t)mesh->numVerts * 3);
		m_impl->meshNrmByteOffset[mesh] = nrmOff;
		uvOff = (uint32_t)(m_impl->allUVCPU.size() * sizeof(float));
		if (mesh->uvArray) m_impl->allUVCPU.insert(m_impl->allUVCPU.end(), mesh->uvArray, mesh->uvArray + (size_t)mesh->numVerts * 2);
		else               m_impl->allUVCPU.insert(m_impl->allUVCPU.end(), (size_t)mesh->numVerts * 2, 0.0f);
		m_impl->meshUVByteOffset[mesh] = uvOff;
		m_impl->allNrmDirty = true;
	}
	else { nrmOff = nit->second; uvOff = m_impl->meshUVByteOffset[mesh]; }

	// Material albedo texture -> bindless slot (flat color when none / table full).
	uint32_t texIdx = 0xFFFFFFFFu;
	if (mat && mat->diff)
	{
		auto tit = m_impl->matTexSlot.find(mat->diff);
		if (tit != m_impl->matTexSlot.end()) texIdx = tit->second;
		else if (m_impl->matTexSRVs.size() < Impl::kMaxMatTex)
		{
			ITextureView* srv = m_impl->GetTexSRV(mat->diff);
			if (srv) { texIdx = (uint32_t)m_impl->matTexSRVs.size(); m_impl->matTexSRVs.push_back(srv); m_impl->matTexPtr.push_back(mat->diff); m_impl->matTexSlot[mat->diff] = texIdx; }
		}
	}

	float4x4 world = float4x4::Scale(scale[0], scale[1], scale[2])
	               * Diligent::Quaternion<float>(quat[0], quat[1], quat[2], quat[3]).ToMatrix()
	               * float4x4::Translation(pos[0], pos[1], pos[2]);

	TLASBuildInstanceData inst;
	inst.pBLAS    = blas;
	inst.Mask     = 0xFF;
	inst.Flags    = RAYTRACING_INSTANCE_NONE;
	inst.CustomId = (Uint32)m_impl->rtInstances.size();   // -> g_Instances index (CommittedInstanceID in the shader)
	inst.ContributionToHitGroupIndex = 0;   // USER_DEFINED binding: must be explicit (not TLAS_INSTANCE_OFFSET_AUTO)

	// Per-instance material for hit shading.
	Impl::RTInstanceData d{}; d.nrmOffset = nrmOff; d.uvOffset = uvOff; d.texIndex = texIdx;
	d.pad = (mat && mat->shaderGuid == "unlit") ? 1u : 0u;   // unlit -> flat base colour in reflections
	float alb[3] = {1, 1, 1}, em[3] = {0, 0, 0}; float metal = 0.0f, rough = 0.6f, emI = 0.0f;
	if (mat)
	{
		alb[0] = (float)mat->color.r; alb[1] = (float)mat->color.g; alb[2] = (float)mat->color.b;
		metal = mat->metallic; rough = mat->roughness;
		em[0] = (float)mat->emissive.r; em[1] = (float)mat->emissive.g; em[2] = (float)mat->emissive.b; emI = mat->emissiveIntensity;
	}
	d.albedoMetal[0] = alb[0]; d.albedoMetal[1] = alb[1]; d.albedoMetal[2] = alb[2]; d.albedoMetal[3] = metal;
	d.emissiveRough[0] = em[0] * emI; d.emissiveRough[1] = em[1] * emI; d.emissiveRough[2] = em[2] * emI; d.emissiveRough[3] = rough;
	m_impl->rtInstData.push_back(d);
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

	// Concatenated normals (immutable; rebuilt only when a new mesh appeared).
	if (d->allNrmDirty && !d->allNrmCPU.empty())
	{
		d->rtNrmBuf.Release(); d->rtNrmSRV = nullptr;
		BufferDesc bd; bd.Name = "RT Normals"; bd.Usage = USAGE_IMMUTABLE; bd.BindFlags = BIND_SHADER_RESOURCE;
		bd.Mode = BUFFER_MODE_RAW; bd.Size = (Uint64)d->allNrmCPU.size() * sizeof(float);
		BufferData bdat{d->allNrmCPU.data(), bd.Size};
		d->device->CreateBuffer(bd, &bdat, &d->rtNrmBuf);
		if (d->rtNrmBuf) d->rtNrmSRV = d->rtNrmBuf->GetDefaultView(BUFFER_VIEW_SHADER_RESOURCE);

		d->rtUVBuf.Release(); d->rtUVSRV = nullptr;
		if (!d->allUVCPU.empty())
		{
			BufferDesc ud; ud.Name = "RT UVs"; ud.Usage = USAGE_IMMUTABLE; ud.BindFlags = BIND_SHADER_RESOURCE;
			ud.Mode = BUFFER_MODE_RAW; ud.Size = (Uint64)d->allUVCPU.size() * sizeof(float);
			BufferData udat{d->allUVCPU.data(), ud.Size};
			d->device->CreateBuffer(ud, &udat, &d->rtUVBuf);
			if (d->rtUVBuf) d->rtUVSRV = d->rtUVBuf->GetDefaultView(BUFFER_VIEW_SHADER_RESOURCE);
		}
		d->allNrmDirty = false;
	}
	// Per-instance data (rebuilt each frame; grows capacity as needed).
	if (!d->rtInstBuf || d->rtInstCapacity < count)
	{
		d->rtInstBuf.Release(); d->rtInstSRV = nullptr; d->rtInstCapacity = count;
		BufferDesc bd; bd.Name = "RT Instances"; bd.Usage = USAGE_DEFAULT; bd.BindFlags = BIND_SHADER_RESOURCE;
		bd.Mode = BUFFER_MODE_STRUCTURED; bd.ElementByteStride = sizeof(Impl::RTInstanceData);
		bd.Size = (Uint64)sizeof(Impl::RTInstanceData) * count;
		d->device->CreateBuffer(bd, nullptr, &d->rtInstBuf);
		if (d->rtInstBuf) d->rtInstSRV = d->rtInstBuf->GetDefaultView(BUFFER_VIEW_SHADER_RESOURCE);
	}
	if (d->rtInstBuf && !d->rtInstData.empty())
		d->context->UpdateBuffer(d->rtInstBuf, 0, (Uint64)sizeof(Impl::RTInstanceData) * d->rtInstData.size(),
		                         d->rtInstData.data(), RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

	d->rtSceneReady = true;
}
