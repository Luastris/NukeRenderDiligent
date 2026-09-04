#include "NukeDiligentImpl.h"
#include <sstream>
#include <cctype>

// HLSL scalar/vector type name -> component count (0 = unsupported).
static int RTCompsOf(const std::string& t)
{
	if (t == "float" || t == "int" || t == "uint" || t == "bool") return 1;
	if (t == "float2" || t == "int2" || t == "uint2") return 2;
	if (t == "float3" || t == "int3" || t == "uint3") return 3;
	if (t == "float4" || t == "int4" || t == "uint4") return 4;
	return 0;
}

// Generate a closest-hit shader for a material shader from its MatCB schema + its "<name>.surf.hlsl".
// Returns the HLSL source.
std::string NukeDiligent::Impl::GenChitSource(const std::string& name, const std::string& ps)
{
	std::string decls, loads;
	size_t cb = ps.find("cbuffer MatCB");
	if (cb != std::string::npos)
	{
		size_t open = ps.find('{', cb), close = (open == std::string::npos) ? std::string::npos : ps.find('}', open);
		if (open != std::string::npos && close != std::string::npos)
		{
			std::string raw = ps.substr(open + 1, close - open - 1), b;
			for (size_t i = 0; i < raw.size(); )   // strip // comments + preprocessor lines (no ';' — they'd swallow the next decl)
			{
				if ((raw[i] == '/' && i + 1 < raw.size() && raw[i + 1] == '/') || raw[i] == '#')
					{ while (i < raw.size() && raw[i] != '\n') ++i; }
				else b += raw[i++];
			}
			// A cbuffer that #includes the std block (matcb_std): the RT mat block carries the
			// hardcoded std head (Color/Params/Params2/Emissive2 = 64 bytes) and then the shader's
			// OWN decls packed compactly — both this codegen and the CPU pack (AddRTInstanceRange)
			// start those at 64.
			uint32_t off = (raw.find("#include") != std::string::npos) ? 64u : 0u;
			for (size_t p = 0; p < b.size(); )
			{
				size_t sc = b.find(';', p); if (sc == std::string::npos) break;
				std::string stmt = b.substr(p, sc - p); p = sc + 1;
				size_t eq = stmt.find('='); if (eq != std::string::npos) stmt = stmt.substr(0, eq);
				std::istringstream is(stmt); std::string type, nm; if (!(is >> type >> nm)) continue;
				int n = RTCompsOf(type); if (n == 0) continue;
				std::string ident; for (char c : nm) { if (std::isalnum((unsigned char)c) || c == '_') ident += c; else break; }
				if (ident.empty()) continue;
				if ((off % 16) + (uint32_t)n * 4 > 16) off = (off + 15u) & ~15u;   // cbuffer 16-byte register packing
				const char* suf = (n == 1) ? "" : (n == 2) ? "2" : (n == 3) ? "3" : "4";
				std::string ld = "g_MatBytes.Load" + std::string(suf) + "(o+" + std::to_string(off) + ")";
				if (type.rfind("float", 0) == 0)    ld = "asfloat(" + ld + ")";
				else if (type.rfind("int", 0) == 0) ld = "asint(" + ld + ")";
				else if (type == "bool")            ld = "(" + ld + "!=0)";
				decls += "static " + type + " " + ident + ";\n";
				loads += "  " + ident + " = " + ld + ";\n";
				off += (uint32_t)n * 4;
			}
		}
	}
	std::ostringstream s;
	s << "#include \"rt_common.hlsl\"\n" << decls << "static uint __texIndex;\n"
	  << "void __LoadMat(uint o){\n" << loads << "}\n"
	  << "#define MAT_BASE_TEX(uv) ((__texIndex!=0xFFFFFFFFu)? g_MatTex[NonUniformResourceIndex(__texIndex)].SampleLevel(g_MatTex_sampler,(uv),0) : float4(1,1,1,1))\n";
	// Code-registered surface body (module-embedded shaders) beats the file include.
	auto reg = rtSurfSources.find(name);
	if (reg != rtSurfSources.end()) s << reg->second << "\n";
	else s << "#include \"" << name << ".surf.hlsl\"\n";
	s
	  << "[shader(\"closesthit\")] void main(inout RTPayload p, in BuiltInTriangleIntersectionAttributes attr){\n"
	  << "  RTInstanceData inst = g_Instances[InstanceID()];\n"
	  << "  __texIndex = inst.texIndex; __LoadMat(inst.matByteOffset);\n"
	  << "  float3 wdir = WorldRayDirection();\n"
	  << "  SurfaceIn IN;\n"
	  << "  IN.uv = FetchUV(inst.uvOffset, PrimitiveIndex(), attr.barycentrics);\n"
	  << "  float3 geomN = FetchWorldNormal(inst.nrmOffset, PrimitiveIndex(), attr.barycentrics, ObjectToWorld3x4());\n"
	  << "  IN.worldNormal = ApplyNormalMap(inst, PrimitiveIndex(), IN.uv, geomN, ObjectToWorld3x4());\n"
	  << "  if (dot(IN.worldNormal,wdir)>0.0) IN.worldNormal=-IN.worldNormal;\n"
	  << "  IN.worldPos = WorldRayOrigin()+wdir*RayTCurrent(); IN.viewDir=-wdir;\n"
	  << "  SurfaceOut O=(SurfaceOut)0; O.albedo=float3(1,1,1); O.roughness=1.0; O.alpha=1.0; O.unlit=false;\n"
	  << "  Surface(IN,O);\n"
	  << "  if (O.unlit){ float wT0=RTWaterTrans(WorldRayOrigin(),IN.worldPos); p.color=O.emissive*wT0+RTWaterLook(wdir)*(1.0-wT0); return; }\n"
	  << "  float aoM=SampleAO(inst,IN.uv); float3 specM=SampleSpec(inst,IN.uv);\n"
	  << "  float3 col = ShadeSurface(IN.worldPos,IN.worldNormal,IN.viewDir,O.albedo,O.metallic,O.roughness,O.emissive,aoM,specM);\n"
	  << "  float3 R=reflect(wdir,IN.worldNormal); float3 env=ReflEnv(R,O.roughness), traced=env;\n"
	  << "  if (p.depth<(uint)g_RTParams.z){ RayDesc ray; ray.Origin=IN.worldPos+IN.worldNormal*0.08+R*0.05; ray.Direction=R; ray.TMin=0.02; ray.TMax=(g_RTParams.y>0.5)?g_RTParams.y:1000.0; RTPayload p2; p2.color=0.0; p2.depth=p.depth+1; TraceRay(g_TLAS,RAY_FLAG_NONE,RT_REFLECT_MASK,0,1,0,ray,p2); traced=p2.color; }\n"
	  << "  col += SpecFr(IN.worldNormal,IN.viewDir,O.roughness,O.albedo,O.metallic,specM)*lerp(traced,env,O.roughness);\n"
	  << "  float wT=RTWaterTrans(WorldRayOrigin(),IN.worldPos); p.color=col*wT+RTWaterLook(wdir)*(1.0-wT);\n}\n";
	return s.str();
}

// Shared grow-only scratch for one-shot BLAS builds (bend/dynamic meshes keep their own —
// they refit per frame and the scratch must outlive the mesh's builds).
IBuffer* NukeDiligent::Impl::BlasScratchFor(Uint64 size)
{
	if (size > blasSharedScratchSize)
	{
		Trash(blasSharedScratch);
		blasSharedScratch.Release();
		BufferDesc sbd; sbd.Name = "BLAS shared scratch"; sbd.Usage = USAGE_DEFAULT; sbd.BindFlags = BIND_RAY_TRACING;
		sbd.Size = blasSharedScratchSize = (size * 5) / 4;   // slack: nearby nodes vary in size
		device->CreateBuffer(sbd, nullptr, &blasSharedScratch);
	}
	return blasSharedScratch;
}

// Get-or-build the bottom-level AS for a mesh from its position buffer (non-indexed triangle
// soup, numVerts/3 triangles). Cached for the mesh's lifetime; null if not buildable.
IBottomLevelAS* NukeDiligent::Impl::GetMeshBLAS(Mesh* mesh)
{
	auto it = blasCache.find(mesh);
	if (it != blasCache.end()) return it->second;
	MeshGPU* gp = GetMeshGPU(mesh);
	if (!gp || !gp->PosBuf() || gp->numVerts < 3) { blasCache[mesh] = {}; return nullptr; }

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
	IBuffer* scratchPtr = nullptr;
	if (gp->posBent || mesh->rtDynamic)
	{
		// Per-frame BLAS rebuilds keep a dedicated scratch alive with the mesh.
		BufferDesc sbd; sbd.Name = "BLAS scratch"; sbd.Usage = USAGE_DEFAULT; sbd.BindFlags = BIND_RAY_TRACING;
		sbd.Size = blas->GetScratchBufferSizes().Build;
		device->CreateBuffer(sbd, nullptr, &scratch);
		gp->blasScratch = scratch;
		scratchPtr = scratch;
	}
	else
		scratchPtr = BlasScratchFor(blas->GetScratchBufferSizes().Build);

	BLASBuildTriangleData td;
	td.GeometryName         = "geo";
	td.pVertexBuffer        = gp->posBent ? gp->posBent.RawPtr() : gp->PosBuf();   // bend meshes trace the BENT positions
	td.VertexOffset         = gp->posBent ? 0 : gp->PosOfs();
	td.VertexStride         = 3 * sizeof(float);
	td.VertexCount          = (Uint32)gp->numVerts;
	td.VertexValueType      = VT_FLOAT32;
	td.VertexComponentCount = 3;
	td.PrimitiveCount       = (Uint32)(gp->numVerts / 3);
	// Alpha-tested geometry must be NON-OPAQUE or traversal commits without running the alpha test.
	td.Flags                = mesh->rtAlphaTested ? RAYTRACING_GEOMETRY_FLAG_NONE : RAYTRACING_GEOMETRY_FLAG_OPAQUE;

	BuildBLASAttribs ba;
	ba.pBLAS                  = blas;
	ba.pTriangleData          = &td;
	ba.TriangleDataCount      = 1;
	ba.pScratchBuffer         = scratchPtr;
	ba.BLASTransitionMode     = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
	ba.GeometryTransitionMode = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
	ba.ScratchBufferTransitionMode = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
	context->BuildBLAS(ba);

	blasCache[mesh] = blas;
	return blas;
}

// Get-or-build a BLAS over an INDEX-BUFFER RANGE of a v4 indexed mesh (a material section, or
// the whole LOD0). Positions stay the shared vertex buffer; the range picks the triangles.
IBottomLevelAS* NukeDiligent::Impl::GetMeshBLASRange(Mesh* mesh, uint32_t firstIndex, uint32_t indexCount)
{
	const std::pair<Mesh*, uint64_t> key(mesh, ((uint64_t)firstIndex << 32) | indexCount);
	auto it = blasSectionCache.find(key);
	if (it != blasSectionCache.end()) return it->second;
	MeshGPU* gp = GetMeshGPU(mesh);
	if (!gp || !gp->PosBuf() || !gp->IdxBuf() || indexCount < 3) { blasSectionCache[key] = {}; return nullptr; }

	BLASTriangleDesc tri;
	tri.GeometryName         = "geo";
	tri.MaxVertexCount       = (Uint32)gp->numVerts;
	tri.VertexValueType      = VT_FLOAT32;
	tri.VertexComponentCount = 3;
	tri.MaxPrimitiveCount    = (Uint32)(indexCount / 3);
	tri.IndexType            = VT_UINT32;

	BottomLevelASDesc desc;
	desc.Name          = "Mesh BLAS range";
	desc.pTriangles    = &tri;
	desc.TriangleCount = 1;
	desc.Flags         = RAYTRACING_BUILD_AS_PREFER_FAST_TRACE;
	RefCntAutoPtr<IBottomLevelAS> blas;
	device->CreateBLAS(desc, &blas);
	if (!blas) { blasSectionCache[key] = {}; return nullptr; }

	IBuffer* scratchPtr = BlasScratchFor(blas->GetScratchBufferSizes().Build);

	BLASBuildTriangleData td;
	td.GeometryName         = "geo";
	td.pVertexBuffer        = gp->PosBuf();
	td.VertexOffset         = gp->PosOfs();
	td.VertexStride         = 3 * sizeof(float);
	td.VertexCount          = (Uint32)gp->numVerts;
	td.VertexValueType      = VT_FLOAT32;
	td.VertexComponentCount = 3;
	td.pIndexBuffer         = gp->IdxBuf();
	td.IndexOffset          = gp->IdxOfs() + (Uint64)firstIndex * sizeof(uint32_t);   // 4-aligned by construction
	td.IndexType            = VT_UINT32;
	td.PrimitiveCount       = (Uint32)(indexCount / 3);
	td.Flags                = mesh->rtAlphaTested ? RAYTRACING_GEOMETRY_FLAG_NONE : RAYTRACING_GEOMETRY_FLAG_OPAQUE;

	BuildBLASAttribs ba;
	ba.pBLAS                  = blas;
	ba.pTriangleData          = &td;
	ba.TriangleDataCount      = 1;
	ba.pScratchBuffer         = scratchPtr;
	ba.BLASTransitionMode     = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
	ba.GeometryTransitionMode = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
	ba.ScratchBufferTransitionMode = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
	context->BuildBLAS(ba);

	blasSectionCache[key] = blas;
	return blas;
}

// Build the empty TLAS bound to g_TLAS when there is no scene TLAS, so the shader resource
// is always valid; all ray queries against it miss. Built once.
void NukeDiligent::Impl::EnsureRTFallback()
{
	if (fallbackTLAS || !rtSupported) return;

	// Zero-instance TLAS: requires the NUKE patch in DeviceContextVkImpl::BuildTLAS (null upload block).
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

void NukeDiligent::setRTReflection(float intensity, float maxDist, int bounces, float roughCutoff)
{
	m_impl->rtCfgIntensity = intensity; m_impl->rtCfgMaxDist = maxDist;
	m_impl->rtCfgBounces = bounces; m_impl->rtCfgRoughCut = roughCutoff;
}

void NukeDiligent::beginRTScene()
{
	if (!m_impl->rtSupported) return;
	m_impl->EnsureRTFallback();
	m_impl->rtInstances.clear();
	m_impl->rtBendMeshes.clear();
	m_impl->rtDynMeshes.clear();
	m_impl->rtDynColCPU.clear();
	// rtInstanceNames is a grow-only pool ("i0","i1",...) — instance i always maps to "i<i>",
	// so re-accumulating never needs fresh strings.
	m_impl->rtInstData.clear();
	m_impl->rtInstShaderGuid.clear();
	m_impl->allMatCPU.clear();
	m_impl->rtMatBlockCache.clear();
	m_impl->rtSceneReady = false;
}

void NukeDiligent::addRTInstance(Mesh* mesh, Material* mat, const float pos[3], const float quat[4], const float scale[3], bool inReflections, bool castShadows)
{
	if (!m_impl->rtSupported || !mesh) return;
	// Rays trace LOD0 (full quality); soup meshes resolve to (0, numVerts).
	uint32_t first = 0, count = 0;
	m_impl->LodRange(mesh, 0, first, count);
	AddRTInstanceRange(mesh, mat, pos, quat, scale, inReflections, castShadows, first, count);
}

void NukeDiligent::addRTInstanceMulti(Mesh* mesh, Material* const* mats, int matCount,
                                      const float pos[3], const float quat[4], const float scale[3],
                                      bool inReflections, bool castShadows)
{
	if (!m_impl->rtSupported || !mesh) return;
	if (mesh->numIndices <= 0 || mesh->sections.empty())
	{
		addRTInstance(mesh, matCount > 0 ? mats[0] : nullptr, pos, quat, scale, inReflections, castShadows);
		return;
	}
	MeshLOD L = mesh->Lod(0);
	for (int s = 0; s < L.sectionCount; ++s)
	{
		MeshSection sec = mesh->Section(L.firstSection + s);
		Material* m = (sec.slot >= 0 && sec.slot < matCount && mats[sec.slot]) ? mats[sec.slot]
		            : (matCount > 0 ? mats[0] : nullptr);
		if (m && m->blendMode != 0) continue;   // transparent sections stay out of the TLAS (matches the raster gather)
		const bool secCs = castShadows && (!m || m->castShadows);   // per-SECTION shadow gate
		if (!inReflections && !secCs) continue;
		AddRTInstanceRange(mesh, m, pos, quat, scale, inReflections, secCs, sec.firstIndex, sec.indexCount);
	}
}

void NukeDiligent::AddRTInstanceRange(Mesh* mesh, Material* mat,
                                      const float pos[3], const float quat[4], const float scale[3],
                                      bool inReflections, bool castShadows,
                                      uint32_t firstIndex, uint32_t indexCount)
{
	const bool indexed = mesh->numIndices > 0 && mesh->indexArray != nullptr;
	IBottomLevelAS* blas = indexed ? m_impl->GetMeshBLASRange(mesh, firstIndex, indexCount)
	                               : m_impl->GetMeshBLAS(mesh);
	if (Impl::MeshGPU* gpb = m_impl->GetMeshGPU(mesh))
	{
		if (gpb->posBent) m_impl->rtBendMeshes.push_back(mesh);   // NukeBend + BLAS refit this frame
		if (mesh->rtDynamic) m_impl->rtDynMeshes.push_back(mesh); // engine rewrote the verts -> BLAS rebuild
	}
	if (!blas || !mesh->normalArray || mesh->numVerts < 3 || indexCount < 3) return;

	uint32_t nrmOff, uvOff, posOff;
	auto nit = m_impl->meshNrmByteOffset.find(mesh);
	if (nit == m_impl->meshNrmByteOffset.end())
	{
		// Indexed meshes UNROLL per IB entry (entry == triangle corner) so the shaders'
		// PrimitiveIndex()*3 addressing keeps working; soup appends its per-vertex arrays 1:1.
		const int entries = indexed ? mesh->numIndices : mesh->numVerts;
		nrmOff = (uint32_t)(m_impl->allNrmCPU.size() * sizeof(float));
		m_impl->allNrmCPU.reserve(m_impl->allNrmCPU.size() + (size_t)entries * 3);
		for (int e = 0; e < entries; ++e)
		{
			const uint32_t v = indexed ? mesh->indexArray[e] : (uint32_t)e;
			m_impl->allNrmCPU.push_back(mesh->normalArray[(size_t)v * 3 + 0]);
			m_impl->allNrmCPU.push_back(mesh->normalArray[(size_t)v * 3 + 1]);
			m_impl->allNrmCPU.push_back(mesh->normalArray[(size_t)v * 3 + 2]);
		}
		m_impl->meshNrmByteOffset[mesh] = nrmOff;
		uvOff = (uint32_t)(m_impl->allUVCPU.size() * sizeof(float));
		if (mesh->uvArray)
		{
			m_impl->allUVCPU.reserve(m_impl->allUVCPU.size() + (size_t)entries * 2);
			for (int e = 0; e < entries; ++e)
			{
				const uint32_t v = indexed ? mesh->indexArray[e] : (uint32_t)e;
				m_impl->allUVCPU.push_back(mesh->uvArray[(size_t)v * 2 + 0]);
				m_impl->allUVCPU.push_back(mesh->uvArray[(size_t)v * 2 + 1]);
			}
		}
		else m_impl->allUVCPU.insert(m_impl->allUVCPU.end(), (size_t)entries * 2, 0.0f);
		m_impl->meshUVByteOffset[mesh] = uvOff;
		posOff = (uint32_t)(m_impl->allPosCPU.size() * sizeof(float));
		m_impl->allPosCPU.reserve(m_impl->allPosCPU.size() + (size_t)entries * 3);
		for (int e = 0; e < entries; ++e)
		{
			const uint32_t v = indexed ? mesh->indexArray[e] : (uint32_t)e;
			m_impl->allPosCPU.push_back(mesh->vertexArray[(size_t)v * 3 + 0]);
			m_impl->allPosCPU.push_back(mesh->vertexArray[(size_t)v * 3 + 1]);
			m_impl->allPosCPU.push_back(mesh->vertexArray[(size_t)v * 3 + 2]);
		}
		m_impl->meshPosByteOffset[mesh] = posOff;
		m_impl->allNrmDirty = true;
	}
	else { nrmOff = nit->second; uvOff = m_impl->meshUVByteOffset[mesh]; posOff = m_impl->meshPosByteOffset[mesh]; }
	// This instance's slice of the pools: a section BLAS numbers its primitives from 0, so the
	// per-instance offsets start at the range's first triangle corner.
	nrmOff += firstIndex * 3 * sizeof(float);
	uvOff  += firstIndex * 2 * sizeof(float);
	posOff += firstIndex * 3 * sizeof(float);

	// Register a texture in the bindless map array; 0xFFFFFFFF when absent or the table is full.
	auto slotFor = [&](Texture* t) -> uint32_t {
		if (!t) return 0xFFFFFFFFu;
		auto tit = m_impl->matTexSlot.find(t);
		if (tit != m_impl->matTexSlot.end()) return tit->second;
		if (m_impl->matTexSRVs.size() >= Impl::kMaxMatTex) return 0xFFFFFFFFu;
		ITextureView* srv = m_impl->GetTexSRV(t);
		if (!srv) return 0xFFFFFFFFu;
		uint32_t idx = (uint32_t)m_impl->matTexSRVs.size();
		m_impl->matTexSRVs.push_back(srv); m_impl->matTexPtr.push_back(t); m_impl->matTexSlot[t] = idx;
		return idx;
	};
	uint32_t texIdx  = mat ? slotFor(mat->diff) : 0xFFFFFFFFu;
	uint32_t nrmIdx  = mat ? slotFor(mat->norm) : 0xFFFFFFFFu;
	uint32_t mrIdx   = mat ? slotFor(mat->mr)   : 0xFFFFFFFFu;
	uint32_t aoIdx   = mat ? slotFor(mat->ao)   : 0xFFFFFFFFu;
	uint32_t emIdx   = mat ? slotFor(mat->em)   : 0xFFFFFFFFu;
	uint32_t specIdx = mat ? slotFor(mat->spec) : 0xFFFFFFFFu;

	float4x4 world = float4x4::Scale(scale[0], scale[1], scale[2])
	               * Diligent::Quaternion<float>(quat[0], quat[1], quat[2], quat[3]).ToMatrix()
	               * float4x4::Translation(pos[0], pos[1], pos[2]);

	TLASBuildInstanceData inst;
	inst.pBLAS    = blas;
	// TLAS visibility bits: 0x01 = reflection rays (RT_REFLECT_MASK), 0x02 = shadow rays; upper bits reserved.
	inst.Mask     = (Uint8)(0xFC | (inReflections ? 0x01 : 0x00) | (castShadows ? 0x02 : 0x00));
	// Every instance stays NON-opaque on purpose: the any-hit alpha test (cutout / hashed / wipe /
	// particle footprints) must run for reflections, and every RayQuery consumer (RT shadows,
	// RT-AO, DDGI probes) commits its candidates explicitly — the material checkboxes keep working.
	inst.Flags    = RAYTRACING_INSTANCE_NONE;
	inst.CustomId = (Uint32)m_impl->rtInstances.size();   // -> g_Instances index (InstanceID() in the shader)
	inst.ContributionToHitGroupIndex = TLAS_INSTANCE_OFFSET_AUTO;   // PER_TLAS binding: offset computed by Diligent

	Impl::RTInstanceData d{}; d.nrmOffset = nrmOff; d.uvOffset = uvOff; d.posOffset = posOff;
	d.texIndex = texIdx; d.nrmTexIndex = nrmIdx; d.mrTexIndex = mrIdx; d.aoTexIndex = aoIdx; d.emTexIndex = emIdx; d.specTexIndex = specIdx;
	d.nrmFlipG = (mat && mat->norm && mat->norm->invertGreen) ? 1u : 0u;   // green convention (OpenGL +Y)
	float alb[4] = {1, 1, 1, 1}, em[3] = {0, 0, 0}; float metal = 0.0f, rough = 0.6f, emI = 0.0f, specF = 1.0f;
	if (mat)
	{
		alb[0] = (float)mat->color.r; alb[1] = (float)mat->color.g; alb[2] = (float)mat->color.b; alb[3] = (float)mat->color.a;
		metal = mat->metallic; rough = mat->roughness; specF = mat->specular;
		em[0] = (float)mat->emissive.r; em[1] = (float)mat->emissive.g; em[2] = (float)mat->emissive.b; emI = mat->emissiveIntensity;
	}
	d.specularFactor = specF;
	d.albedoMetal[0] = alb[0]; d.albedoMetal[1] = alb[1]; d.albedoMetal[2] = alb[2]; d.albedoMetal[3] = metal;
	d.emissiveRough[0] = em[0] * emI; d.emissiveRough[1] = em[1] * emI; d.emissiveRough[2] = em[2] * emI; d.emissiveRough[3] = rough;
	d.colOffset = 0xFFFFFFFFu;
	if (mesh->rtColorArray && mesh->rtDynamic)
	{
		d.colOffset = (uint32_t)(m_impl->rtDynColCPU.size() * sizeof(float));
		m_impl->rtDynColCPU.insert(m_impl->rtDynColCPU.end(), mesh->rtColorArray,
		                           mesh->rtColorArray + (size_t)mesh->numVerts * 4);
	}
	d.shadowShape = (uint32_t)mesh->rtShadowShape;
	d.shadowAlpha = alb[3];
	d.pad0 = 0;

	// MatCB block: must match the raster MatCB packing (NukeDiligent_Scene.cpp). The block is
	// a pure function of the Material — repeats within one accumulation just copy the first
	// build (terrain re-adds dozens of nodes sharing one material every frame).
	d.matByteOffset = (uint32_t)m_impl->allMatCPU.size();
	auto blockIt = mat ? m_impl->rtMatBlockCache.find(mat) : m_impl->rtMatBlockCache.end();
	if (blockIt != m_impl->rtMatBlockCache.end())
	{
		m_impl->allMatCPU.resize(m_impl->allMatCPU.size() + Impl::kMatBlock, 0);
		memcpy(m_impl->allMatCPU.data() + d.matByteOffset,
		       m_impl->allMatCPU.data() + blockIt->second, Impl::kMatBlock);
	}
	else
	{
	m_impl->allMatCPU.resize(m_impl->allMatCPU.size() + Impl::kMatBlock, 0);
	float* mb = reinterpret_cast<float*>(m_impl->allMatCPU.data() + d.matByteOffset);
	mb[0] = alb[0]; mb[1] = alb[1]; mb[2] = alb[2]; mb[3] = alb[3];                       // g_Color@0
	mb[4] = (texIdx != 0xFFFFFFFFu) ? 1.0f : 0.0f; mb[5] = 0.0f; mb[6] = metal; mb[7] = rough;  // g_Params@16
	mb[8] = (mrIdx != 0xFFFFFFFFu) ? 1.0f : 0.0f; mb[9] = (aoIdx != 0xFFFFFFFFu) ? 1.0f : 0.0f;
	mb[10] = (emI > 0.0f) ? 1.0f : 0.0f; mb[11] = specF;                                  // g_Params2 (hasMR,hasAO,hasEm,specularFactor)
	mb[12] = em[0] * emI; mb[13] = em[1] * emI; mb[14] = em[2] * emI; mb[15] = emI;       // g_Emissive2@48
	if (mat && mat->shader)                                                              // custom props overlay
	{
		// Std-included MatCB (module shaders, e.g. "terrain"): raster offsets sit past the
		// 256-byte RT block, so the block carries the hardcoded std head (64 bytes) + the
		// shader's OWN props packed compactly — the same walk GenChitSource emits.
		const bool stdIncluded = !mat->shader->includeProps.empty();
		uint32_t coff = 64;
		for (const nuke::ShaderProp& sp : mat->shader->props)
		{
			uint32_t dst = sp.offset;
			if (stdIncluded)
			{
				if (mat->shader->FromInclude(sp.name)) continue;   // std head is hardcoded above
				if ((coff % 16) + (uint32_t)sp.components * 4 > 16) coff = (coff + 15u) & ~15u;
				dst = coff;
				coff += (uint32_t)sp.components * 4;
			}
			auto pv = mat->props.find(sp.name);
			const float* v = (pv != mat->props.end()) ? pv->second.data() : sp.def;
			if (dst + (uint32_t)sp.components * 4 <= Impl::kMatBlock)
				memcpy(m_impl->allMatCPU.data() + d.matByteOffset + dst, v, (size_t)sp.components * 4);
		}
	}
	if (mat) m_impl->rtMatBlockCache[mat] = d.matByteOffset;
	}

	m_impl->rtInstShaderGuid.push_back(mat ? mat->shaderGuid : std::string());
	m_impl->rtInstData.push_back(d);
	// InstanceMatrix is 3x4 row-major; our world matrix is row-vector (v*M), hence the transpose.
	for (int r = 0; r < 3; ++r)
		for (int c = 0; c < 4; ++c)
			inst.Transform.data[r][c] = world.m[c][r];

	if (m_impl->rtInstanceNames.size() <= m_impl->rtInstances.size())
		m_impl->rtInstanceNames.push_back("i" + std::to_string(m_impl->rtInstances.size()));
	m_impl->rtInstances.push_back(inst);
}

// Run the NukeBend compute over every bend mesh seen this frame and rebuild its BLAS over
// the bent positions. Sets blasBentThisFrame.
void NukeDiligent::Impl::BendRTMeshes()
{
	blasBentThisFrame = false;
	if (!bendCSPSO || !bendCSSRB || rtBendMeshes.empty()) return;
	std::sort(rtBendMeshes.begin(), rtBendMeshes.end());
	rtBendMeshes.erase(std::unique(rtBendMeshes.begin(), rtBendMeshes.end()), rtBendMeshes.end());
	for (Mesh* m : rtBendMeshes)
	{
		MeshGPU* gp = GetMeshGPU(m);
		if (!gp || !gp->posBent || !gp->bendSrc || !gp->bendData || !gp->bendPivot) continue;
		auto bit = blasCache.find(m);
		if (bit == blasCache.end() || !bit->second || !gp->blasScratch) continue;
		{
			MapHelper<float> pc(context, bendCSParamsCB, MAP_WRITE, MAP_FLAG_DISCARD);
			if (pc == nullptr) continue;
			pc[0] = pc[1] = pc[2] = 0.0f;                // g_AtomOffset (layers live near identity)
			*((Uint32*)&pc[3]) = (Uint32)gp->numVerts;   // g_VertCount (std140: uint in the float3's tail)
		}
		auto set = [&](const char* n, IDeviceObject* o)
		{ if (auto* v = bendCSSRB->GetVariableByName(SHADER_TYPE_COMPUTE, n)) v->Set(o); };
		set("g_SrcPos",    gp->bendSrc->GetDefaultView(BUFFER_VIEW_SHADER_RESOURCE));
		set("g_BendData",  gp->bendData->GetDefaultView(BUFFER_VIEW_SHADER_RESOURCE));
		set("g_BendPivot", gp->bendPivot->GetDefaultView(BUFFER_VIEW_SHADER_RESOURCE));
		set("g_DstPos",    gp->posBent->GetDefaultView(BUFFER_VIEW_UNORDERED_ACCESS));
		context->SetPipelineState(bendCSPSO);
		context->CommitShaderResources(bendCSSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		DispatchComputeAttribs da(((Uint32)gp->numVerts + 63) / 64, 1, 1);
		context->DispatchCompute(da);

		BLASBuildTriangleData td;
		td.GeometryName         = "geo";
		td.pVertexBuffer        = gp->posBent;
		td.VertexStride         = 3 * sizeof(float);
		td.VertexCount          = (Uint32)gp->numVerts;
		td.VertexValueType      = VT_FLOAT32;
		td.VertexComponentCount = 3;
		td.PrimitiveCount       = (Uint32)(gp->numVerts / 3);
		td.Flags                = RAYTRACING_GEOMETRY_FLAG_OPAQUE;
		BuildBLASAttribs ba;
		ba.pBLAS                  = bit->second;
		ba.pTriangleData          = &td;
		ba.TriangleDataCount      = 1;
		ba.pScratchBuffer         = gp->blasScratch;
		ba.BLASTransitionMode     = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
		ba.GeometryTransitionMode = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
		ba.ScratchBufferTransitionMode = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
		context->BuildBLAS(ba);
		blasBentThisFrame = true;
	}
}

// Rebuild the cached BLAS of every mesh whose vertices the engine rewrote this frame.
// Only sets blasBentThisFrame — BendRTMeshes owns the reset.
void NukeDiligent::Impl::RebuildDynamicBLAS()
{
	if (rtDynMeshes.empty()) return;
	std::sort(rtDynMeshes.begin(), rtDynMeshes.end());
	rtDynMeshes.erase(std::unique(rtDynMeshes.begin(), rtDynMeshes.end()), rtDynMeshes.end());
	for (Mesh* m : rtDynMeshes)
	{
		MeshGPU* gp = GetMeshGPU(m);
		auto bit = blasCache.find(m);
		if (!gp || !gp->pos || bit == blasCache.end() || !bit->second || !gp->blasScratch) continue;
		BLASBuildTriangleData td;
		td.GeometryName         = "geo";
		td.pVertexBuffer        = gp->pos;
		td.VertexStride         = 3 * sizeof(float);
		td.VertexCount          = (Uint32)gp->numVerts;
		td.VertexValueType      = VT_FLOAT32;
		td.VertexComponentCount = 3;
		td.PrimitiveCount       = (Uint32)(gp->numVerts / 3);
		td.Flags                = m->rtAlphaTested ? RAYTRACING_GEOMETRY_FLAG_NONE : RAYTRACING_GEOMETRY_FLAG_OPAQUE;
		BuildBLASAttribs ba;
		ba.pBLAS                  = bit->second;
		ba.pTriangleData          = &td;
		ba.TriangleDataCount      = 1;
		ba.pScratchBuffer         = gp->blasScratch;
		ba.BLASTransitionMode     = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
		ba.GeometryTransitionMode = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
		ba.ScratchBufferTransitionMode = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
		context->BuildBLAS(ba);
		blasBentThisFrame = true;
		static bool logged = false;
		if (!logged) { logged = true; cout << "[NukeDiligent]\tdynamic BLAS rebuild active (particles)" << endl; }
	}
}

void NukeDiligent::buildRTScene()
{
	auto* d = m_impl;
	if (!d->rtSupported || d->rtInstances.empty()) { d->rtSceneReady = false; return; }
	Impl::GpuMark gm(d, "rt.build");   // BLAS bends/rebuilds + TLAS build/refit
	d->BendRTMeshes();         // sway the foliage BLASes BEFORE the TLAS build/refit
	d->RebuildDynamicBLAS();   // particle quad BLASes follow this frame's vertex data
	const Uint32 count = (Uint32)d->rtInstances.size();
	// Re-point InstanceName: the names vector may have reallocated during accumulation.
	for (Uint32 i = 0; i < count; ++i) d->rtInstances[i].InstanceName = d->rtInstanceNames[i].c_str();

	// Static skip: a byte-identical scene keeps last frame's TLAS/buffers untouched.
	{
		// FNV over 8-byte words (byte tail): the scene blob is tens of KB and this runs every
		// frame, so the byte-at-a-time walk was itself showing up in the profile.
		auto fnv = [](uint64_t h, const void* p, size_t n)
		{
			const unsigned char* b = (const unsigned char*)p;
			for (; n >= 8; b += 8, n -= 8) { uint64_t w; memcpy(&w, b, 8); h ^= w; h *= 1099511628211ull; }
			for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
			return h;
		};
		uint64_t full = 1469598103934665603ull;
		for (Uint32 i = 0; i < count; ++i)
		{
			const auto& ins = d->rtInstances[i];
			full = fnv(full, &ins.pBLAS, sizeof(ins.pBLAS));
			full = fnv(full, &ins.Mask, sizeof(ins.Mask));
			full = fnv(full, &ins.CustomId, sizeof(ins.CustomId));
			full = fnv(full, &ins.Transform, sizeof(ins.Transform));
		}
		if (!d->rtInstData.empty()) full = fnv(full, d->rtInstData.data(), d->rtInstData.size() * sizeof(Impl::RTInstanceData));
		if (!d->allMatCPU.empty())  full = fnv(full, d->allMatCPU.data(), d->allMatCPU.size());
		const bool unchanged = full == d->lastRTFullSig && d->rtSceneReady && d->tlas && !d->allNrmDirty &&
		                       !d->blasBentThisFrame;   // bent BLAS -> the TLAS must refit over it
		d->lastRTFullSig = full;
		if (unchanged) return;
	}

	bool recreated = false;
	if (!d->tlas || d->tlasMaxInstances < count)   // (re)create when capacity grows
	{
		recreated = true;
		d->Trash(d->tlas); d->Trash(d->tlasScratch); d->Trash(d->tlasInstanceBuf);
		d->tlas.Release(); d->tlasScratch.Release(); d->tlasInstanceBuf.Release();
		TopLevelASDesc td; td.Name = "Scene TLAS"; td.MaxInstanceCount = count;
		td.Flags = RAYTRACING_BUILD_AS_PREFER_FAST_TRACE | RAYTRACING_BUILD_AS_ALLOW_UPDATE;   // allow per-frame refit
		d->device->CreateTLAS(td, &d->tlas);
		if (!d->tlas) { d->rtSceneReady = false; return; }
		d->tlasMaxInstances = count;
		BufferDesc sbd; sbd.Name = "TLAS scratch"; sbd.Usage = USAGE_DEFAULT; sbd.BindFlags = BIND_RAY_TRACING;
		auto ssz = d->tlas->GetScratchBufferSizes(); sbd.Size = (ssz.Build > ssz.Update) ? ssz.Build : ssz.Update;
		d->device->CreateBuffer(sbd, nullptr, &d->tlasScratch);
		BufferDesc ibd; ibd.Name = "TLAS instances"; ibd.Usage = USAGE_DEFAULT; ibd.BindFlags = BIND_RAY_TRACING;
		ibd.Size = Uint64{TLAS_INSTANCE_DATA_SIZE} * count;
		d->device->CreateBuffer(ibd, nullptr, &d->tlasInstanceBuf);
	}
	if (!d->tlasScratch || !d->tlasInstanceBuf) { d->rtSceneReady = false; return; }

	// Refit only while the topology (count + BLAS set) holds; periodic full rebuild keeps AS quality.
	size_t sig = count;
	for (Uint32 i = 0; i < count; ++i) sig = sig * 1315423911ull + (size_t)d->rtInstances[i].pBLAS;
	bool refit = !recreated && sig == d->lastTlasSig && (d->tlasFrameCtr % 32 != 0);
	d->lastTlasSig = sig; ++d->tlasFrameCtr;

	BuildTLASAttribs ba;
	ba.pTLAS                        = d->tlas;
	ba.pInstances                   = d->rtInstances.data();
	ba.InstanceCount                = count;
	ba.pInstanceBuffer              = d->tlasInstanceBuf;
	ba.pScratchBuffer               = d->tlasScratch;
	ba.Update                       = refit;                                 // perf: refit moved transforms vs full rebuild
	ba.BindingMode                  = HIT_GROUP_BINDING_MODE_PER_INSTANCE;   // per-instance hit group (per-shader RT shading)
	ba.HitGroupStride               = 1;                                     // one ray type (reflection)
	ba.TLASTransitionMode           = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
	ba.BLASTransitionMode           = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
	ba.InstanceBufferTransitionMode = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
	ba.ScratchBufferTransitionMode  = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
	d->context->BuildTLAS(ba);

	if (d->allNrmDirty && !d->allNrmCPU.empty())
	{
		d->Trash(d->rtNrmBuf);
		d->rtNrmBuf.Release(); d->rtNrmSRV = nullptr;
		BufferDesc bd; bd.Name = "RT Normals"; bd.Usage = USAGE_IMMUTABLE; bd.BindFlags = BIND_SHADER_RESOURCE;
		bd.Mode = BUFFER_MODE_RAW; bd.Size = (Uint64)d->allNrmCPU.size() * sizeof(float);
		BufferData bdat{d->allNrmCPU.data(), bd.Size};
		d->device->CreateBuffer(bd, &bdat, &d->rtNrmBuf);
		if (d->rtNrmBuf) d->rtNrmSRV = d->rtNrmBuf->GetDefaultView(BUFFER_VIEW_SHADER_RESOURCE);

		d->Trash(d->rtUVBuf);
		d->rtUVBuf.Release(); d->rtUVSRV = nullptr;
		if (!d->allUVCPU.empty())
		{
			BufferDesc ud; ud.Name = "RT UVs"; ud.Usage = USAGE_IMMUTABLE; ud.BindFlags = BIND_SHADER_RESOURCE;
			ud.Mode = BUFFER_MODE_RAW; ud.Size = (Uint64)d->allUVCPU.size() * sizeof(float);
			BufferData udat{d->allUVCPU.data(), ud.Size};
			d->device->CreateBuffer(ud, &udat, &d->rtUVBuf);
			if (d->rtUVBuf) d->rtUVSRV = d->rtUVBuf->GetDefaultView(BUFFER_VIEW_SHADER_RESOURCE);
		}

		d->Trash(d->rtPosBuf);
		d->rtPosBuf.Release(); d->rtPosSRV = nullptr;
		if (!d->allPosCPU.empty())
		{
			BufferDesc pd; pd.Name = "RT Positions"; pd.Usage = USAGE_IMMUTABLE; pd.BindFlags = BIND_SHADER_RESOURCE;
			pd.Mode = BUFFER_MODE_RAW; pd.Size = (Uint64)d->allPosCPU.size() * sizeof(float);
			BufferData pdat{d->allPosCPU.data(), pd.Size};
			d->device->CreateBuffer(pd, &pdat, &d->rtPosBuf);
			if (d->rtPosBuf) d->rtPosSRV = d->rtPosBuf->GetDefaultView(BUFFER_VIEW_SHADER_RESOURCE);
		}
		d->allNrmDirty = false;
	}
	if (!d->rtInstBuf || d->rtInstCapacity < count)
	{
		d->Trash(d->rtInstBuf);
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

	// Per-instance MatCB blocks (g_MatBytes) for auto-generated hit shaders — RAW buffer, grows as needed.
	if (!d->allMatCPU.empty())
	{
		if (!d->rtMatBuf || d->rtMatCapacity < (uint32_t)d->allMatCPU.size())
		{
			d->Trash(d->rtMatBuf);
			d->rtMatBuf.Release(); d->rtMatSRV = nullptr; d->rtMatCapacity = (uint32_t)d->allMatCPU.size();
			BufferDesc bd; bd.Name = "RT MatBytes"; bd.Usage = USAGE_DEFAULT; bd.BindFlags = BIND_SHADER_RESOURCE;
			bd.Mode = BUFFER_MODE_RAW; bd.Size = (Uint64)d->allMatCPU.size();
			d->device->CreateBuffer(bd, nullptr, &d->rtMatBuf);
			if (d->rtMatBuf) d->rtMatSRV = d->rtMatBuf->GetDefaultView(BUFFER_VIEW_SHADER_RESOURCE);
		}
		if (d->rtMatBuf)
			d->context->UpdateBuffer(d->rtMatBuf, 0, (Uint64)d->allMatCPU.size(), d->allMatCPU.data(), RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	}

	if (!d->rtDynColCPU.empty())
	{
		const Uint64 need = (Uint64)d->rtDynColCPU.size() * sizeof(float);
		if (!d->rtDynColBuf || d->rtDynColCap < need)
		{
			d->Trash(d->rtDynColBuf);
			d->rtDynColBuf.Release(); d->rtDynColSRV = nullptr; d->rtDynColCap = need;
			BufferDesc bd; bd.Name = "RT DynColors"; bd.Usage = USAGE_DEFAULT; bd.BindFlags = BIND_SHADER_RESOURCE;
			bd.Mode = BUFFER_MODE_RAW; bd.Size = need;
			d->device->CreateBuffer(bd, nullptr, &d->rtDynColBuf);
			if (d->rtDynColBuf) d->rtDynColSRV = d->rtDynColBuf->GetDefaultView(BUFFER_VIEW_SHADER_RESOURCE);
		}
		if (d->rtDynColBuf)
			d->context->UpdateBuffer(d->rtDynColBuf, 0, need, d->rtDynColCPU.data(), RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	}

	d->rtSceneReady = true;
}

// Build the DXR reflection pipeline (ray-gen + miss + closest-hit + any-hit + SBT), including
// an auto-generated closest-hit per custom material shader. Returns false on failure.
bool NukeDiligent::Impl::BuildRTPipeline()
{
	if (rtPSO && !rtPipelineDirty) return true;
	auto rtSf = ShaderFactory();
	if (!rtSupported || !rtSf) return false;
	if (rtPSO) { rtPSO.Release(); rtSRB.Release(); rtSBT.Release(); }   // a new surf shader appeared -> rebuild
	rtPipelineDirty = false;

	auto mk = [&](const char* file, SHADER_TYPE type, const char* dbg, RefCntAutoPtr<IShader>& out)
	{
		ShaderCreateInfo sci; sci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
		sci.ShaderCompiler = SHADER_COMPILER_DXC; sci.HLSLVersion = ShaderVersion{6, 5};
		sci.pShaderSourceStreamFactory = rtSf;            // loads the file + resolves #include "rt_common.hlsl"
		sci.FilePath = file; sci.EntryPoint = "main"; sci.Desc = {dbg, type, true};
		device->CreateShader(sci, &out);
		if (!out) cout << "[NukeDiligent]\tRT shader build failed: " << file << endl;
		return (bool)out;
	};
	RefCntAutoPtr<IShader> rg, rm, rch, rah;
	if (!mk("rt_rgen.hlsl",  SHADER_TYPE_RAY_GEN,         "RT RayGen",     rg))  return false;
	if (!mk("rt_rmiss.hlsl", SHADER_TYPE_RAY_MISS,        "RT Miss",       rm))  return false;
	if (!mk("rt_rchit.hlsl", SHADER_TYPE_RAY_CLOSEST_HIT, "RT ClosestHit", rch)) return false;
	// Any-hit alpha test, shared by every hit group; only runs for non-opaque geometry.
	if (!mk("rt_ahit.hlsl",  SHADER_TYPE_RAY_ANY_HIT,     "RT AnyHit",     rah)) return false;

	// Auto-generate a closest-hit per material shader that ships a "<name>.surf.hlsl" (codegen from its schema).
	auto mkSrc = [&](const std::string& src, const char* dbg, RefCntAutoPtr<IShader>& out)
	{
		ShaderCreateInfo sci; sci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
		sci.ShaderCompiler = SHADER_COMPILER_DXC; sci.HLSLVersion = ShaderVersion{6, 5};
		sci.pShaderSourceStreamFactory = rtSf;   // resolves #include "rt_common.hlsl" / "<name>.surf.hlsl"
		sci.Source = src.c_str(); sci.EntryPoint = "main"; sci.Desc = {dbg, SHADER_TYPE_RAY_CLOSEST_HIT, true};
		device->CreateShader(sci, &out);
		if (!out) cout << "[NukeDiligent]\tRT chit codegen failed: " << dbg << endl;
		return (bool)out;
	};
	std::vector<RefCntAutoPtr<IShader>> customChits;
	std::vector<std::string> hitNames; hitNames.push_back("HitGroup");
	shaderHitGroup.clear();
	for (auto& kv : rtSurfShaders)
	{
		RefCntAutoPtr<IShader> c; std::string dbg = "RT chit " + kv.first;
		if (!mkSrc(GenChitSource(kv.first, kv.second), dbg.c_str(), c)) continue;   // broken custom shader -> default chit still applies
		customChits.push_back(c);
		hitNames.push_back("HitGroup_" + kv.first);
		shaderHitGroup[kv.first] = hitNames.back();
		cout << "[NukeDiligent]\tRT auto hit group for shader '" << kv.first << "'" << endl;
	}

	RayTracingPipelineStateCreateInfo ci;
	ci.PSODesc.Name = "RT Reflect PSO";
	ci.PSODesc.PipelineType = PIPELINE_TYPE_RAY_TRACING;
	RayTracingGeneralShaderGroup gen[2] = { {"Main", rg}, {"Miss", rm} };
	std::vector<RayTracingTriangleHitShaderGroup> hit;
	hit.push_back({hitNames[0].c_str(), rch, rah});                              // default: standard PBR + alpha any-hit
	for (size_t i = 0; i < customChits.size(); ++i) hit.push_back({hitNames[i + 1].c_str(), customChits[i], rah});
	ci.pGeneralShaders = gen; ci.GeneralShaderCount = 2;
	ci.pTriangleHitShaders = hit.data(); ci.TriangleHitShaderCount = (Uint32)hit.size();
	ci.RayTracingPipeline.MaxRecursionDepth = 8;       // primary + bounces; the configured depth caps actual recursion
	ci.RayTracingPipeline.ShaderRecordSize  = 0;
	ci.MaxAttributeSize = sizeof(float) * 2;           // BuiltInTriangleIntersectionAttributes (barycentrics)
	ci.MaxPayloadSize   = sizeof(float) * 4;           // RTPayload { float3 color; uint depth; }

	SamplerDesc samp; samp.MinFilter = FILTER_TYPE_LINEAR; samp.MagFilter = FILTER_TYPE_LINEAR; samp.MipFilter = FILTER_TYPE_LINEAR;
	samp.AddressU = TEXTURE_ADDRESS_CLAMP; samp.AddressV = TEXTURE_ADDRESS_CLAMP; samp.AddressW = TEXTURE_ADDRESS_CLAMP;
	ImmutableSamplerDesc imms[] = {
		{SHADER_TYPE_ALL_RAY_TRACING, "g_Probe",  samp},
		{SHADER_TYPE_ALL_RAY_TRACING, "g_MatTex", samp},
	};
	ShaderResourceVariableDesc vars[] = {
		{SHADER_TYPE_ALL_RAY_TRACING, "RTRefCB", SHADER_RESOURCE_VARIABLE_TYPE_STATIC},
		{SHADER_TYPE_ALL_RAY_TRACING, "FrameCB", SHADER_RESOURCE_VARIABLE_TYPE_STATIC},
	};
	ci.PSODesc.ResourceLayout.DefaultVariableType = SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC;   // TLAS/gbuffer/bindless/output
	ci.PSODesc.ResourceLayout.Variables           = vars; ci.PSODesc.ResourceLayout.NumVariables = 2;
	ci.PSODesc.ResourceLayout.ImmutableSamplers   = imms; ci.PSODesc.ResourceLayout.NumImmutableSamplers = 2;

	device->CreateRayTracingPipelineState(ci, &rtPSO);
	if (!rtPSO) { cout << "[NukeDiligent]\tRT pipeline PSO build failed" << endl; return false; }
	for (SHADER_TYPE t : {SHADER_TYPE_RAY_GEN, SHADER_TYPE_RAY_MISS, SHADER_TYPE_RAY_CLOSEST_HIT, SHADER_TYPE_RAY_ANY_HIT})
	{
		if (auto* v = rtPSO->GetStaticVariableByName(t, "RTRefCB")) v->Set(rtRefCB);
		if (auto* v = rtPSO->GetStaticVariableByName(t, "FrameCB")) v->Set(worldFrameCB);
	}
	rtPSO->CreateShaderResourceBinding(&rtSRB, true);

	ShaderBindingTableDesc sd; sd.Name = "RT Reflect SBT"; sd.pPSO = rtPSO;
	device->CreateSBT(sd, &rtSBT);
	if (!rtSBT) { cout << "[NukeDiligent]\tRT SBT creation failed" << endl; return false; }
	rtSBT->BindRayGenShader("Main");
	rtSBT->BindMissShader("Miss", 0);                    // hit group is (re)bound per frame in RunRTReflectPipeline
	cout << "[NukeDiligent]\tRT reflection pipeline ready" << endl;
	return true;
}

void NukeDiligent::Impl::EnsureRTOutput(int w, int h)
{
	if (w <= 0 || h <= 0) return;
	if (rtOutTex && rtOutW == w && rtOutH == h) return;
	// Per-size cache: a shared UAV must never be released/recreated mid-frame.
	const uint64_t key = ((uint64_t)(uint32_t)w << 32) | (uint32_t)h;
	SizedTexSet& s = rtOutCache[key];
	if (!s.a)
	{
		TextureDesc td; td.Name = "RT Reflect Output"; td.Type = RESOURCE_DIM_TEX_2D;
		td.Width = (Uint32)w; td.Height = (Uint32)h; td.Format = HDR_FMT;
		td.BindFlags = BIND_UNORDERED_ACCESS | BIND_SHADER_RESOURCE;
		device->CreateTexture(td, nullptr, &s.a);
	}
	rtOutTex = s.a;
	rtOutW = w; rtOutH = h;
	s.lastUsed = ++sizedClock;
	EvictSized(rtOutCache, key);
}

void NukeDiligent::Impl::RunRTReflectPipeline(ITextureView* srcSRV, ITexture* dstTex, int w, int h, const std::vector<float>& params)
{
	if (!dstTex || !srcSRV) return;
	// TraceRays/CopyTexture must not target a still-bound render target — unbind first.
	context->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE);
	// A custom shader appeared/changed -> rebuild with its hit group, in the background; the
	// pass passes the colour through until the new pipeline lands.
	if (rtPipelineDirty && !rtBuilding.exchange(true))
		EnqueueBuild([this] { BuildRTPipeline(); }, [this] { rtBuilding = false; }, kPrioRT, "RT reflection pipeline");
	// No scene to trace -> pass the chain colour through. Must be BlitTexture, not CopyTexture:
	// source and destination formats differ (RGBA8 scene vs RGBA16F scratch).
	if (rtBuilding || !rtPSO || !rtSBT || !rtSceneReady || !tlas)
	{
		BlitTexture(srcSRV, dstTex);
		return;
	}
	EnsureRTOutput(w, h);
	if (!rtOutTex) return;

	{   // RTRefCB: clip->view + view->world + camera + (intensity, maxDist, maxDepth) + water
		struct CB { float4x4 ip, iv; float4 cam; float4 prm; float4 waterOcc; float4 waterCol; float4 waterAbs; };
		MapHelper<CB> cb(context, rtRefCB, MAP_WRITE, MAP_FLAG_DISCARD);
		cb->ip  = curProjNoJitter.Inverse(); cb->iv = curView.Inverse();   // unjittered: must match the gbuffer depth
		cb->cam = float4(curCamPos[0], curCamPos[1], curCamPos[2], 1.0f);
		float intensity = rtCfgIntensity;
		float maxDist   = rtCfgMaxDist;
		float maxDepth  = (float)rtCfgBounces;
		float roughCut  = rtCfgRoughCut;
		maxDepth = (maxDepth < 1.0f) ? 1.0f : (maxDepth > 7.0f ? 7.0f : maxDepth);   // PSO MaxRecursionDepth = 8
		if (roughCut < 0.05f) roughCut = 0.05f;
		cb->prm = float4(intensity, maxDist, maxDepth, roughCut);
		// Water occlusion state, published by the water module via SetRTWaterState.
		cb->waterOcc = float4(rtWaterOcc[0], rtWaterOcc[1], rtWaterOcc[2], 0.0f);
		cb->waterCol = float4(rtWaterCol[0], rtWaterCol[1], rtWaterCol[2], 0.0f);
		cb->waterAbs = float4(rtWaterAbs[0], rtWaterAbs[1], rtWaterAbs[2], rtWaterOcc[2]);
	}

	// Bind dynamic resources for every RT stage that references them (null lookups are harmless).
	auto setv = [&](const char* n, IDeviceObject* o)
	{
		if (auto* v = rtSRB->GetVariableByName(SHADER_TYPE_RAY_GEN, n))         v->Set(o);
		if (auto* v = rtSRB->GetVariableByName(SHADER_TYPE_RAY_MISS, n))        v->Set(o);
		if (auto* v = rtSRB->GetVariableByName(SHADER_TYPE_RAY_CLOSEST_HIT, n)) v->Set(o);
		if (auto* v = rtSRB->GetVariableByName(SHADER_TYPE_RAY_ANY_HIT, n))     v->Set(o);   // alpha-test any-hit
	};
	setv("g_TLAS",     (IDeviceObject*)tlas.RawPtr());
	setv("g_Output",   rtOutTex->GetDefaultView(TEXTURE_VIEW_UNORDERED_ACCESS));
	setv("g_GBuffer",  gbufSRV);
	setv("g_Depth",    gbufDepthSRV);
	setv("g_Source",   srcSRV);
	setv("g_Probe",    (probeActive && probeCubeSRV) ? probeCubeSRV : fallbackCubeSRV);
	setv("g_AllNrm",   rtNrmSRV);
	setv("g_AllUV",    rtUVSRV ? rtUVSRV : rtNrmSRV);
	setv("g_AllPos",   rtPosSRV ? rtPosSRV : rtNrmSRV);
	setv("g_Instances",rtInstSRV);
	setv("g_MatBytes", rtMatSRV ? rtMatSRV : rtInstSRV);   // fallback must be a valid non-null SRV
	setv("g_DynCol",   rtDynColSRV ? rtDynColSRV : rtNrmSRV);   // fallback must be a valid RAW SRV
	{   // bindless albedo array (re-resolve each frame -> animated textures update)
		ITextureView* white = whiteTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
		IDeviceObject* arr[Impl::kMaxMatTex];
		for (uint32_t k = 0; k < kMaxMatTex; ++k)
		{
			if (k < matTexSRVs.size()) { if (ITextureView* s = GetTexSRV(matTexPtr[k])) matTexSRVs[k] = s; arr[k] = matTexSRVs[k] ? matTexSRVs[k] : white; }
			else arr[k] = white;
		}
		if (auto* v = rtSRB->GetVariableByName(SHADER_TYPE_RAY_CLOSEST_HIT, "g_MatTex")) v->SetArray(arr, 0, kMaxMatTex);
		if (auto* v = rtSRB->GetVariableByName(SHADER_TYPE_RAY_ANY_HIT,     "g_MatTex")) v->SetArray(arr, 0, kMaxMatTex);
	}

	// The SBT accumulates hit-group bindings by instance name, so it must be reset before
	// re-binding — stale entries make TraceRays reject the instance-to-shader mapping.
	rtSBT->ResetHitGroups();
	for (size_t i = 0; i < rtInstData.size() && i < rtInstanceNames.size(); ++i)
	{
		const char* group = "HitGroup";   // default = standard PBR
		if (i < rtInstShaderGuid.size()) { auto it = shaderHitGroup.find(rtInstShaderGuid[i]); if (it != shaderHitGroup.end()) group = it->second.c_str(); }
		rtSBT->BindHitGroupForInstance(tlas, rtInstanceNames[i].c_str(), 0, group);
	}
	context->SetPipelineState(rtPSO);
	context->CommitShaderResources(rtSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	context->UpdateSBT(rtSBT);
	TraceRaysAttribs ta; ta.pSBT = rtSBT; ta.DimensionX = (Uint32)w; ta.DimensionY = (Uint32)h; ta.DimensionZ = 1;
	context->TraceRays(ta);

	// Format-safe: rtOut is fixed RGBA16F, the destination may be the RGBA8 scene target.
	BlitTexture(rtOutTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE), dstTex);
}
