#include "NukeDiligentImpl.h"
#include <sstream>
#include <cctype>

// float/float2/int4/... -> component count (0 = unsupported, skip). Mirrors the engine's MatCB schema parse.
static int RTCompsOf(const std::string& t)
{
	if (t == "float" || t == "int" || t == "uint" || t == "bool") return 1;
	if (t == "float2" || t == "int2" || t == "uint2") return 2;
	if (t == "float3" || t == "int3" || t == "uint3") return 3;
	if (t == "float4" || t == "int4" || t == "uint4") return 4;
	return 0;
}

// Generate a closest-hit shader for a material shader from its MatCB schema + its "<name>.surf.hlsl".
// Per-field statics are loaded from the per-instance MatCB block (g_MatBytes); the shader's Surface() then
// runs verbatim (same code as the raster pass) and the harness lights/recurses the result.
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
			for (size_t i = 0; i < raw.size(); )   // strip // comments
			{ if (raw[i] == '/' && i + 1 < raw.size() && raw[i + 1] == '/') { while (i < raw.size() && raw[i] != '\n') ++i; } else b += raw[i++]; }
			uint32_t off = 0;
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
	  << "#define MAT_BASE_TEX(uv) ((__texIndex!=0xFFFFFFFFu)? g_MatTex[NonUniformResourceIndex(__texIndex)].SampleLevel(g_MatTex_sampler,(uv),0) : float4(1,1,1,1))\n"
	  << "#include \"" << name << ".surf.hlsl\"\n"
	  << "[shader(\"closesthit\")] void main(inout RTPayload p, in BuiltInTriangleIntersectionAttributes attr){\n"
	  << "  RTInstanceData inst = g_Instances[InstanceID()];\n"
	  << "  __texIndex = inst.texIndex; __LoadMat(inst.matByteOffset);\n"
	  << "  float3 wdir = WorldRayDirection();\n"
	  << "  SurfaceIn IN;\n"
	  << "  IN.uv = FetchUV(inst.uvOffset, PrimitiveIndex(), attr.barycentrics);\n"
	  << "  IN.worldNormal = FetchWorldNormal(inst.nrmOffset, PrimitiveIndex(), attr.barycentrics, ObjectToWorld3x4());\n"
	  << "  if (dot(IN.worldNormal,wdir)>0.0) IN.worldNormal=-IN.worldNormal;\n"
	  << "  IN.worldPos = WorldRayOrigin()+wdir*RayTCurrent(); IN.viewDir=-wdir;\n"
	  << "  SurfaceOut O=(SurfaceOut)0; O.albedo=float3(1,1,1); O.roughness=1.0; O.alpha=1.0; O.unlit=false;\n"
	  << "  Surface(IN,O);\n"
	  << "  if (O.unlit){ p.color=O.emissive; return; }\n"
	  << "  float3 col = ShadeSurface(IN.worldPos,IN.worldNormal,IN.viewDir,O.albedo,O.metallic,O.roughness,O.emissive);\n"
	  << "  float3 R=reflect(wdir,IN.worldNormal); float3 env=ReflEnv(R,O.roughness), traced=env;\n"
	  << "  if (p.depth<(uint)g_RTParams.z){ RayDesc ray; ray.Origin=IN.worldPos+IN.worldNormal*0.08+R*0.05; ray.Direction=R; ray.TMin=0.02; ray.TMax=(g_RTParams.y>0.5)?g_RTParams.y:1000.0; RTPayload p2; p2.color=0.0; p2.depth=p.depth+1; TraceRay(g_TLAS,RAY_FLAG_NONE,0xFF,0,1,0,ray,p2); traced=p2.color; }\n"
	  << "  col += SpecFr(IN.worldNormal,IN.viewDir,O.roughness,O.albedo,O.metallic)*lerp(traced,env,O.roughness);\n"
	  << "  p.color=col;\n}\n";
	return s.str();
}

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
	m_impl->rtInstanceNames.clear();
	m_impl->rtInstData.clear();
	m_impl->rtInstShaderGuid.clear();
	m_impl->allMatCPU.clear();
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
	inst.CustomId = (Uint32)m_impl->rtInstances.size();   // -> g_Instances index (InstanceID() in the shader)
	inst.ContributionToHitGroupIndex = TLAS_INSTANCE_OFFSET_AUTO;   // PER_TLAS binding: offset computed by Diligent

	// Per-instance material. albedoMetal/emissiveRough feed the standard PBR hit shader; the full MatCB block
	// (g_MatBytes) feeds auto-generated per-shader hit shaders (they load their own params from it).
	Impl::RTInstanceData d{}; d.nrmOffset = nrmOff; d.uvOffset = uvOff; d.texIndex = texIdx;
	float alb[4] = {1, 1, 1, 1}, em[3] = {0, 0, 0}; float metal = 0.0f, rough = 0.6f, emI = 0.0f;
	if (mat)
	{
		alb[0] = (float)mat->color.r; alb[1] = (float)mat->color.g; alb[2] = (float)mat->color.b; alb[3] = (float)mat->color.a;
		metal = mat->metallic; rough = mat->roughness;
		em[0] = (float)mat->emissive.r; em[1] = (float)mat->emissive.g; em[2] = (float)mat->emissive.b; emI = mat->emissiveIntensity;
	}
	d.albedoMetal[0] = alb[0]; d.albedoMetal[1] = alb[1]; d.albedoMetal[2] = alb[2]; d.albedoMetal[3] = metal;
	d.emissiveRough[0] = em[0] * emI; d.emissiveRough[1] = em[1] * emI; d.emissiveRough[2] = em[2] * emI; d.emissiveRough[3] = rough;

	// MatCB block — same packing as the raster MatCB (NukeDiligent_Scene.cpp): standard fields + custom props.
	d.matByteOffset = (uint32_t)m_impl->allMatCPU.size();
	m_impl->allMatCPU.resize(m_impl->allMatCPU.size() + Impl::kMatBlock, 0);
	float* mb = reinterpret_cast<float*>(m_impl->allMatCPU.data() + d.matByteOffset);
	mb[0] = alb[0]; mb[1] = alb[1]; mb[2] = alb[2]; mb[3] = alb[3];                       // g_Color@0
	mb[4] = (texIdx != 0xFFFFFFFFu) ? 1.0f : 0.0f; mb[5] = 0.0f; mb[6] = metal; mb[7] = rough;  // g_Params@16
	mb[8] = 0.0f; mb[9] = 0.0f; mb[10] = (emI > 0.0f) ? 1.0f : 0.0f; mb[11] = 1.0f;       // g_Params2@32
	mb[12] = em[0] * emI; mb[13] = em[1] * emI; mb[14] = em[2] * emI; mb[15] = emI;       // g_Emissive2@48
	if (mat && mat->shader)                                                              // custom props overlay
		for (const nuke::ShaderProp& sp : mat->shader->props)
		{
			auto pv = mat->props.find(sp.name);
			const float* v = (pv != mat->props.end()) ? pv->second.data() : sp.def;
			if (sp.offset + (uint32_t)sp.components * 4 <= Impl::kMatBlock)
				memcpy(m_impl->allMatCPU.data() + d.matByteOffset + sp.offset, v, (size_t)sp.components * 4);
		}

	m_impl->rtInstShaderGuid.push_back(mat ? mat->shaderGuid : std::string());
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

	bool recreated = false;
	if (!d->tlas || d->tlasMaxInstances < count)   // (re)create when capacity grows
	{
		recreated = true;
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

	// Refit (update) instead of a full rebuild when the topology (instance count + BLAS set) is unchanged — only
	// transforms moved. A periodic full rebuild keeps the AS quality from degrading. Topology change -> rebuild.
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

	// Per-instance MatCB blocks (g_MatBytes) for auto-generated hit shaders — RAW buffer, grows as needed.
	if (!d->allMatCPU.empty())
	{
		if (!d->rtMatBuf || d->rtMatCapacity < (uint32_t)d->allMatCPU.size())
		{
			d->rtMatBuf.Release(); d->rtMatSRV = nullptr; d->rtMatCapacity = (uint32_t)d->allMatCPU.size();
			BufferDesc bd; bd.Name = "RT MatBytes"; bd.Usage = USAGE_DEFAULT; bd.BindFlags = BIND_SHADER_RESOURCE;
			bd.Mode = BUFFER_MODE_RAW; bd.Size = (Uint64)d->allMatCPU.size();
			d->device->CreateBuffer(bd, nullptr, &d->rtMatBuf);
			if (d->rtMatBuf) d->rtMatSRV = d->rtMatBuf->GetDefaultView(BUFFER_VIEW_SHADER_RESOURCE);
		}
		if (d->rtMatBuf)
			d->context->UpdateBuffer(d->rtMatBuf, 0, (Uint64)d->allMatCPU.size(), d->allMatCPU.data(), RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	}

	d->rtSceneReady = true;
}

// =================================================================================================
// RT reflection PIPELINE — real DXR: ray-gen + miss + closest-hit + SBT, native recursion.
// rt_rgen finds the primary reflector and spawns one reflection ray; rt_rchit reproduces the engine
// material model at the hit and recurses (mirror-in-mirror) via TraceRay; rt_rmiss = environment.
// Replaces the inline-RayQuery post pass.
// =================================================================================================
bool NukeDiligent::Impl::BuildRTPipeline()
{
	if (rtPSO && !rtPipelineDirty) return true;
	if (!rtSupported || !shaderFactory) return false;
	if (rtPSO) { rtPSO.Release(); rtSRB.Release(); rtSBT.Release(); }   // a new surf shader appeared -> rebuild
	rtPipelineDirty = false;

	auto mk = [&](const char* file, SHADER_TYPE type, const char* dbg, RefCntAutoPtr<IShader>& out)
	{
		ShaderCreateInfo sci; sci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
		sci.ShaderCompiler = SHADER_COMPILER_DXC; sci.HLSLVersion = ShaderVersion{6, 5};
		sci.pShaderSourceStreamFactory = shaderFactory;            // loads the file + resolves #include "rt_common.hlsl"
		sci.FilePath = file; sci.EntryPoint = "main"; sci.Desc = {dbg, type, true};
		device->CreateShader(sci, &out);
		if (!out) cout << "[NukeDiligent]\tRT shader build failed: " << file << endl;
		return (bool)out;
	};
	RefCntAutoPtr<IShader> rg, rm, rch;
	if (!mk("rt_rgen.hlsl",  SHADER_TYPE_RAY_GEN,         "RT RayGen",     rg))  return false;
	if (!mk("rt_rmiss.hlsl", SHADER_TYPE_RAY_MISS,        "RT Miss",       rm))  return false;
	if (!mk("rt_rchit.hlsl", SHADER_TYPE_RAY_CLOSEST_HIT, "RT ClosestHit", rch)) return false;

	// Auto-generate a closest-hit per material shader that ships a "<name>.surf.hlsl" (codegen from its schema).
	auto mkSrc = [&](const std::string& src, const char* dbg, RefCntAutoPtr<IShader>& out)
	{
		ShaderCreateInfo sci; sci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
		sci.ShaderCompiler = SHADER_COMPILER_DXC; sci.HLSLVersion = ShaderVersion{6, 5};
		sci.pShaderSourceStreamFactory = shaderFactory;   // resolves #include "rt_common.hlsl" / "<name>.surf.hlsl"
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
	hit.push_back({hitNames[0].c_str(), rch});                                   // default: standard PBR
	for (size_t i = 0; i < customChits.size(); ++i) hit.push_back({hitNames[i + 1].c_str(), customChits[i]});
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
	for (SHADER_TYPE t : {SHADER_TYPE_RAY_GEN, SHADER_TYPE_RAY_MISS, SHADER_TYPE_RAY_CLOSEST_HIT})
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
	rtOutW = w; rtOutH = h; rtOutTex.Release();
	TextureDesc td; td.Name = "RT Reflect Output"; td.Type = RESOURCE_DIM_TEX_2D;
	td.Width = (Uint32)w; td.Height = (Uint32)h; td.Format = HDR_FMT;
	td.BindFlags = BIND_UNORDERED_ACCESS | BIND_SHADER_RESOURCE;
	device->CreateTexture(td, nullptr, &rtOutTex);
}

void NukeDiligent::Impl::RunRTReflectPipeline(ITextureView* srcSRV, ITexture* dstTex, int w, int h, const std::vector<float>& params)
{
	if (!dstTex || !srcSRV) return;
	if (rtPipelineDirty) BuildRTPipeline();   // a custom shader appeared/changed -> rebuild with its hit group
	// No scene to trace (no opaque meshes) -> pass the chain colour through unchanged.
	if (!rtPSO || !rtSBT || !rtSceneReady || !tlas)
	{
		if (ITexture* srcTex = srcSRV->GetTexture())
			context->CopyTexture(CopyTextureAttribs{srcTex, RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
			                                        dstTex, RESOURCE_STATE_TRANSITION_MODE_TRANSITION});
		return;
	}
	EnsureRTOutput(w, h);
	if (!rtOutTex) return;

	{   // RTRefCB: clip->view + view->world + camera + (intensity, maxDist, maxDepth)
		struct CB { float4x4 ip, iv; float4 cam; float4 prm; };
		MapHelper<CB> cb(context, rtRefCB, MAP_WRITE, MAP_FLAG_DISCARD);
		cb->ip  = curProj.Inverse(); cb->iv = curView.Inverse();
		cb->cam = float4(curCamPos[0], curCamPos[1], curCamPos[2], 1.0f);
		float intensity = rtCfgIntensity;   // GLOBAL settings (Project Settings -> config), not the per-effect chip
		float maxDist   = rtCfgMaxDist;
		float maxDepth  = (float)rtCfgBounces;
		float roughCut  = rtCfgRoughCut;
		maxDepth = (maxDepth < 1.0f) ? 1.0f : (maxDepth > 7.0f ? 7.0f : maxDepth);   // PSO MaxRecursionDepth = 8
		if (roughCut < 0.05f) roughCut = 0.05f;
		cb->prm = float4(intensity, maxDist, maxDepth, roughCut);
	}

	// Bind dynamic resources for every RT stage that references them (null lookups are harmless).
	auto setv = [&](const char* n, IDeviceObject* o)
	{
		if (auto* v = rtSRB->GetVariableByName(SHADER_TYPE_RAY_GEN, n))         v->Set(o);
		if (auto* v = rtSRB->GetVariableByName(SHADER_TYPE_RAY_MISS, n))        v->Set(o);
		if (auto* v = rtSRB->GetVariableByName(SHADER_TYPE_RAY_CLOSEST_HIT, n)) v->Set(o);
	};
	setv("g_TLAS",     (IDeviceObject*)tlas.RawPtr());
	setv("g_Output",   rtOutTex->GetDefaultView(TEXTURE_VIEW_UNORDERED_ACCESS));
	setv("g_GBuffer",  gbufSRV);
	setv("g_Depth",    gbufDepthSRV);
	setv("g_Source",   srcSRV);
	setv("g_Probe",    (probeActive && probeCubeSRV) ? probeCubeSRV : fallbackCubeSRV);
	setv("g_AllNrm",   rtNrmSRV);
	setv("g_AllUV",    rtUVSRV ? rtUVSRV : rtNrmSRV);
	setv("g_Instances",rtInstSRV);
	setv("g_MatBytes", rtMatSRV ? rtMatSRV : rtInstSRV);   // per-instance MatCB blocks (auto-gen chits); rtInstSRV = valid non-null fallback
	{   // bindless albedo array (re-resolve each frame -> animated textures update)
		ITextureView* white = whiteTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
		IDeviceObject* arr[Impl::kMaxMatTex];
		for (uint32_t k = 0; k < kMaxMatTex; ++k)
		{
			if (k < matTexSRVs.size()) { if (ITextureView* s = GetTexSRV(matTexPtr[k])) matTexSRVs[k] = s; arr[k] = matTexSRVs[k] ? matTexSRVs[k] : white; }
			else arr[k] = white;
		}
		if (auto* v = rtSRB->GetVariableByName(SHADER_TYPE_RAY_CLOSEST_HIT, "g_MatTex")) v->SetArray(arr, 0, kMaxMatTex);
	}

	// The TLAS is rebuilt every frame -> refresh the SBT's per-instance hit-group mapping each frame. Each
	// instance routes to its shader's closest-hit (unlit -> its own RT shader; everything else -> standard PBR).
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

	context->CopyTexture(CopyTextureAttribs{rtOutTex, RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
	                                        dstTex,   RESOURCE_STATE_TRANSITION_MODE_TRANSITION});
}
