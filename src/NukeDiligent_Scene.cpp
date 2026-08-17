#include "NukeDiligentImpl.h"
#include "../include/NukeDiligentNative.h"
#include <array>   // module pass hooks (camera begin / post point)

namespace nukediligent { const WaterHooks& ActiveWaterHooks(); }

// Camera basis -> curView/curProj/curCamPos (left-handed look-at). Shared by beginCamera and the SSR prepass.
void NukeDiligent::Impl::SetCameraViewProj(const NukeCameraDesc& cam, int w, int h)
{
	const float aspect = (h > 0) ? (float)w / (float)h : 1.0f;
	float3 P(cam.camPos[0], cam.camPos[1], cam.camPos[2]);
	float3 F = normalize(float3(cam.camFwd[0], cam.camFwd[1], cam.camFwd[2]));
	float3 U = float3(cam.camUp[0], cam.camUp[1], cam.camUp[2]);
	float3 R = normalize(cross(U, F)); U = cross(F, R);
	curView = float4x4(R.x, U.x, F.x, 0.f, R.y, U.y, F.y, 0.f, R.z, U.z, F.z, 0.f, -dot(P, R), -dot(P, U), -dot(P, F), 1.f);
	static const bool diag = []{ const char* e = std::getenv("NUKE_TM_DIAG"); return e && *e == '1'; }();
	if (diag)
	{
		static int n = 0;
		if (n < 8) { ++n; std::cout << "[NukeDiligent]\tDIAG cam P(" << P.x << "," << P.y << "," << P.z
		                            << ") F(" << F.x << "," << F.y << "," << F.z << ") vp " << w << "x" << h
		                            << " ortho " << cam.ortho << " near " << cam.nearZ << " far " << cam.farZ << std::endl; }
	}
	// Projection: perspective, orthographic, or an element-wise blend of the two (cam.ortho tween).
	float4x4 persp = float4x4::Projection(cam.fov, aspect, cam.nearZ, cam.farZ, false);
	if (cam.ortho <= 0.0001f)
		curProj = persp;
	else
	{
		float halfH = (cam.orthoSize > 1e-4f) ? cam.orthoSize : 1.0f;
		float4x4 orth = float4x4::Ortho(2.0f * halfH * aspect, 2.0f * halfH, cam.nearZ, cam.farZ, false);
		if (cam.ortho >= 0.9999f) curProj = orth;
		else
			for (int r = 0; r < 4; ++r)
				for (int c = 0; c < 4; ++c)
					curProj.m[r][c] = persp.m[r][c] * (1.0f - cam.ortho) + orth.m[r][c] * cam.ortho;
	}
	curCamPos[0] = P.x; curCamPos[1] = P.y; curCamPos[2] = P.z;
	curCamFwd[0] = F.x; curCamFwd[1] = F.y; curCamFwd[2] = F.z;
	curCamEditor = cam.editorCamera != 0;   // module passes read this through the native hatch
}

// Target size for a camera (matches beginCamera): backbuffer (target 0) or the off-screen RT.
bool NukeDiligent::Impl::CameraSize(const NukeCameraDesc& cam, int& w, int& h)
{
	if (cam.target == 0)
	{
		if (!swapChain) return false;
		w = (int)swapChain->GetDesc().Width; h = (int)swapChain->GetDesc().Height;
	}
	else
	{
		auto it = rts.find(cam.target);
		if (it == rts.end()) return false;
		w = it->second.w; h = it->second.h;
	}
	return w > 0 && h > 0;
}

void NukeDiligent::renderObject(Mesh* mesh, Material* mat,
                                const float pos[3], const float quat[4], const float scale[3])
{
	if (!mesh) return;
	// One material for the whole mesh: the active LOD is one contiguous IB range = one draw.
	uint32_t first = 0, count = 0;
	m_impl->LodRange(mesh, m_impl->SelectLod(mesh, pos, scale), first, count);
	RenderObjectRange(mesh, mat, pos, quat, scale, first, count);
}

void NukeDiligent::renderObjectMulti(Mesh* mesh, Material* const* mats, int matCount,
                                     const float pos[3], const float quat[4], const float scale[3],
                                     int blendPass)
{
	if (!mesh) return;
	if (mesh->numIndices <= 0 || mesh->sections.empty())   // slot-less mesh: plain draw with slot 0
	{
		Material* m = matCount > 0 ? mats[0] : nullptr;
		const int bm = m ? m->blendMode : 0;
		if (blendPass == 0 && bm != 0) return;
		if (blendPass == 1 && bm == 0) return;
		renderObject(mesh, m, pos, quat, scale);
		return;
	}
	MeshLOD L = mesh->Lod(m_impl->SelectLod(mesh, pos, scale));
	for (int s = 0; s < L.sectionCount; ++s)
	{
		MeshSection sec = mesh->Section(L.firstSection + s);
		Material* m = (sec.slot >= 0 && sec.slot < matCount && mats[sec.slot]) ? mats[sec.slot]
		            : (matCount > 0 ? mats[0] : nullptr);
		const int bm = m ? m->blendMode : 0;
		if (blendPass == 0 && bm != 0) continue;
		if (blendPass == 1 && bm == 0) continue;
		RenderObjectRange(mesh, m, pos, quat, scale, sec.firstIndex, sec.indexCount);
	}
}

void NukeDiligent::RenderObjectRange(Mesh* mesh, Material* mat,
                                     const float pos[3], const float quat[4], const float scale[3],
                                     uint32_t firstIndex, uint32_t indexCount)
{
	if (m_impl->worldPipes.empty() || indexCount == 0) return;
	Impl::MeshGPU* gp = m_impl->GetMeshGPU(mesh);
	if (!gp) return;
	++m_impl->statDraws;                              // frame stats (status bar)
	m_impl->statTris += (int)indexCount / 3;
	Impl::MeshGPU& g = *gp;

	float4x4 world = float4x4::Scale(scale[0], scale[1], scale[2])
	               * Diligent::Quaternion<float>(quat[0], quat[1], quat[2], quat[3]).ToMatrix()
	               * float4x4::Translation(pos[0], pos[1], pos[2]);
	float4x4 wvp = world * m_impl->curView * m_impl->curProj;

	struct CBData { float4x4 wvp; float4x4 world; };
	{
		MapHelper<CBData> cb(m_impl->context, m_impl->worldCB, MAP_WRITE, MAP_FLAG_DISCARD);
		cb->wvp = wvp; cb->world = world;
	}
	m_impl->instWorldCBPass = 0;          // worldCB now holds THIS object's matrices, not the instanced VP
	m_impl->lastInstBind.pso = nullptr;   // and the pipeline/vertex-buffer state is about to change

	float col[4] = { 1, 1, 1, 1 };
	float metallic = 0.0f, roughness = 0.6f;
	float emissive[3] = { 0, 0, 0 }, emissiveI = 0.0f;
	float specF = 1.0f;
	ITextureView* srv = nullptr; ITextureView* nsrv = nullptr;
	ITextureView* mrsrv = nullptr; ITextureView* aosrv = nullptr; ITextureView* emsrv = nullptr; ITextureView* specsrv = nullptr;
	ITextureView* wipesrv = nullptr; ITextureView* heightsrv = nullptr;
	if (mat)
	{
		col[0] = (float)mat->color.r; col[1] = (float)mat->color.g; col[2] = (float)mat->color.b; col[3] = (float)mat->color.a;
		metallic = mat->metallic; roughness = mat->roughness; specF = mat->specular;
		emissive[0] = (float)mat->emissive.r; emissive[1] = (float)mat->emissive.g; emissive[2] = (float)mat->emissive.b;
		emissiveI = mat->emissiveIntensity;
		if (mat->diff) srv   = m_impl->GetTexSRV(mat->diff);
		if (mat->norm) nsrv  = m_impl->GetTexSRV(mat->norm);
		if (mat->mr)   mrsrv = m_impl->GetTexSRV(mat->mr);
		if (mat->ao)   aosrv = m_impl->GetTexSRV(mat->ao);
		if (mat->em)   emsrv = m_impl->GetTexSRV(mat->em);
		if (mat->spec) specsrv = m_impl->GetTexSRV(mat->spec);
		if (mat->wipe) wipesrv = m_impl->GetTexSRV(mat->wipe);
		if (mat->liveSurface.height) heightsrv = m_impl->GetTexSRV(mat->liveSurface.height);
	}
	// Overlay slot maps (LM-3 states/layers), OvTexNames() order; last = the painted 3D mask.
	ITextureView* ovsrv[Impl::kOvTexCount] = {};
	if (mat)
	{
		for (int s = 0; s < mat->liveOvCount && s < nuke::Material::kOverlaySlots; ++s)
		{
			const nuke::Material::OverlayRT& ov = mat->liveOv[s];
			if (ov.albedo) ovsrv[s * 4 + 0] = m_impl->GetTexSRV(ov.albedo);
			if (ov.normal) ovsrv[s * 4 + 1] = m_impl->GetTexSRV(ov.normal);
			if (ov.mrTex)  ovsrv[s * 4 + 2] = m_impl->GetTexSRV(ov.mrTex);
			if (ov.mask)   ovsrv[s * 4 + 3] = m_impl->GetTexSRV(ov.mask);
		}
		if (mat->liveDrawSet && mat->liveDrawMask3D) ovsrv[Impl::kOvSlots * 4] = m_impl->GetTexSRV(mat->liveDrawMask3D);
		if (mat->detail)    ovsrv[Impl::kOvSlots * 4 + 1] = m_impl->GetTexSRV(mat->detail);
		if (mat->detailNrm) ovsrv[Impl::kOvSlots * 4 + 2] = m_impl->GetTexSRV(mat->detailNrm);
	}
	uint64_t h = (mat && mat->shader && mat->shader->rendererHandle) ? mat->shader->rendererHandle
	                                                                  : m_impl->defaultWorldHandle;
	Impl::WorldPipe* pipe = m_impl->PipeFor(h);
	if (!pipe) return;
	Impl::WorldPipe& wp = *pipe;

	// Displacement tessellation: opaque/cutout material with a height map + Disp Scale, near
	// enough that the distance-faded factor exceeds 1 (far away the plain PSO takes over and
	// the DS displacement fades to zero, so the handover is seam-free).
	float tessF = 0.0f;
	if (wp.psoTess && wp.srbTess && mat && mat->liveSurface.height && mat->liveSurface.dispScale > 0.0f
	    && (mat->blendMode == 0 || mat->blendMode == 3) && !m_impl->wireframe)
	{
		const float4x4 inv = m_impl->curView.Inverse();
		const float dx = inv.m30 - pos[0], dy = inv.m31 - pos[1], dz = inv.m32 - pos[2];
		float dist = sqrtf(dx * dx + dy * dy + dz * dz);
		// Distance to the SURFACE, not the pivot: a big floor's centre sits far away while the
		// camera stands right on it — subtract the world-space bounding radius.
		if (mesh)
		{
			mesh->EnsureBounds();
			const float ex = std::max(fabsf(mesh->aabbMin[0]), fabsf(mesh->aabbMax[0])) * fabsf(scale[0]);
			const float ey = std::max(fabsf(mesh->aabbMin[1]), fabsf(mesh->aabbMax[1])) * fabsf(scale[1]);
			const float ez = std::max(fabsf(mesh->aabbMin[2]), fabsf(mesh->aabbMax[2])) * fabsf(scale[2]);
			dist = std::max(dist - sqrtf(ex * ex + ey * ey + ez * ez), 1.0f);
		}
		const float f = std::min(48.0f / std::max(dist, 1.0f), 12.0f);
		if (f > 1.05f) tessF = f;
	}
	m_impl->tessFillFactor = tessF;
	if (tessF > 0.0f) m_impl->matCBFor = nullptr;   // the factor is per-DRAW: force the refill
	// Overlay draw context (per-atom values + painted mask) is per-DRAW too.
	if (mat && mat->liveOvCount > 0 && mat->liveDrawSet) m_impl->matCBFor = nullptr;

	// Gated on the material changing; dynamic CBs recycle per frame, so the gate is pass-scoped.
	if (m_impl->matCBFor != mat || m_impl->matCBPass != m_impl->passSerial)
	{
		m_impl->matCBFor = mat; m_impl->matCBPass = m_impl->passSerial;
		if (m_impl->drawFlagsCB)
		{
			MapHelper<float> fc(m_impl->context, m_impl->drawFlagsCB, MAP_WRITE, MAP_FLAG_DISCARD);
			if (fc != nullptr) { fc[0] = (mat && !mat->receiveShadows) ? 0.0f : 1.0f; fc[1] = fc[2] = fc[3] = 0.0f; }
		}
		// MatCB layout mirrors the HLSL cbuffer: color @0, params @16/@32/@48, then the shader's
		// custom props at the engine-parsed offsets (Shader::props).
		MapHelper<Uint8> mb(m_impl->context, m_impl->worldMatCB, MAP_WRITE, MAP_FLAG_DISCARD);
		Uint8* p = mb;
		memset(p, 0, Impl::kMatCBBytes);
		memcpy(p + 0, col, sizeof(float) * 4);
		// g_Params.y: 0 = no normal; >0 = normal, OpenGL green (flip); <0 = normal, DirectX green (no flip).
		float nrmY = nsrv ? ((mat && mat->norm && !mat->norm->invertGreen) ? -1.0f : 1.0f) : 0.0f;
		float prm[4] = { srv ? 1.0f : 0.0f, nrmY, metallic, roughness };
		memcpy(p + 16, prm, sizeof(float) * 4);   // g_Params (hasBase, hasNormal±greenConv, metallic, roughness)
		float prm2[4] = { mrsrv ? 1.0f : 0.0f, aosrv ? 1.0f : 0.0f, emsrv ? 1.0f : 0.0f, specF };
		memcpy(p + 32, prm2, sizeof(float) * 4);  // g_Params2 (hasMR, hasAO, hasEm, specularFactor)
		float emv[4] = { emissive[0], emissive[1], emissive[2], emissiveI };
		memcpy(p + 48, emv, sizeof(float) * 4);   // g_Emissive2 (rgb, intensity)
		if (mat && mat->shader)
			for (const nuke::ShaderProp& sp : mat->shader->props)
			{
				auto pv = mat->props.find(sp.name);   // unset -> the shader's HLSL default
				const float* v = (pv != mat->props.end()) ? pv->second.data() : sp.def;
				uint32_t bytes = (uint32_t)sp.components * sizeof(float);
				if (sp.offset + bytes <= Impl::kMatCBBytes) memcpy(p + sp.offset, v, bytes);
			}
		// Tessellation factor is per-DRAW (camera distance): patch it over g_Disp.w at the
		// shader's own parsed offset (custom shaders may lay their props out differently).
		if (m_impl->tessFillFactor > 0.0f && mat && mat->shader)
			for (const nuke::ShaderProp& sp : mat->shader->props)
				if (sp.name == "g_Disp" && sp.components == 4 && sp.offset + 16 <= Impl::kMatCBBytes)
				{ memcpy(p + sp.offset + 12, &m_impl->tessFillFactor, sizeof(float)); break; }
		// Overlay draw context: per-atom state values over g_Ov*.x, painted-mask channels over
		// g_OvP*.z and the world->mask transform — patched at the parsed offsets, same rule.
		if (mat && mat->liveOvCount > 0 && mat->liveDrawSet && mat->shader)
		{
			static const auto kOvV = []{ std::array<std::string, Impl::kOvSlots> a; for (int s = 0; s < Impl::kOvSlots; ++s) a[s] = "g_Ov"  + std::to_string(s); return a; }();
			static const auto kOvP = []{ std::array<std::string, Impl::kOvSlots> a; for (int s = 0; s < Impl::kOvSlots; ++s) a[s] = "g_OvP" + std::to_string(s); return a; }();
			static const char* const kOvM[3] = { "g_OvM0", "g_OvM1", "g_OvM2" };
			for (const nuke::ShaderProp& sp : mat->shader->props)
			{
				if (sp.components != 4 || sp.offset + 16 > Impl::kMatCBBytes) continue;
				float* d = (float*)(p + sp.offset);
				for (int s = 0; s < Impl::kOvSlots; ++s)
				{
					if (sp.name == kOvV[s] && mat->liveDrawValue[s] >= 0.0f) d[0] = mat->liveDrawValue[s];
					if (sp.name == kOvP[s]) d[2] = mat->liveDrawMaskChan[s];
				}
				if (mat->liveDrawMask3D)
				{
					for (int r = 0; r < 3; ++r)
						if (sp.name == kOvM[r]) memcpy(d, &mat->liveDrawMaskXform[r * 4], 16);
					if (sp.name == "g_OvMQ") { d[0] = mat->liveDrawMaskRes; d[1] = 1.0f; }
				}
			}
		}
	}

	// Gate on the pointer changing: Diligent rewrites the descriptor cache on every DYNAMIC-var Set().
	auto bindIf = [](IShaderResourceVariable* v, IDeviceObject* o, IDeviceObject*& cached)
	{ if (v && o && o != cached) { v->Set(o); cached = o; } };
	ITextureView* whiteSRV = m_impl->whiteTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
	bindIf(wp.texVar,  srv  ? srv  : whiteSRV, wp.lastBind[0]);
	bindIf(wp.normVar, nsrv ? nsrv : m_impl->flatNormTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE), wp.lastBind[1]);
	bindIf(wp.mrVar,   mrsrv   ? mrsrv   : whiteSRV, wp.lastBind[2]);
	bindIf(wp.aoVar,   aosrv   ? aosrv   : whiteSRV, wp.lastBind[3]);
	bindIf(wp.emVar,   emsrv   ? emsrv   : whiteSRV, wp.lastBind[4]);
	bindIf(wp.specVar, specsrv ? specsrv : whiteSRV, wp.lastBind[5]);
	bindIf(wp.shadowVar, m_impl->shadowSRV ? (IDeviceObject*)m_impl->shadowSRV : (IDeviceObject*)whiteSRV, wp.lastBind[6]);
	bindIf(wp.cubeVar,   m_impl->shadowCubeSRV, wp.lastBind[7]);
	bindIf(wp.probeVar,  (m_impl->probeActive && m_impl->probeCubeSRV) ? m_impl->probeCubeSRV : m_impl->fallbackCubeSRV, wp.lastBind[8]);
	bindIf(wp.tlasVar,   (m_impl->rtSceneReady && m_impl->tlas) ? (IDeviceObject*)m_impl->tlas.RawPtr() : (IDeviceObject*)m_impl->fallbackTLAS.RawPtr(), wp.lastBind[9]);
	bindIf(wp.rtInstVar, (IDeviceObject*)(m_impl->rtInstSRV ? m_impl->rtInstSRV : m_impl->rtNrmSRV), wp.lastBind[10]);
	bindIf(wp.wipeVar, wipesrv ? wipesrv : whiteSRV, wp.lastBind[11]);
	bindIf(wp.heightVar, heightsrv ? heightsrv : whiteSRV, wp.lastBind[12]);
	{
		ITextureView* flatN = m_impl->flatNormTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
		for (int k = 0; k < Impl::kOvTexCount; ++k)
			bindIf(wp.ovVar[k], ovsrv[k] ? ovsrv[k] : (((k < Impl::kOvSlots * 4 && (k & 3) == 1) || k == Impl::kOvSlots * 4 + 2) ? flatN : whiteSRV), wp.lastBind[13 + k]);
		bindIf(wp.flowVar, (mat && mat->flow) ? m_impl->GetTexSRV(mat->flow) : whiteSRV, wp.lastBind[13 + Impl::kOvTexCount]);
		bindIf(wp.refrVar, m_impl->refrSRV ? m_impl->refrSRV : whiteSRV, wp.lastBind[13 + Impl::kOvTexCount + 1]);
	bindIf(wp.mskVar, (mat && mat->mskStamp) ? m_impl->GetTexSRV(mat->mskStamp) : whiteSRV, wp.lastBind[13 + Impl::kOvTexCount + 2]);
	}

	IDeviceContext* ctx = m_impl->context;
	// Slot 3 = optional vertex colors; extra bound buffers are ignored by 3-element layouts.
	IBuffer* vbs[]    = { g.pos, g.nrm, g.uv, g.col };
	Uint64   offs[]   = { 0, 0, 0, 0 };
	ctx->SetVertexBuffers(0, g.col ? 4 : 3, vbs, offs, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, SET_VERTEX_BUFFERS_FLAG_RESET);
	// Blend variant for this material; wireframe draw mode overrides them all.
	IPipelineState* pso = wp.pso;
	IShaderResourceBinding* srb = wp.srb;
	// Vertex-color variant: mesh has a color stream + the material asks for tint/overlay-mask.
	// Displacement tessellation takes precedence (same materials still render, un-tinted).
	const bool vcolDraw = mat && mat->vcolorMode > 0 && g.col && wp.psoVcol && tessF <= 0.0f && !m_impl->wireframe;
	if (m_impl->wireframe && wp.psoWire) pso = wp.psoWire;
	else if (vcolDraw)
	{
		if      (mat->blendMode == 1 && wp.psoVcolBlend) pso = wp.psoVcolBlend;
		else if (mat->blendMode == 2 && wp.psoVcolAdd)   pso = wp.psoVcolAdd;
		else                                             pso = wp.psoVcol;
	}
	else if (mat && mat->blendMode == 1 && wp.psoBlend) pso = wp.psoBlend;
	else if (mat && mat->blendMode == 2 && wp.psoAdd)   pso = wp.psoAdd;
	else if (tessF > 0.0f)
	{
		// Tess draws are rare: bind the whole set by NAME on the tess SRB, no redundancy gates.
		pso = wp.psoTess; srb = wp.srbTess;
		ITextureView* flatN = m_impl->flatNormTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
		auto TP = [&](SHADER_TYPE st, const char* nm, IDeviceObject* o)
		{ if (o) if (auto* v = wp.srbTess->GetVariableByName(st, nm)) v->Set(o); };
		TP(SHADER_TYPE_PIXEL, "g_Tex",        srv     ? srv     : whiteSRV);
		TP(SHADER_TYPE_PIXEL, "g_Normal",     nsrv    ? nsrv    : flatN);
		TP(SHADER_TYPE_PIXEL, "g_MetalRough", mrsrv   ? mrsrv   : whiteSRV);
		TP(SHADER_TYPE_PIXEL, "g_Occlusion",  aosrv   ? aosrv   : whiteSRV);
		TP(SHADER_TYPE_PIXEL, "g_Emissive",   emsrv   ? emsrv   : whiteSRV);
		TP(SHADER_TYPE_PIXEL, "g_Spec",       specsrv ? specsrv : whiteSRV);
		TP(SHADER_TYPE_PIXEL, "g_WipeMask",   wipesrv ? wipesrv : whiteSRV);
		TP(SHADER_TYPE_PIXEL, "g_Height",     heightsrv ? heightsrv : whiteSRV);
		TP(SHADER_TYPE_DOMAIN, "g_Height",    heightsrv ? heightsrv : whiteSRV);
		TP(SHADER_TYPE_PIXEL, "g_Shadow",     m_impl->shadowSRV ? (IDeviceObject*)m_impl->shadowSRV : (IDeviceObject*)whiteSRV);
		TP(SHADER_TYPE_PIXEL, "g_ShadowCube", m_impl->shadowCubeSRV);
		TP(SHADER_TYPE_PIXEL, "g_Probe",      (m_impl->probeActive && m_impl->probeCubeSRV) ? m_impl->probeCubeSRV : m_impl->fallbackCubeSRV);
		TP(SHADER_TYPE_PIXEL, "g_TLAS",       (m_impl->rtSceneReady && m_impl->tlas) ? (IDeviceObject*)m_impl->tlas.RawPtr() : (IDeviceObject*)m_impl->fallbackTLAS.RawPtr());
		TP(SHADER_TYPE_PIXEL, "g_RTInst",     (IDeviceObject*)(m_impl->rtInstSRV ? m_impl->rtInstSRV : m_impl->rtNrmSRV));
		for (int k = 0; k < Impl::kOvTexCount; ++k)
			TP(SHADER_TYPE_PIXEL, Impl::OvTexNames()[k].c_str(), ovsrv[k] ? ovsrv[k] : (((k < Impl::kOvSlots * 4 && (k & 3) == 1) || k == Impl::kOvSlots * 4 + 2) ? (IDeviceObject*)flatN : (IDeviceObject*)whiteSRV));
		TP(SHADER_TYPE_PIXEL, "g_Flow",      (mat && mat->flow) ? (IDeviceObject*)m_impl->GetTexSRV(mat->flow) : (IDeviceObject*)whiteSRV);
		TP(SHADER_TYPE_PIXEL, "g_MskStamp",  (mat && mat->mskStamp) ? (IDeviceObject*)m_impl->GetTexSRV(mat->mskStamp) : (IDeviceObject*)whiteSRV);
		TP(SHADER_TYPE_PIXEL, "g_SceneRefr", m_impl->refrSRV ? (IDeviceObject*)m_impl->refrSRV : (IDeviceObject*)whiteSRV);
	}
	ctx->SetPipelineState(pso);
	ctx->CommitShaderResources(srb, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	if (g.idx)
	{
		ctx->SetIndexBuffer(g.idx, 0, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		DrawIndexedAttribs da{(Uint32)indexCount, VT_UINT32, DRAW_FLAG_VERIFY_STATES};
		da.FirstIndexLocation = (Uint32)firstIndex;
		ctx->DrawIndexed(da);
	}
	else
	{
		DrawAttribs da{(Uint32)indexCount, DRAW_FLAG_VERIFY_STATES};
		da.StartVertexLocation = (Uint32)firstIndex;
		ctx->Draw(da);
	}
	// The overlay draw context was for THIS draw only: consume it so a later draw of the same
	// material from another pass (preview, module) falls back to the CB's global values.
	if (mat && mat->liveDrawSet) mat->liveDrawSet = false;
}

// LM-6 background refraction: snapshot the opaque scene (resolve when MSAA) so transparent
// refractive draws can sample what is behind them. The copy invalidates on size/format change;
// beginCamera nulls refrSRV so a camera without refraction never samples a stale snapshot.
void NukeDiligent::beginTransparent()
{
	Impl& im = *m_impl;
	if (!im.curRTV) return;
	ITexture* src = im.curRTV->GetTexture();
	const TextureDesc& sd = src->GetDesc();
	if (im.refrTex && (im.refrTex->GetDesc().Width != sd.Width || im.refrTex->GetDesc().Height != sd.Height
	                   || im.refrTex->GetDesc().Format != sd.Format))
	{
		im.Trash(im.refrTex);
		im.refrTex.Release();
	}
	if (!im.refrTex)
	{
		TextureDesc td;
		td.Name = "Scene refraction copy";
		td.Type = RESOURCE_DIM_TEX_2D;
		td.Width = sd.Width; td.Height = sd.Height;
		td.Format = sd.Format; td.MipLevels = 1; td.SampleCount = 1;
		td.BindFlags = BIND_SHADER_RESOURCE; td.Usage = USAGE_DEFAULT;
		im.device->CreateTexture(td, nullptr, &im.refrTex);
	}
	if (!im.refrTex) return;
	// Copying/resolving from a bound target is invalid - unbind, snapshot, rebind.
	im.context->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE);
	if (sd.SampleCount > 1)
	{
		ResolveTextureSubresourceAttribs ra;
		im.context->ResolveTextureSubresource(src, im.refrTex, ra);
	}
	else
	{
		CopyTextureAttribs cp(src, RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
		                      im.refrTex, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		im.context->CopyTexture(cp);
	}
	im.context->SetRenderTargets(1, &im.curRTV, im.curDSV, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	im.refrSRV = im.refrTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
}

void NukeDiligent::selectionOutlineBegin()
{
	if (!m_impl->outlineMaskPSO || !m_impl->outlineEdgePSO || !m_impl->curRTV) return;
	if (!m_impl->outlineStamp.current(m_impl->samples, m_impl->SceneFmt())) return;   // rebuilding
	m_impl->EnsureOutlineMask(m_impl->curRTW, m_impl->curRTH);
	if (!m_impl->outlineMaskRTV || !m_impl->outlineMaskSRV) return;
	const float zero[4] = { 0, 0, 0, 0 };
	m_impl->context->SetRenderTargets(1, &m_impl->outlineMaskRTV, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	m_impl->context->ClearRenderTarget(m_impl->outlineMaskRTV, zero, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	m_impl->outlineOpen = true;
}

// One mesh into the open mask (alpha = 1 over the object). LOD0 only: the outline follows the
// silhouette the camera sees, and the IB carries the coarser shells after it.
void NukeDiligent::selectionOutlineAdd(Mesh* mesh, const float pos[3], const float quat[4], const float scale[3])
{
	if (!m_impl->outlineOpen || !mesh) return;
	Impl::MeshGPU* gp = m_impl->GetMeshGPU(mesh);
	if (!gp) return;
	Impl::MeshGPU& g = *gp;
	IDeviceContext* ctx = m_impl->context;

	float4x4 world = float4x4::Scale(scale[0], scale[1], scale[2])
	               * Diligent::Quaternion<float>(quat[0], quat[1], quat[2], quat[3]).ToMatrix()
	               * float4x4::Translation(pos[0], pos[1], pos[2]);
	float4x4 wvp = world * m_impl->curView * m_impl->curProj;
	struct CBData { float4x4 wvp; float4x4 world; };
	{ MapHelper<CBData> cb(ctx, m_impl->worldCB, MAP_WRITE, MAP_FLAG_DISCARD); cb->wvp = wvp; cb->world = world; }

	IBuffer* vbs[]  = { g.pos, g.nrm, g.uv };
	Uint64   offs[] = { 0, 0, 0 };
	ctx->SetVertexBuffers(0, 3, vbs, offs, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, SET_VERTEX_BUFFERS_FLAG_RESET);
	ctx->SetPipelineState(m_impl->outlineMaskPSO);
	ctx->CommitShaderResources(m_impl->outlineMaskSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	if (g.idx)
	{
		uint32_t l0First = 0, l0Count = 0;
		m_impl->LodRange(mesh, 0, l0First, l0Count);
		ctx->SetIndexBuffer(g.idx, 0, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		DrawIndexedAttribs da{(Uint32)l0Count, VT_UINT32, DRAW_FLAG_VERIFY_STATES};
		da.FirstIndexLocation = (Uint32)l0First;
		ctx->DrawIndexed(da);
	}
	else
	{
		DrawAttribs da{(Uint32)g.numVerts, DRAW_FLAG_VERIFY_STATES};
		ctx->Draw(da);
	}
}

void NukeDiligent::selectionOutlineEnd()
{
	if (!m_impl->outlineOpen) return;
	m_impl->outlineOpen = false;
	IDeviceContext* ctx = m_impl->context;

	// Fullscreen edge-detect over the accumulated mask -> border into the camera RT.
	struct EdgeData { float texel[4]; };
	{
		MapHelper<EdgeData> cb(ctx, m_impl->outlineEdgeCB, MAP_WRITE, MAP_FLAG_DISCARD);
		cb->texel[0] = (m_impl->curRTW > 0) ? 1.0f / m_impl->curRTW : 0.0f;
		cb->texel[1] = (m_impl->curRTH > 0) ? 1.0f / m_impl->curRTH : 0.0f;
		cb->texel[2] = 2.0f;   // outline thickness in pixels (constant on screen)
		cb->texel[3] = 0.0f;
	}
	ctx->SetRenderTargets(1, &m_impl->curRTV, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	ctx->SetVertexBuffers(0, 0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, SET_VERTEX_BUFFERS_FLAG_RESET);
	if (m_impl->outlineEdgeMaskVar) m_impl->outlineEdgeMaskVar->Set(m_impl->outlineMaskSRV);
	ctx->SetPipelineState(m_impl->outlineEdgePSO);
	ctx->CommitShaderResources(m_impl->outlineEdgeSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	DrawAttribs fs{3, DRAW_FLAG_VERIFY_STATES};
	ctx->Draw(fs);

	// Must restore the camera targets WITH depth: endCamera's flushes use D32 PSOs, and a
	// depth-less binding (DSV = UNKNOWN) makes every later draw mismatch and skip depth testing.
	ctx->SetRenderTargets(1, &m_impl->curRTV, m_impl->curDSV, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
}

// Single-mesh convenience: the batched path with one entry.
void NukeDiligent::renderSelectionOutline(Mesh* mesh, const float pos[3], const float quat[4], const float scale[3])
{
	selectionOutlineBegin();
	selectionOutlineAdd(mesh, pos, quat, scale);
	selectionOutlineEnd();
}

// Fill the world FrameCB (camera pos, ambient, lights, shadow params, sky IBL, reflection probe).
// Shared by the camera pass and the probe cube-face passes.
void NukeDiligent::Impl::WriteFrameCB(const float3& P)
{
	MapHelper<FrameCBData> fb(context, worldFrameCB, MAP_WRITE, MAP_FLAG_DISCARD);
	if (fb == nullptr) return;   // dead device (removal mid-frame): Map returns null
	memset(fb, 0, sizeof(FrameCBData));
	fb->camPos[0] = P.x; fb->camPos[1] = P.y; fb->camPos[2] = P.z;
	fb->ambient[0] = sky.ambient[0]; fb->ambient[1] = sky.ambient[1];
	fb->ambient[2] = sky.ambient[2]; fb->ambient[3] = sky.ambientIntensity;

	std::vector<NukeLight> src = lights;
	if (src.empty())
	{
		NukeLight sun; sun.type = 0; sun.dir[0] = -0.4f; sun.dir[1] = -0.85f; sun.dir[2] = -0.35f;
		sun.color[0] = sun.color[1] = sun.color[2] = 1.0f; sun.intensity = 3.0f; src.push_back(sun);
	}
	int n = (int)src.size();
	if (n > kMaxLights)
	{
		static bool warned = false;
		if (!warned) { warned = true; cout << "[NukeDiligent]\tlight budget exceeded (" << n << " > " << kMaxLights << ") — extras dropped" << endl; }
		n = kMaxLights;
	}
	fb->lightCount[0] = (float)n;
	for (int k = 0; k < n; ++k)
	{
		const NukeLight& L = src[k]; GPULight& g = fb->lights[k];
		g.posType[0] = L.pos[0]; g.posType[1] = L.pos[1]; g.posType[2] = L.pos[2]; g.posType[3] = (float)L.type;
		g.dirRange[0] = L.dir[0]; g.dirRange[1] = L.dir[1]; g.dirRange[2] = L.dir[2]; g.dirRange[3] = L.range;
		g.colorIntensity[0] = L.color[0]; g.colorIntensity[1] = L.color[1]; g.colorIntensity[2] = L.color[2]; g.colorIntensity[3] = L.intensity;
		g.spot[0] = L.spotInner; g.spot[1] = L.spotOuter; g.spot[2] = (float)lightSlot[k]; g.spot[3] = (float)lightCube[k];
	}
	for (int s = 0; s < SHADOW_SLOTS; ++s) memcpy(fb->shadowVP + s * 16, &slotVP[s], sizeof(float) * 16);
	fb->shadowParams[0] = (float)numShadowSlots;
	// Shadow bias in NDC scales with Shadow Distance, so clamp to world units: depth bias capped
	// at ~3 cm equivalent, normal offset floored at half a shadow texel clamped to [0.5, 3] cm.
	{
		const float depthRange = 2.0f * (shadowDistance > 0.5f ? shadowDistance : 0.5f);
		const float ndcCap     = 0.03f / depthRange;
		fb->shadowParams[3] = shadowDepthBias < ndcCap ? shadowDepthBias : ndcCap;
		float nof = 0.5f * shadowDistance / (float)shadowRes;
		nof = nof < 0.005f ? 0.005f : (nof > 0.03f ? 0.03f : nof);
		fb->shadowParams[1] = (shadowNormalBias > nof) ? shadowNormalBias : nof;
	}
	fb->shadowParams[2] = (1.0f / (float)shadowRes) * shadowSoftness;
	for (int k = 0; k < 3; ++k) { fb->skyTop[k] = sky.top[k]; fb->skyHorizon[k] = sky.horizon[k]; fb->skyGround[k] = sky.ground[k]; }
	fb->skyParams[0] = sky.skyIntensity; fb->skyParams[1] = (sky.mode == 1) ? 1.0f : 0.0f;
	fb->skyParams[2] = hdr ? 0.0f : 1.0f; fb->skyParams[3] = sky.whitePoint;   // .w = SDR tonemap white point (world.ps HDR-off path)
	const bool probe = probeActive && probeCubeSRV;   // off during the probe's own capture -> no feedback
	fb->probePos[0] = probePos[0]; fb->probePos[1] = probePos[1]; fb->probePos[2] = probePos[2]; fb->probePos[3] = probe ? 1.0f : 0.0f;
	fb->probeParams[0] = probeIntensity; fb->probeParams[1] = probeMaxMip; fb->probeParams[2] = 0; fb->probeParams[3] = 0;
	const bool box = probe && (probeBoxHalf[0] > 0.f || probeBoxHalf[1] > 0.f || probeBoxHalf[2] > 0.f);
	fb->probeBox[0] = probeBoxHalf[0]; fb->probeBox[1] = probeBoxHalf[1]; fb->probeBox[2] = probeBoxHalf[2]; fb->probeBox[3] = box ? 1.0f : 0.0f;
	memcpy(fb->wind,  windDirStrength, sizeof(fb->wind));    // 7.2: g_Wind (dir.xyz, gusted strength)
	memcpy(fb->wind2, windParams,      sizeof(fb->wind2));   //      g_Wind2 (turbAmount, 1/turbScale, time, gustFreq)
}

void NukeDiligent::beginCamera(const NukeCameraDesc& cam)
{
	m_impl->refrSRV = nullptr;   // a refraction snapshot never outlives its camera pass
	m_impl->GpuPass("scene");   // geometry of this camera, up to the post chain in endCamera
	++m_impl->passSerial;   // invalidate the per-draw redundancy gates (shared CBs re-map per pass)
	m_impl->curTarget = cam.target;   // feedback guard: GetTexSRV won't sample the RT we draw into
	// LOD anchor: mesh LOD selection measures distance from the camera drawing the frame
	// (shadow/probe passes reuse the latest camera, not the light).
	m_impl->lodCamPos[0] = cam.camPos[0]; m_impl->lodCamPos[1] = cam.camPos[1]; m_impl->lodCamPos[2] = cam.camPos[2];
	m_impl->curMSAA = false; m_impl->curResolveSrc = nullptr; m_impl->curResolveDst = nullptr;
	m_impl->curPostSrc = nullptr; m_impl->curPostDst = nullptr;
	const bool ms = m_impl->samples > 1;
	ITextureView* rtv = nullptr;
	ITextureView* dsv = nullptr;
	int w = 0, h = 0;
	if (cam.target == 0)
	{
		// Backbuffer path renders to an HDR intermediate; endCamera's post pass tonemaps into the swap chain.
		w = (int)m_impl->swapChain->GetDesc().Width;
		h = (int)m_impl->swapChain->GetDesc().Height;
		m_impl->EnsureBackbufferMS(w, h);
		Impl::RT& bb = m_impl->backbufferMS;
		rtv = bb.rtv; dsv = bb.dsv;
		if (ms) { m_impl->curMSAA = true; m_impl->curResolveSrc = bb.colorMS; m_impl->curResolveDst = bb.color; }
		m_impl->curPostSrc = bb.hdrSRV;
		m_impl->curPostDst = m_impl->swapChain->GetCurrentBackBufferRTV();
	}
	else
	{
		auto it = m_impl->rts.find(cam.target);
		if (it == m_impl->rts.end()) return;
		Impl::RT& rt = it->second;
		rtv = rt.rtv; dsv = rt.dsv; w = rt.w; h = rt.h;
		if (ms && rt.colorMS) { m_impl->curMSAA = true; m_impl->curResolveSrc = rt.colorMS; m_impl->curResolveDst = rt.color; }
		m_impl->curPostSrc = rt.hdrSRV;
		m_impl->curPostDst = rt.postRTV;
	}
	if (!rtv) return;
	m_impl->curRTV = rtv; m_impl->curDSV = dsv;                     // for the selection-outline pass (restore)
	m_impl->curRTW = w; m_impl->curRTH = h;
	m_impl->cameraPassActive = true;   // sprites may draw from here until endCamera completes
	{   // module camera-begin hook (water resets its per-camera underwater candidate here);
		// its GPU work (FFT/ripple/SWE/FLIP sims) times as "water.sim", then "scene" resumes.
		const nukediligent::WaterHooks& wh = nukediligent::ActiveWaterHooks();
		if (wh.onCameraBegin)
		{
			m_impl->GpuPass("water.sim");
			wh.onCameraBegin(wh.user);
			m_impl->GpuPass("scene");
		}
	}

	IDeviceContext* ctx = m_impl->context;
	ctx->SetRenderTargets(1, &rtv, dsv, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	// Clear alpha is carried premultiplied through the final pass — it drives per-pixel
	// transparency on a composited window (ignored on an opaque one).
	ctx->ClearRenderTarget(rtv, cam.clear, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	if (dsv)
		ctx->ClearDepthStencil(dsv, CLEAR_DEPTH_FLAG, 1.f, 0, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	Viewport vp; vp.TopLeftX = 0; vp.TopLeftY = 0; vp.Width = (float)w; vp.Height = (float)h; vp.MinDepth = 0; vp.MaxDepth = 1;
	ctx->SetViewports(1, &vp, w, h);

	m_impl->SetCameraViewProj(cam, w, h);   // curView/curProj/curCamPos (shared with the SSR gbuffer prepass)
	m_impl->curNear = cam.nearZ; m_impl->curFar = cam.farZ;   // soft-particle depth linearization
	m_impl->curProjNoJitter = m_impl->curProj;   // unjittered — TAA reprojection + the depth prepass use this
	if (m_impl->curTAA && w > 0 && h > 0)        // TAA: jitter the COLOUR projection sub-pixel (Halton); depth stays clean
	{
		m_impl->curProj.m[2][0] += m_impl->curJitterX * 2.0f / (float)w;   // pixel -> NDC (row-vector: clip.x += vz*offset)
		m_impl->curProj.m[2][1] += m_impl->curJitterY * 2.0f / (float)h;
	}

	float3 P(cam.camPos[0], cam.camPos[1], cam.camPos[2]);
	m_impl->WriteFrameCB(P);

	m_impl->DrawSky();   // procedural sky behind the scene (after clear, before geometry)
}

void NukeDiligent::setSky(const NukeSky& s) { m_impl->sky = s; m_impl->toneExposure = s.exposure; m_impl->toneWhite = s.whitePoint; }

// Halton low-discrepancy sequence (1-based index) — even sub-pixel coverage for the TAA jitter.
static float Halton(int i, int b) { float f = 1.0f, r = 0.0f; while (i > 0) { f /= b; r += f * (i % b); i /= b; } return r; }

// Enable/disable TAA for the camera about to render; when enabled, advances the sub-pixel
// jitter (Halton 2,3; ±0.5 px) that the next beginCamera applies to the colour projection.
void NukeDiligent::setCameraTAA(bool enabled)
{
	m_impl->curTAA = enabled;
	if (!enabled) return;
	int idx = (m_impl->taaFrame % 8) + 1;   // period-8 Halton
	++m_impl->taaFrame;
	m_impl->curJitterX = Halton(idx, 2) - 0.5f;
	m_impl->curJitterY = Halton(idx, 3) - 0.5f;
}

// ---- debug/gizmo lines (iRender::drawDebugLine) ----------------------------------------

void NukeDiligent::drawDebugLine(const float a[3], const float b[3], const float color[4])
{
	std::lock_guard<std::mutex> lock(m_impl->debugMutex);
	auto& v = m_impl->debugVerts;
	v.insert(v.end(), { a[0], a[1], a[2], color[0], color[1], color[2], color[3],
	                    b[0], b[1], b[2], color[0], color[1], color[2], color[3] });
}

void NukeDiligent::drawDebugLineDepth(const float a[3], const float b[3], const float color[4])
{
	std::lock_guard<std::mutex> lock(m_impl->debugMutex);
	auto& v = m_impl->debugVertsDepth;
	v.insert(v.end(), { a[0], a[1], a[2], color[0], color[1], color[2], color[3],
	                    b[0], b[1], b[2], color[0], color[1], color[2], color[3] });
}

void NukeDiligent::Impl::CreateDebugResources()
{
	debugPSO.Release(); debugPSOBB.Release(); debugSRB.Release(); debugSRBBB.Release(); debugCB.Release();
	std::string vs = shaderSource("debug.vs"), ps = shaderSource("debug.ps");
	if (vs.empty() || ps.empty()) { cout << "[NukeDiligent]	debug-line shaders missing" << endl; return; }
	ShaderCreateInfo sci; sci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
	RefCntAutoPtr<IShader> v, p;
	sci.Desc = {"Debug VS", SHADER_TYPE_VERTEX, true}; sci.Source = vs.c_str(); CreateShaderCached(sci, &v);
	sci.Desc = {"Debug PS", SHADER_TYPE_PIXEL, true};  sci.Source = ps.c_str(); CreateShaderCached(sci, &p);
	if (!v || !p) return;

	BufferDesc cbd; cbd.Name = "DebugCB"; cbd.Size = sizeof(float4x4);
	cbd.Usage = USAGE_DYNAMIC; cbd.BindFlags = BIND_UNIFORM_BUFFER; cbd.CPUAccessFlags = CPU_ACCESS_WRITE;
	device->CreateBuffer(cbd, nullptr, &debugCB);

	// Post-last overlay: LDR target, single-sample, no depth (scene depth is already resolved).
	auto build = [&](TEXTURE_FORMAT fmt, const char* name,
	                 RefCntAutoPtr<IPipelineState>& pso, RefCntAutoPtr<IShaderResourceBinding>& srb)
	{
		GraphicsPipelineStateCreateInfo ci; ci.PSODesc.Name = name;
		auto& gp = ci.GraphicsPipeline;
		gp.NumRenderTargets = 1; gp.RTVFormats[0] = fmt;
		gp.DSVFormat = TEX_FORMAT_UNKNOWN;
		gp.PrimitiveTopology = PRIMITIVE_TOPOLOGY_LINE_LIST;
		gp.RasterizerDesc.CullMode = CULL_MODE_NONE;
		gp.DepthStencilDesc.DepthEnable = False;
		gp.DepthStencilDesc.DepthWriteEnable = False;
		gp.SmplDesc.Count = 1;
		LayoutElement layout[] = {
			{0, 0, 3, VT_FLOAT32, False},   // pos
			{1, 0, 4, VT_FLOAT32, False},   // color
		};
		gp.InputLayout.LayoutElements = layout; gp.InputLayout.NumElements = 2;
		ci.pVS = v; ci.pPS = p;
		CreateGraphicsPipelineStateCached(ci, &pso);
		if (pso)
		{
			if (auto* sv = pso->GetStaticVariableByName(SHADER_TYPE_VERTEX, "DebugCB")) sv->Set(debugCB);
			pso->CreateShaderResourceBinding(&srb, true);
		}
	};
	build(TEX_FORMAT_RGBA8_UNORM, "Debug Lines PSO", debugPSO, debugSRB);
	TEXTURE_FORMAT bbFmt = swapChain ? swapChain->GetDesc().ColorBufferFormat : TEX_FORMAT_RGBA8_UNORM;
	build(bbFmt, "Debug Lines PSO BB", debugPSOBB, debugSRBBB);
	cout << "[NukeDiligent]	debug-line pipeline" << (debugPSO ? " ready" : " FAILED") << endl;
}

void NukeDiligent::Impl::DrawDebugLines(bool toBackbuffer)
{
	IPipelineState* pso = toBackbuffer ? debugPSOBB : debugPSO;
	IShaderResourceBinding* srb = toBackbuffer ? debugSRBBB : debugSRB;
	if (!pso) return;
	if (!toBackbuffer && !debugStamp.current(samples, SceneFmt())) return;   // rebuilding
	std::vector<float> verts;
	{
		std::lock_guard<std::mutex> lock(debugMutex);
		verts = debugVerts;   // snapshot: emission may continue from the fixed thread
	}
	if (verts.empty()) return;
	const int vertCount = (int)(verts.size() / 7);

	if (!debugVB || debugVBSize < vertCount)
	{
		Trash(debugVB);   // grows mid-frame
		debugVB.Release();
		while (debugVBSize < vertCount) debugVBSize = debugVBSize ? debugVBSize * 2 : 1024;
		BufferDesc bd; bd.Name = "Debug VB"; bd.BindFlags = BIND_VERTEX_BUFFER;
		bd.Usage = USAGE_DYNAMIC; bd.CPUAccessFlags = CPU_ACCESS_WRITE;
		bd.Size = (Uint64)debugVBSize * 7 * sizeof(float);
		device->CreateBuffer(bd, nullptr, &debugVB);
		if (!debugVB) return;
	}
	{
		MapHelper<float> mv(context, debugVB, MAP_WRITE, MAP_FLAG_DISCARD);
		std::memcpy(mv, verts.data(), verts.size() * sizeof(float));
	}
	{
		MapHelper<float4x4> cb(context, debugCB, MAP_WRITE, MAP_FLAG_DISCARD);
		*cb = curView * curProj;
	}
	IBuffer* vbs[] = { debugVB };
	const Uint64 offs[] = { 0 };
	context->SetVertexBuffers(0, 1, vbs, offs, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, SET_VERTEX_BUFFERS_FLAG_RESET);
	context->SetPipelineState(pso);
	context->CommitShaderResources(srb, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	DrawAttribs da{(Uint32)vertCount, DRAW_FLAG_VERIFY_STATES};
	context->Draw(da);
}

// Depth-tested gizmo lines, drawn while the (MS) scene color+depth are still bound. PSO is built
// lazily against the current SceneFmt()/samples; the batch is consumed (never bleeds to the next camera).
void NukeDiligent::Impl::DrawDepthDebugLines()
{
	std::vector<float> verts;
	{
		std::lock_guard<std::mutex> lock(debugMutex);
		verts.swap(debugVertsDepth);   // consume
	}
	if (verts.empty() || !debugCB) return;

	if (!debugDepthPSO || debugDepthSamples != (int)samples || debugDepthFmt != SceneFmt())
	{
		if (debugDepthPSO) Trash(debugDepthPSO);   // rebuild on an MSAA/HDR flip
		debugDepthPSO.Release(); debugDepthSRB.Release();
		std::string vsSrc = shaderSource("debug.vs"), psSrc = shaderSource("debug.ps");
		if (vsSrc.empty() || psSrc.empty()) return;
		ShaderCreateInfo sci; sci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
		RefCntAutoPtr<IShader> v, p;
		sci.Desc = {"Debug VS", SHADER_TYPE_VERTEX, true}; sci.Source = vsSrc.c_str(); CreateShaderCached(sci, &v);
		sci.Desc = {"Debug PS", SHADER_TYPE_PIXEL, true};  sci.Source = psSrc.c_str(); CreateShaderCached(sci, &p);
		if (!v || !p) return;
		GraphicsPipelineStateCreateInfo ci; ci.PSODesc.Name = "Debug Lines PSO (depth)";
		auto& gp = ci.GraphicsPipeline;
		gp.NumRenderTargets = 1; gp.RTVFormats[0] = SceneFmt();
		gp.DSVFormat = TEX_FORMAT_D32_FLOAT;
		gp.PrimitiveTopology = PRIMITIVE_TOPOLOGY_LINE_LIST;
		gp.RasterizerDesc.CullMode = CULL_MODE_NONE;
		gp.DepthStencilDesc.DepthEnable = True;         // occluded by scene geometry
		gp.DepthStencilDesc.DepthWriteEnable = False;
		gp.SmplDesc.Count = samples;                    // matches the bound MS camera targets
		LayoutElement layout[] = {
			{0, 0, 3, VT_FLOAT32, False},   // pos
			{1, 0, 4, VT_FLOAT32, False},   // color
		};
		gp.InputLayout.LayoutElements = layout; gp.InputLayout.NumElements = 2;
		ci.pVS = v; ci.pPS = p;
		CreateGraphicsPipelineStateCached(ci, &debugDepthPSO);
		if (!debugDepthPSO) return;
		if (auto* sv = debugDepthPSO->GetStaticVariableByName(SHADER_TYPE_VERTEX, "DebugCB")) sv->Set(debugCB);
		debugDepthPSO->CreateShaderResourceBinding(&debugDepthSRB, true);
		debugDepthSamples = (int)samples; debugDepthFmt = SceneFmt();
	}

	const int vertCount = (int)(verts.size() / 7);
	if (!debugVB || debugVBSize < vertCount)   // shared dynamic VB (each draw maps with DISCARD)
	{
		Trash(debugVB);
		debugVB.Release();
		while (debugVBSize < vertCount) debugVBSize = debugVBSize ? debugVBSize * 2 : 1024;
		BufferDesc bd; bd.Name = "Debug VB"; bd.BindFlags = BIND_VERTEX_BUFFER;
		bd.Usage = USAGE_DYNAMIC; bd.CPUAccessFlags = CPU_ACCESS_WRITE;
		bd.Size = (Uint64)debugVBSize * 7 * sizeof(float);
		device->CreateBuffer(bd, nullptr, &debugVB);
		if (!debugVB) return;
	}
	{
		MapHelper<float> mv(context, debugVB, MAP_WRITE, MAP_FLAG_DISCARD);
		std::memcpy(mv, verts.data(), verts.size() * sizeof(float));
	}
	{
		MapHelper<float4x4> cb(context, debugCB, MAP_WRITE, MAP_FLAG_DISCARD);
		*cb = curView * curProj;
	}
	IBuffer* vbs[] = { debugVB };
	const Uint64 offs[] = { 0 };
	context->SetVertexBuffers(0, 1, vbs, offs, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, SET_VERTEX_BUFFERS_FLAG_RESET);
	context->SetPipelineState(debugDepthPSO);
	context->CommitShaderResources(debugDepthSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	DrawAttribs da{(Uint32)vertCount, DRAW_FLAG_VERIFY_STATES};
	context->Draw(da);
}

void NukeDiligent::drawEditorGrid(float step) { m_impl->gridStep = step; }

// The analytic infinite grid: a camera-centred ground quad whose PS draws AA lines with x10
// adaptive LOD and a distance fade — no geometric boundary to ever see. Depth-tested against
// the still-bound MS scene depth, drawn under the gizmo lines.
void NukeDiligent::Impl::DrawEditorGridPass()
{
	if (gridStep <= 0.0f) return;
	if (!gridPSO || gridSamples != (int)samples || gridFmt != SceneFmt())
	{
		if (gridPSO) Trash(gridPSO);
		gridPSO.Release(); gridSRB.Release();
		std::string vsSrc = shaderSource("grid.vs"), psSrc = shaderSource("grid.ps");
		if (vsSrc.empty() || psSrc.empty()) return;
		if (!gridCB)
		{
			BufferDesc cbd; cbd.Name = "GridCB"; cbd.Size = sizeof(float4x4) + sizeof(float) * 8;
			cbd.Usage = USAGE_DYNAMIC; cbd.BindFlags = BIND_UNIFORM_BUFFER; cbd.CPUAccessFlags = CPU_ACCESS_WRITE;
			device->CreateBuffer(cbd, nullptr, &gridCB);
			if (!gridCB) return;
		}
		ShaderCreateInfo sci; sci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
		RefCntAutoPtr<IShader> v, p;
		sci.Desc = {"Grid VS", SHADER_TYPE_VERTEX, true}; sci.Source = vsSrc.c_str(); CreateShaderCached(sci, &v);
		sci.Desc = {"Grid PS", SHADER_TYPE_PIXEL, true};  sci.Source = psSrc.c_str(); CreateShaderCached(sci, &p);
		if (!v || !p) return;
		GraphicsPipelineStateCreateInfo ci; ci.PSODesc.Name = "Editor Grid PSO";
		auto& gp = ci.GraphicsPipeline;
		gp.NumRenderTargets = 1; gp.RTVFormats[0] = SceneFmt();
		gp.DSVFormat = TEX_FORMAT_D32_FLOAT;
		gp.PrimitiveTopology = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		gp.RasterizerDesc.CullMode = CULL_MODE_NONE;
		gp.DepthStencilDesc.DepthEnable = True;          // occluded by scene geometry
		gp.DepthStencilDesc.DepthWriteEnable = False;
		gp.SmplDesc.Count = samples;
		auto& rt0 = gp.BlendDesc.RenderTargets[0];
		rt0.BlendEnable = True;
		rt0.SrcBlend = BLEND_FACTOR_SRC_ALPHA;  rt0.DestBlend = BLEND_FACTOR_INV_SRC_ALPHA;
		rt0.SrcBlendAlpha = BLEND_FACTOR_ZERO;  rt0.DestBlendAlpha = BLEND_FACTOR_ONE;
		ci.pVS = v; ci.pPS = p;
		CreateGraphicsPipelineStateCached(ci, &gridPSO);
		if (!gridPSO) return;
		if (auto* sv = gridPSO->GetStaticVariableByName(SHADER_TYPE_VERTEX, "GridCB")) sv->Set(gridCB);
		if (auto* sp = gridPSO->GetStaticVariableByName(SHADER_TYPE_PIXEL,  "GridCB")) sp->Set(gridCB);
		gridPSO->CreateShaderResourceBinding(&gridSRB, true);
		gridSamples = (int)samples; gridFmt = SceneFmt();
	}
	if (!gridPSO || !gridSRB) return;

	// Camera position from the inverse view; the fade reach grows with camera height so the
	// grid always dissolves well inside its quad, never at an edge.
	float4x4 invView = curView.Inverse();
	const float cx = invView.m30, cy = invView.m31, cz = invView.m32;
	struct CB { float4x4 vp; float cam[4]; float fade[4]; };
	{
		MapHelper<CB> cb(context, gridCB, MAP_WRITE, MAP_FLAG_DISCARD);
		cb->vp = curView * curProj;
		cb->cam[0] = cx; cb->cam[1] = cy; cb->cam[2] = cz; cb->cam[3] = gridStep;
		cb->fade[0] = std::max(400.0f, std::fabs(cy) * 30.0f);   // fade distance
		cb->fade[1] = 0.9f;                                      // master alpha
		cb->fade[2] = 0.0f; cb->fade[3] = 0.0f;
	}
	context->SetVertexBuffers(0, 0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, SET_VERTEX_BUFFERS_FLAG_RESET);
	context->SetPipelineState(gridPSO);
	context->CommitShaderResources(gridSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	DrawAttribs da{6, DRAW_FLAG_VERIFY_STATES};
	context->Draw(da);
}

void NukeDiligent::endCamera()
{
	m_impl->DrawEditorGridPass();    // the infinite grid, under the gizmo lines
	m_impl->DrawDepthDebugLines();   // depth-tested gizmos: against this camera's still-bound MS depth
	m_impl->FlushSprites();     // draw any pending sprite batch WHILE the (MS) camera targets are still bound
	m_impl->FlushSpritesLit();  // ...and the pending lit batch (tilemap normal-mapped runs)
	m_impl->FlushScreenPre();   // WithWorld screen-space canvas sprites: into the scene, before post
	// 1) Resolve the multisampled HDR color into the single-sample HDR texture (post-pass input).
	if (m_impl->curMSAA && m_impl->curResolveSrc && m_impl->curResolveDst)
	{
		ResolveTextureSubresourceAttribs ra;
		ra.SrcTextureTransitionMode = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
		ra.DstTextureTransitionMode = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
		ra.Format = m_impl->SceneFmt();
		m_impl->context->ResolveTextureSubresource(m_impl->curResolveSrc, m_impl->curResolveDst, ra);
	}
	m_impl->GpuPass("post");   // resolve is done; from here the frame is fullscreen work
	// 1.5) Module post hook — after the resolve, BEFORE the user chain: its output is scene content.
	ITextureView* chainSrc = m_impl->curPostSrc;
	{
		const nukediligent::WaterHooks& wh = nukediligent::ActiveWaterHooks();
		if (wh.onCameraPost && chainSrc)
		{
			m_impl->GpuPass("water.post");   // water surface/underwater compose times separately
			if (ITextureView* replaced = wh.onCameraPost(wh.user, chainSrc))
				chainSrc = replaced;
			m_impl->GpuPass("post");
		}
	}
	if (!m_impl->postChain.empty() && chainSrc && m_impl->curRTW > 0 && m_impl->curRTH > 0)
	{
		m_impl->EnsureScratch(m_impl->curRTW, m_impl->curRTH);
		const int w = m_impl->curRTW, h = m_impl->curRTH;
		ITextureView* srcSRV = chainSrc;
		int idx = 0;
		for (auto& cs : m_impl->postChain)
		{
			auto pit = m_impl->postPipes.find(cs.pipeline);
			if (pit == m_impl->postPipes.end()) continue;
			if (!pit->second.pso && !pit->second.isRTRef) continue;   // RT reflections run a ray-tracing pipeline, not a graphics PSO
			Diligent::ITexture* dstTex = m_impl->scratch[idx % 2];
			if (!dstTex) break;
			ITextureView* dstRTV = dstTex->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET);
			// The heavy built-ins time as their own gpu.* slices; "post" resumes after each
			// (slices of one name sum, so the remaining fullscreen work still reads as post).
			if (pit->second.isSSR)     // built-in screen-space reflections (samples the prepass G-buffer + depth)
			{
				if (!m_impl->gbufActive) continue;   // no prepass ran -> skip this stage (src passes through unchanged)
				m_impl->GpuPass("ssr");
				m_impl->RunSSR(pit->second, srcSRV, dstRTV, w, h, cs.params);
				m_impl->GpuPass("post");
			}
			else if (pit->second.isRTRef)   // built-in ray-traced reflections (real DXR pipeline: rt_rgen/rmiss/rchit + SBT)
			{
				if (!m_impl->gbufActive) continue;   // needs the gbuffer prepass (reflector roughness/metalness); no TLAS -> passthrough inside
				m_impl->GpuPass("rt.trace");
				m_impl->RunRTReflectPipeline(srcSRV, dstTex, w, h, cs.params);
				m_impl->GpuPass("post");
			}
			else if (pit->second.isTAA)   // built-in temporal AA (jittered accumulation; needs the depth prepass)
			{
				if (!m_impl->gbufActive) continue;   // no depth prepass -> skip (src passes through)
				m_impl->GpuPass("taa");
				m_impl->RunTAA(pit->second, srcSRV, dstTex, w, h, cs.params);
				m_impl->GpuPass("post");
			}
			else if (pit->second.isBloom)   // built-in multi-pass bloom (params: x=threshold, y=intensity)
			{
				float thr = cs.params.size() > 0 ? cs.params[0] : 1.0f;
				float inten = cs.params.size() > 1 ? cs.params[1] : 0.6f;
				m_impl->GpuPass("bloom");
				m_impl->RunBloom(srcSRV, dstRTV, w, h, thr, inten);
				m_impl->GpuPass("post");
			}
			else                       // single fullscreen custom effect
			{
				{
					MapHelper<float> cb(m_impl->context, m_impl->postParamsCB, MAP_WRITE, MAP_FLAG_DISCARD);
					int n = (int)cs.params.size(); if (n > 64) n = 64;
					for (int k = 0; k < 64; ++k) cb[k] = (k < n) ? cs.params[k] : 0.0f;
				}
				{
					MapHelper<float> fb(m_impl->context, m_impl->postFrameCB, MAP_WRITE, MAP_FLAG_DISCARD);
					fb[0] = (float)w; fb[1] = (float)h; fb[2] = w ? 1.0f / w : 0.0f; fb[3] = h ? 1.0f / h : 0.0f;
					fb[4] = fb[5] = fb[6] = fb[7] = 0.0f;
				}
				m_impl->context->SetRenderTargets(1, &dstRTV, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
				Viewport vp; vp.TopLeftX = 0; vp.TopLeftY = 0; vp.Width = (float)w; vp.Height = (float)h; vp.MinDepth = 0; vp.MaxDepth = 1;
				m_impl->context->SetViewports(1, &vp, w, h);
				if (pit->second.srcVar) pit->second.srcVar->Set(srcSRV);
				m_impl->context->SetPipelineState(pit->second.pso);
				m_impl->context->CommitShaderResources(pit->second.srb, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
				DrawAttribs da{3, DRAW_FLAG_VERIFY_STATES};
				m_impl->context->Draw(da);
			}
			srcSRV = dstTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
			++idx;
		}
		chainSrc = srcSRV;
	}
	// 3) Final tonemap/encode into the output (RT's post texture, or the backbuffer for target 0).
	m_impl->GpuPass("tonemap");
	if (chainSrc && m_impl->curPostDst)
	{
		m_impl->RunPostPass(chainSrc, m_impl->curPostDst, m_impl->curRTW, m_impl->curRTH, m_impl->curTarget == 0);

		// Gizmo lines last, over the final LDR image (target still bound by RunPostPass): TAA has
		// no velocity for lines and the RT-reflection composite would overwrite them.
		m_impl->DrawDebugLines(m_impl->curTarget == 0);
		m_impl->FlushScreenPost(m_impl->curTarget == 0);   // AfterPost screen-space canvas sprites (crisp HUD)
	}

	m_impl->GpuPassEnd();
	m_impl->curMSAA = false; m_impl->curResolveSrc = nullptr; m_impl->curResolveDst = nullptr;
	m_impl->curPostSrc = nullptr; m_impl->curPostDst = nullptr;
	m_impl->curTarget = 0;
	m_impl->cameraPassActive = false;
}

void NukeDiligent::getViewProj(float* view16, float* proj16)
{
	if (view16) memcpy(view16, m_impl->curView.Data(), 16 * sizeof(float));
	if (proj16) memcpy(proj16, m_impl->curProj.Data(), 16 * sizeof(float));
}

// Push the current global wind (direction+strength, params) for this frame.
void NukeDiligent::setWind(const float dirStrength[4], const float params[4])
{
	memcpy(m_impl->windDirStrength, dirStrength, sizeof(m_impl->windDirStrength));
	memcpy(m_impl->windParams, params, sizeof(m_impl->windParams));
	m_impl->UpdateBendCB();   // wind + pushers land in the instanced-VS BendCB once per frame
}

// Store the world positions (xyz + radius) that part foliage this frame; the CB write rides setWind.
void NukeDiligent::setBendPushers(const float* xyzr, int count)
{
	m_impl->bendPusherCount = (!xyzr || count <= 0) ? 0 : (count > 8 ? 8 : count);
	if (m_impl->bendPusherCount > 0)
		memcpy(m_impl->bendPushers, xyzr, (size_t)m_impl->bendPusherCount * 4 * sizeof(float));
}

// Store foliage bend volumes (12 floats each, see irender.h); the CB write rides setWind.
void NukeDiligent::setBendVolumes(const float* vols, int count)
{
	m_impl->bendVolumeCount = (!vols || count <= 0) ? 0 : (count > 16 ? 16 : count);
	if (m_impl->bendVolumeCount > 0)
		memcpy(m_impl->bendVolumes, vols, (size_t)m_impl->bendVolumeCount * 12 * sizeof(float));
}

// Write BendCB. Layout MUST match the instanced vertex shaders: g_WindV (dir.xyz, strength),
// g_WindT (time, pusherCount, volCount), g_WindP, g_Push[8] (xyz, radius), g_Vol[16 x 3 float4].
void NukeDiligent::Impl::UpdateBendCB()
{
	if (!bendCB || !context) return;
	struct BendData { float windV[4]; float windT[4]; float windP[4]; float push[8][4]; float vol[16][12]; };
	MapHelper<BendData> cb(context, bendCB, MAP_WRITE, MAP_FLAG_DISCARD);
	if (cb == nullptr) return;
	memcpy(cb->windV, windDirStrength, sizeof(cb->windV));
	cb->windT[0] = windParams[2];                    // wind clock (g_Wind2.z convention)
	cb->windT[1] = (float)bendPusherCount;
	cb->windT[2] = (float)bendVolumeCount;
	cb->windT[3] = 0.f;
	memcpy(cb->windP, windParams, sizeof(cb->windP));   // (turbAmount, 1/turbScale, time, gustFreq)
	memset(cb->push, 0, sizeof(cb->push));
	for (int i = 0; i < bendPusherCount; ++i)
		memcpy(cb->push[i], bendPushers[i], sizeof(float) * 4);
	memset(cb->vol, 0, sizeof(cb->vol));
	for (int i = 0; i < bendVolumeCount; ++i)
		memcpy(cb->vol[i], bendVolumes[i], sizeof(float) * 12);
}

// ---- GPU instancing (7.1) --------------------------------------------------------------

uint64_t NukeDiligent::createInstanceBuffer()
{
	uint64_t id = m_impl->nextInstBuf++;
	m_impl->instBufs[id] = Impl::InstBuf{};
	return id;
}

void NukeDiligent::updateInstanceBuffer(uint64_t id, const NukeInstanceData* data, int count)
{
	auto it = m_impl->instBufs.find(id);
	if (it == m_impl->instBufs.end() || !data || count <= 0) { if (it != m_impl->instBufs.end()) it->second.count = 0; return; }
	Impl::InstBuf& ib = it->second;
	if (!ib.buf || ib.capacity < count)
	{
		m_impl->Trash(ib.buf);   // GPU lifetime rule: never inline-release a live buffer
		ib.buf.Release();
		while (ib.capacity < count) ib.capacity = ib.capacity ? ib.capacity * 2 : 64;
		// USAGE_DEFAULT + UpdateBuffer, NOT dynamic: Vulkan dynamic buffers live in the per-frame
		// dynamic heap, which a multi-megabyte instance set exhausts.
		BufferDesc bd; bd.Name = "Instance VB"; bd.BindFlags = BIND_VERTEX_BUFFER;
		bd.Usage = USAGE_DEFAULT;
		bd.Size = (Uint64)ib.capacity * sizeof(NukeInstanceData);
		m_impl->device->CreateBuffer(bd, nullptr, &ib.buf);
		if (!ib.buf) { ib.capacity = 0; ib.count = 0; return; }
	}
	m_impl->context->UpdateBuffer(ib.buf, 0, (Uint64)count * sizeof(NukeInstanceData), data,
	                              RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	ib.count = count;
}

void NukeDiligent::destroyInstanceBuffer(uint64_t id)
{
	auto it = m_impl->instBufs.find(id);
	if (it == m_impl->instBufs.end()) return;
	m_impl->Trash(it->second.buf);   // 4-frame park — GPU may still be reading it
	m_impl->instBufs.erase(it);
}

// Draw [first, first+count) instances of `instBuf` with one mesh+material in a single call. The
// world transform comes from per-instance attributes, so the CB carries VIEW*PROJ and identity world.
void NukeDiligent::renderObjectInstanced(Mesh* mesh, Material* mat, uint64_t instBuf, int first, int count)
{
	if (m_impl->worldPipes.empty() || count <= 0) return;
	auto bit = m_impl->instBufs.find(instBuf);
	if (bit == m_impl->instBufs.end() || !bit->second.buf) return;
	if (first < 0 || first + count > bit->second.count) return;
	Impl::MeshGPU* gp = m_impl->GetMeshGPU(mesh);
	if (!gp) return;
	++m_impl->statDraws;
	m_impl->statTris += mesh ? mesh->TriCount() * count : 0;
	Impl::MeshGPU& g = *gp;

	// Identical for every instanced draw of the pass: map once, and again if a plain draw overwrote it.
	if (m_impl->instWorldCBPass != m_impl->passSerial)
	{
		struct CBData { float4x4 wvp; float4x4 world; };
		MapHelper<CBData> cb(m_impl->context, m_impl->worldCB, MAP_WRITE, MAP_FLAG_DISCARD);
		if (cb == nullptr) return;
		cb->wvp = m_impl->curView * m_impl->curProj;   // instance rows ARE the world transform
		cb->world = float4x4::Identity();
		m_impl->instWorldCBPass = m_impl->passSerial;
	}

	float col[4] = { 1, 1, 1, 1 };
	float metallic = 0.0f, roughness = 0.6f;
	float emissive[3] = { 0, 0, 0 }, emissiveI = 0.0f;
	float specF = 1.0f;
	ITextureView* srv = nullptr; ITextureView* nsrv = nullptr;
	ITextureView* mrsrv = nullptr; ITextureView* aosrv = nullptr; ITextureView* emsrv = nullptr; ITextureView* specsrv = nullptr;
	ITextureView* wipesrv = nullptr; ITextureView* heightsrv = nullptr;
	if (mat)
	{
		col[0] = (float)mat->color.r; col[1] = (float)mat->color.g; col[2] = (float)mat->color.b; col[3] = (float)mat->color.a;
		metallic = mat->metallic; roughness = mat->roughness; specF = mat->specular;
		emissive[0] = (float)mat->emissive.r; emissive[1] = (float)mat->emissive.g; emissive[2] = (float)mat->emissive.b;
		emissiveI = mat->emissiveIntensity;
		if (mat->diff) srv   = m_impl->GetTexSRV(mat->diff);
		if (mat->norm) nsrv  = m_impl->GetTexSRV(mat->norm);
		if (mat->mr)   mrsrv = m_impl->GetTexSRV(mat->mr);
		if (mat->ao)   aosrv = m_impl->GetTexSRV(mat->ao);
		if (mat->em)   emsrv = m_impl->GetTexSRV(mat->em);
		if (mat->spec) specsrv = m_impl->GetTexSRV(mat->spec);
		if (mat->wipe) wipesrv = m_impl->GetTexSRV(mat->wipe);
		if (mat->liveSurface.height) heightsrv = m_impl->GetTexSRV(mat->liveSurface.height);
	}
	// A shader without an instanced variant falls back to the default world instanced pipeline.
	uint64_t h = (mat && mat->shader && mat->shader->rendererHandle) ? mat->shader->rendererHandle
	                                                                  : m_impl->defaultWorldHandle;
	Impl::WorldPipe* pipe = m_impl->PipeFor(h);
	if (!pipe) return;
	if (!pipe->psoInst)
	{
		if (!m_impl->warnedNoInstPipe)
		{
			m_impl->warnedNoInstPipe = true;
			cout << "[NukeDiligent]\tmaterial shader has no instanced variant (handle NUKE_INSTANCED in its source) — default world shading used" << endl;
		}
		pipe = m_impl->PipeFor(m_impl->defaultWorldHandle);
		if (!pipe || !pipe->psoInst) return;
	}
	Impl::WorldPipe& wp = *pipe;

	// Per-draw flags + material CB — gated on the material changing (see renderObject).
	// The overlay draw context (source atom's values + painted mask) is per-SET: force a refill.
	if (mat && mat->liveOvCount > 0 && mat->liveDrawSet) m_impl->matCBFor = nullptr;
	if (m_impl->matCBFor != mat || m_impl->matCBPass != m_impl->passSerial)
	{
		m_impl->matCBFor = mat; m_impl->matCBPass = m_impl->passSerial;
		if (m_impl->drawFlagsCB)
		{
			MapHelper<float> fc(m_impl->context, m_impl->drawFlagsCB, MAP_WRITE, MAP_FLAG_DISCARD);
			if (fc != nullptr) { fc[0] = (mat && !mat->receiveShadows) ? 0.0f : 1.0f; fc[1] = fc[2] = fc[3] = 0.0f; }
		}
		MapHelper<Uint8> mb(m_impl->context, m_impl->worldMatCB, MAP_WRITE, MAP_FLAG_DISCARD);
		if (mb == nullptr) return;
		Uint8* p = mb;
		memset(p, 0, Impl::kMatCBBytes);
		memcpy(p + 0, col, sizeof(float) * 4);
		float nrmY = nsrv ? ((mat && mat->norm && !mat->norm->invertGreen) ? -1.0f : 1.0f) : 0.0f;
		float prm[4] = { srv ? 1.0f : 0.0f, nrmY, metallic, roughness };
		memcpy(p + 16, prm, sizeof(float) * 4);
		float prm2[4] = { mrsrv ? 1.0f : 0.0f, aosrv ? 1.0f : 0.0f, emsrv ? 1.0f : 0.0f, specF };
		memcpy(p + 32, prm2, sizeof(float) * 4);
		float emv[4] = { emissive[0], emissive[1], emissive[2], emissiveI };
		memcpy(p + 48, emv, sizeof(float) * 4);
		if (mat && mat->shader)
			for (const nuke::ShaderProp& sp : mat->shader->props)
			{
				auto pv = mat->props.find(sp.name);
				const float* v = (pv != mat->props.end()) ? pv->second.data() : sp.def;
				uint32_t bytes = (uint32_t)sp.components * sizeof(float);
				if (sp.offset + bytes <= Impl::kMatCBBytes) memcpy(p + sp.offset, v, bytes);
			}
		// Overlay draw context (source atom's values + painted mask) — same patch as the
		// plain path, at the shader's own parsed offsets.
		if (mat && mat->liveOvCount > 0 && mat->liveDrawSet && mat->shader)
		{
			static const auto kOvV = []{ std::array<std::string, Impl::kOvSlots> a; for (int s = 0; s < Impl::kOvSlots; ++s) a[s] = "g_Ov"  + std::to_string(s); return a; }();
			static const auto kOvP = []{ std::array<std::string, Impl::kOvSlots> a; for (int s = 0; s < Impl::kOvSlots; ++s) a[s] = "g_OvP" + std::to_string(s); return a; }();
			static const char* const kOvM[3] = { "g_OvM0", "g_OvM1", "g_OvM2" };
			for (const nuke::ShaderProp& sp : mat->shader->props)
			{
				if (sp.components != 4 || sp.offset + 16 > Impl::kMatCBBytes) continue;
				float* d = (float*)(p + sp.offset);
				for (int s = 0; s < Impl::kOvSlots; ++s)
				{
					if (sp.name == kOvV[s] && mat->liveDrawValue[s] >= 0.0f) d[0] = mat->liveDrawValue[s];
					if (sp.name == kOvP[s]) d[2] = mat->liveDrawMaskChan[s];
				}
				if (mat->liveDrawMask3D)
				{
					for (int r = 0; r < 3; ++r)
						if (sp.name == kOvM[r]) memcpy(d, &mat->liveDrawMaskXform[r * 4], 16);
					if (sp.name == "g_OvMQ") { d[0] = mat->liveDrawMaskRes; d[1] = 1.0f; }
				}
			}
		}
	}

	auto bindIf = [](IShaderResourceVariable* v, IDeviceObject* o, IDeviceObject*& cached)
	{ if (v && o && o != cached) { v->Set(o); cached = o; } };
	ITextureView* whiteSRV = m_impl->whiteTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
	bindIf(wp.texVarI,  srv  ? srv  : whiteSRV, wp.lastBindI[0]);
	bindIf(wp.normVarI, nsrv ? nsrv : m_impl->flatNormTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE), wp.lastBindI[1]);
	bindIf(wp.mrVarI,   mrsrv   ? mrsrv   : whiteSRV, wp.lastBindI[2]);
	bindIf(wp.aoVarI,   aosrv   ? aosrv   : whiteSRV, wp.lastBindI[3]);
	bindIf(wp.emVarI,   emsrv   ? emsrv   : whiteSRV, wp.lastBindI[4]);
	bindIf(wp.specVarI, specsrv ? specsrv : whiteSRV, wp.lastBindI[5]);
	bindIf(wp.shadowVarI, m_impl->shadowSRV ? (IDeviceObject*)m_impl->shadowSRV : (IDeviceObject*)whiteSRV, wp.lastBindI[6]);
	bindIf(wp.cubeVarI,   m_impl->shadowCubeSRV, wp.lastBindI[7]);
	bindIf(wp.probeVarI,  (m_impl->probeActive && m_impl->probeCubeSRV) ? m_impl->probeCubeSRV : m_impl->fallbackCubeSRV, wp.lastBindI[8]);
	bindIf(wp.tlasVarI,   (m_impl->rtSceneReady && m_impl->tlas) ? (IDeviceObject*)m_impl->tlas.RawPtr() : (IDeviceObject*)m_impl->fallbackTLAS.RawPtr(), wp.lastBindI[9]);
	bindIf(wp.rtInstVarI, (IDeviceObject*)(m_impl->rtInstSRV ? m_impl->rtInstSRV : m_impl->rtNrmSRV), wp.lastBindI[10]);
	bindIf(wp.wipeVarI, wipesrv ? wipesrv : whiteSRV, wp.lastBindI[11]);
	bindIf(wp.heightVarI, heightsrv ? heightsrv : whiteSRV, wp.lastBindI[12]);
	// Overlay slots: the whole-set draw context (source atom's values + painted mask) was pushed
	// by the engine before the chunk loop; patch + bind exactly like the plain path.
	{
		ITextureView* flatN = m_impl->flatNormTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
		ITextureView* ovsrv[Impl::kOvTexCount] = {};
		if (mat)
		{
			for (int s = 0; s < mat->liveOvCount && s < nuke::Material::kOverlaySlots; ++s)
			{
				const nuke::Material::OverlayRT& ov = mat->liveOv[s];
				if (ov.albedo) ovsrv[s * 4 + 0] = m_impl->GetTexSRV(ov.albedo);
				if (ov.normal) ovsrv[s * 4 + 1] = m_impl->GetTexSRV(ov.normal);
				if (ov.mrTex)  ovsrv[s * 4 + 2] = m_impl->GetTexSRV(ov.mrTex);
				if (ov.mask)   ovsrv[s * 4 + 3] = m_impl->GetTexSRV(ov.mask);
			}
			if (mat->liveDrawSet && mat->liveDrawMask3D) ovsrv[Impl::kOvSlots * 4] = m_impl->GetTexSRV(mat->liveDrawMask3D);
		if (mat->detail)    ovsrv[Impl::kOvSlots * 4 + 1] = m_impl->GetTexSRV(mat->detail);
		if (mat->detailNrm) ovsrv[Impl::kOvSlots * 4 + 2] = m_impl->GetTexSRV(mat->detailNrm);
		}
		for (int k = 0; k < Impl::kOvTexCount; ++k)
			bindIf(wp.ovVarI[k], ovsrv[k] ? ovsrv[k] : (((k < Impl::kOvSlots * 4 && (k & 3) == 1) || k == Impl::kOvSlots * 4 + 2) ? flatN : whiteSRV), wp.lastBindI[13 + k]);
		bindIf(wp.flowVarI, (mat && mat->flow) ? m_impl->GetTexSRV(mat->flow) : whiteSRV, wp.lastBindI[13 + Impl::kOvTexCount]);
		bindIf(wp.refrVarI, m_impl->refrSRV ? m_impl->refrSRV : whiteSRV, wp.lastBindI[13 + Impl::kOvTexCount + 1]);
		bindIf(wp.mskVarI, (mat && mat->mskStamp) ? m_impl->GetTexSRV(mat->mskStamp) : whiteSRV, wp.lastBindI[13 + Impl::kOvTexCount + 2]);
	}

	IDeviceContext* ctx = m_impl->context;
	IPipelineState* pso = wp.psoInst;
	if (m_impl->wireframe && wp.psoInstWire) pso = wp.psoInstWire;
	else if (mat) { if (mat->blendMode == 1 && wp.psoInstBlend) pso = wp.psoInstBlend; else if (mat->blendMode == 2 && wp.psoInstAdd) pso = wp.psoInstAdd; }
	// Consecutive chunks share vertex buffers + PSO — bind once; any plain draw resets lastInstBind.
	if (m_impl->lastInstBind.mesh != (const void*)gp || m_impl->lastInstBind.buf != instBuf ||
	    m_impl->lastInstBind.pso != (void*)pso || m_impl->lastInstBind.pass != m_impl->passSerial)
	{
		IBuffer* vbs[]  = { g.pos, g.nrm, g.uv, bit->second.buf };
		Uint64   offs[] = { 0, 0, 0, 0 };
		ctx->SetVertexBuffers(0, 4, vbs, offs, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, SET_VERTEX_BUFFERS_FLAG_RESET);
		ctx->SetPipelineState(pso);
		m_impl->lastInstBind = { gp, instBuf, (void*)pso, m_impl->passSerial };
	}
	ctx->CommitShaderResources(wp.srbInst, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	if (g.idx)
	{
		// LOD0 range only: the whole IB also carries the appended LOD shells.
		uint32_t l0First = 0, l0Count = 0;
		m_impl->LodRange(mesh, 0, l0First, l0Count);
		ctx->SetIndexBuffer(g.idx, 0, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		DrawIndexedAttribs da{(Uint32)l0Count, VT_UINT32, DRAW_FLAG_VERIFY_STATES};
		da.FirstIndexLocation    = (Uint32)l0First;
		da.NumInstances          = (Uint32)count;
		da.FirstInstanceLocation = (Uint32)first;
		ctx->DrawIndexed(da);
	}
	else
	{
		DrawAttribs da{(Uint32)g.numVerts, DRAW_FLAG_VERIFY_STATES};
		da.NumInstances          = (Uint32)count;
		da.FirstInstanceLocation = (Uint32)first;
		ctx->Draw(da);
	}
	// Consume the overlay draw context; the CB keeps the patched values for the set's remaining
	// chunks (same material -> no refill), so pushing once per set is enough.
	if (mat && mat->liveDrawSet) mat->liveDrawSet = false;
}
