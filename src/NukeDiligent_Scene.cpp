#include "NukeDiligentImpl.h"


// Camera basis -> curView/curProj/curCamPos. Shared by beginCamera and the SSR gbuffer prepass so both use the
// exact same transform (left-handed look-at; same projection as beginCamera).
void NukeDiligent::Impl::SetCameraViewProj(const NukeCameraDesc& cam, int w, int h)
{
	const float aspect = (h > 0) ? (float)w / (float)h : 1.0f;
	float3 P(cam.camPos[0], cam.camPos[1], cam.camPos[2]);
	float3 F = normalize(float3(cam.camFwd[0], cam.camFwd[1], cam.camFwd[2]));
	float3 U = float3(cam.camUp[0], cam.camUp[1], cam.camUp[2]);
	float3 R = normalize(cross(U, F)); U = cross(F, R);
	curView = float4x4(R.x, U.x, F.x, 0.f, R.y, U.y, F.y, 0.f, R.z, U.z, F.z, 0.f, -dot(P, R), -dot(P, U), -dot(P, F), 1.f);
	// Projection: perspective, orthographic, or an element-wise blend of the two matrices (the
	// standard perspective<->ortho tween — the engine animates cam.ortho for a smooth transition).
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
	if (m_impl->worldPipes.empty()) return;
	Impl::MeshGPU* gp = m_impl->GetMeshGPU(mesh);
	if (!gp) return;
	++m_impl->statDraws;                              // frame stats (status bar)
	m_impl->statTris += mesh ? mesh->numVerts / 3 : 0;
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

	// Material: base color + PBR maps/params (fallbacks when none).
	float col[4] = { 1, 1, 1, 1 };
	float metallic = 0.0f, roughness = 0.6f;
	float emissive[3] = { 0, 0, 0 }, emissiveI = 0.0f;
	float specF = 1.0f;
	ITextureView* srv = nullptr; ITextureView* nsrv = nullptr;
	ITextureView* mrsrv = nullptr; ITextureView* aosrv = nullptr; ITextureView* emsrv = nullptr; ITextureView* specsrv = nullptr;
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
	}
	// Pick the pipeline for this material's shader (fallback to the built-in "world" pipeline).
	uint64_t h = (mat && mat->shader && mat->shader->rendererHandle) ? mat->shader->rendererHandle
	                                                                  : m_impl->defaultWorldHandle;
	auto pit = m_impl->worldPipes.find(h);
	if (pit == m_impl->worldPipes.end()) pit = m_impl->worldPipes.find(m_impl->defaultWorldHandle);
	if (pit == m_impl->worldPipes.end()) return;
	Impl::WorldPipe& wp = pit->second;

	// Material constant buffer: standard color @0 + params @16, then the shader's custom props at
	// their engine-parsed offsets (Shader::props). The renderer consumes the schema the engine
	// parsed from the shader source — it never reads shader files itself.
	{
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
				// Value from the material INSTANCE's prop map (data only — no engine symbols linked);
				// unset -> the shader's HLSL default.
				auto pv = mat->props.find(sp.name);
				const float* v = (pv != mat->props.end()) ? pv->second.data() : sp.def;
				uint32_t bytes = (uint32_t)sp.components * sizeof(float);
				if (sp.offset + bytes <= Impl::kMatCBBytes) memcpy(p + sp.offset, v, bytes);
			}
	}

	if (wp.texVar)
		wp.texVar->Set(srv ? srv : m_impl->whiteTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
	ITextureView* whiteSRV = m_impl->whiteTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
	if (wp.normVar)
		wp.normVar->Set(nsrv ? nsrv : m_impl->flatNormTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
	if (wp.mrVar) wp.mrVar->Set(mrsrv ? mrsrv : whiteSRV);
	if (wp.aoVar) wp.aoVar->Set(aosrv ? aosrv : whiteSRV);
	if (wp.emVar) wp.emVar->Set(emsrv ? emsrv : whiteSRV);
	if (wp.specVar) wp.specVar->Set(specsrv ? specsrv : whiteSRV);
	if (wp.shadowVar) wp.shadowVar->Set(m_impl->shadowSRV ? m_impl->shadowSRV : whiteSRV);
	if (wp.cubeVar && m_impl->shadowCubeSRV) wp.cubeVar->Set(m_impl->shadowCubeSRV);
	if (wp.probeVar) wp.probeVar->Set((m_impl->probeActive && m_impl->probeCubeSRV) ? m_impl->probeCubeSRV : m_impl->fallbackCubeSRV);
	if (wp.tlasVar)  wp.tlasVar->Set((m_impl->rtSceneReady && m_impl->tlas) ? (IDeviceObject*)m_impl->tlas.RawPtr() : (IDeviceObject*)m_impl->fallbackTLAS.RawPtr());

	IDeviceContext* ctx = m_impl->context;
	IBuffer* vbs[]    = { g.pos, g.nrm, g.uv };
	Uint64   offs[]   = { 0, 0, 0 };
	ctx->SetVertexBuffers(0, 3, vbs, offs, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, SET_VERTEX_BUFFERS_FLAG_RESET);
	// Pick the blend variant for this material (engine sorts transparent/additive back-to-front).
	IPipelineState* pso = wp.pso;
	if (mat) { if (mat->blendMode == 1 && wp.psoBlend) pso = wp.psoBlend; else if (mat->blendMode == 2 && wp.psoAdd) pso = wp.psoAdd; }
	ctx->SetPipelineState(pso);
	ctx->CommitShaderResources(wp.srb, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	DrawAttribs da{(Uint32)g.numVerts, DRAW_FLAG_VERIFY_STATES};
	ctx->Draw(da);
}

void NukeDiligent::renderSelectionOutline(Mesh* mesh, const float pos[3], const float quat[4], const float scale[3])
{
	if (!m_impl->outlineMaskPSO || !m_impl->outlineEdgePSO || !m_impl->curRTV) return;
	Impl::MeshGPU* gp = m_impl->GetMeshGPU(mesh);
	if (!gp) return;
	Impl::MeshGPU& g = *gp;
	m_impl->EnsureOutlineMask(m_impl->curRTW, m_impl->curRTH);
	if (!m_impl->outlineMaskRTV || !m_impl->outlineMaskSRV) return;

	IDeviceContext* ctx = m_impl->context;

	// --- pass 1: render the selected mesh into the mask RT (alpha = 1 over the object) ---
	{
		float4x4 world = float4x4::Scale(scale[0], scale[1], scale[2])
		               * Diligent::Quaternion<float>(quat[0], quat[1], quat[2], quat[3]).ToMatrix()
		               * float4x4::Translation(pos[0], pos[1], pos[2]);
		float4x4 wvp = world * m_impl->curView * m_impl->curProj;
		struct CBData { float4x4 wvp; float4x4 world; };
		{ MapHelper<CBData> cb(ctx, m_impl->worldCB, MAP_WRITE, MAP_FLAG_DISCARD); cb->wvp = wvp; cb->world = world; }

		const float zero[4] = { 0, 0, 0, 0 };
		ctx->SetRenderTargets(1, &m_impl->outlineMaskRTV, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		ctx->ClearRenderTarget(m_impl->outlineMaskRTV, zero, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		IBuffer* vbs[]  = { g.pos, g.nrm, g.uv };
		Uint64   offs[] = { 0, 0, 0 };
		ctx->SetVertexBuffers(0, 3, vbs, offs, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, SET_VERTEX_BUFFERS_FLAG_RESET);
		ctx->SetPipelineState(m_impl->outlineMaskPSO);
		ctx->CommitShaderResources(m_impl->outlineMaskSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		DrawAttribs da{(Uint32)g.numVerts, DRAW_FLAG_VERIFY_STATES};
		ctx->Draw(da);
	}

	// --- pass 2: fullscreen edge-detect over the mask -> draw the border into the camera RT ---
	{
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
	}
}

// Fill the world FrameCB (camera pos, ambient, lights, shadow maps/params, sky IBL, reflection probe).
// Shared by the camera pass and the probe cube-face passes (no duplication).
void NukeDiligent::Impl::WriteFrameCB(const float3& P)
{
	MapHelper<FrameCBData> fb(context, worldFrameCB, MAP_WRITE, MAP_FLAG_DISCARD);
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
	int n = (int)src.size(); if (n > kMaxLights) n = kMaxLights;
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
	fb->shadowParams[1] = shadowNormalBias;
	fb->shadowParams[2] = (1.0f / (float)shadowRes) * shadowSoftness;
	fb->shadowParams[3] = shadowDepthBias;
	for (int k = 0; k < 3; ++k) { fb->skyTop[k] = sky.top[k]; fb->skyHorizon[k] = sky.horizon[k]; fb->skyGround[k] = sky.ground[k]; }
	fb->skyParams[0] = sky.skyIntensity; fb->skyParams[1] = (sky.mode == 1) ? 1.0f : 0.0f;
	fb->skyParams[2] = hdr ? 0.0f : 1.0f; fb->skyParams[3] = sky.whitePoint;   // .w = SDR tonemap white point (world.ps HDR-off path)
	const bool probe = probeActive && probeCubeSRV;   // off during the probe's own capture -> no feedback
	fb->probePos[0] = probePos[0]; fb->probePos[1] = probePos[1]; fb->probePos[2] = probePos[2]; fb->probePos[3] = probe ? 1.0f : 0.0f;
	fb->probeParams[0] = probeIntensity; fb->probeParams[1] = probeMaxMip; fb->probeParams[2] = 0; fb->probeParams[3] = 0;
	const bool box = probe && (probeBoxHalf[0] > 0.f || probeBoxHalf[1] > 0.f || probeBoxHalf[2] > 0.f);
	fb->probeBox[0] = probeBoxHalf[0]; fb->probeBox[1] = probeBoxHalf[1]; fb->probeBox[2] = probeBoxHalf[2]; fb->probeBox[3] = box ? 1.0f : 0.0f;
}

void NukeDiligent::beginCamera(const NukeCameraDesc& cam)
{
	m_impl->curTarget = cam.target;   // feedback guard: GetTexSRV won't sample the RT we draw into
	m_impl->curMSAA = false; m_impl->curResolveSrc = nullptr; m_impl->curResolveDst = nullptr;
	m_impl->curPostSrc = nullptr; m_impl->curPostDst = nullptr;
	const bool ms = m_impl->samples > 1;
	ITextureView* rtv = nullptr;
	ITextureView* dsv = nullptr;
	int w = 0, h = 0;
	if (cam.target == 0)
	{
		// The backbuffer path always renders through an HDR intermediate, then the post pass tonemaps it
		// into the actual (LDR) swap-chain backbuffer in endCamera.
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
	m_impl->curRTV = rtv; m_impl->curRTW = w; m_impl->curRTH = h;   // for the selection-outline pass
	m_impl->cameraPassActive = true;   // sprites may draw from here until endCamera completes

	IDeviceContext* ctx = m_impl->context;
	ctx->SetRenderTargets(1, &rtv, dsv, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	// Clear to the camera's Background colour. Its ALPHA drives transparency on a composited
	// window: opaque geometry writes alpha 1 over it, a procedural sky fills it opaque, and the
	// final pass carries the alpha premultiplied so the desktop shows where the background alpha
	// (and no geometry) is below 1. On an opaque window the alpha is ignored by the final pass.
	ctx->ClearRenderTarget(rtv, cam.clear, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	if (dsv)
		ctx->ClearDepthStencil(dsv, CLEAR_DEPTH_FLAG, 1.f, 0, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	Viewport vp; vp.TopLeftX = 0; vp.TopLeftY = 0; vp.Width = (float)w; vp.Height = (float)h; vp.MinDepth = 0; vp.MaxDepth = 1;
	ctx->SetViewports(1, &vp, w, h);

	m_impl->SetCameraViewProj(cam, w, h);   // curView/curProj/curCamPos (shared with the SSR gbuffer prepass)
	m_impl->curProjNoJitter = m_impl->curProj;   // unjittered — TAA reprojection + the depth prepass use this
	if (m_impl->curTAA && w > 0 && h > 0)        // TAA: jitter the COLOUR projection sub-pixel (Halton); depth stays clean
	{
		m_impl->curProj.m[2][0] += m_impl->curJitterX * 2.0f / (float)w;   // pixel -> NDC (row-vector: clip.x += vz*offset)
		m_impl->curProj.m[2][1] += m_impl->curJitterY * 2.0f / (float)h;
	}

	// PBR lighting buffer for this pass: camera pos + ambient + scene lights (default sun if none).
	float3 P(cam.camPos[0], cam.camPos[1], cam.camPos[2]);
	m_impl->WriteFrameCB(P);

	m_impl->DrawSky();   // procedural sky behind the scene (after clear, before geometry)
}

void NukeDiligent::setSky(const NukeSky& s) { m_impl->sky = s; m_impl->toneExposure = s.exposure; m_impl->toneWhite = s.whitePoint; }

// Halton low-discrepancy sequence (1-based index) — even sub-pixel coverage for the TAA jitter.
static float Halton(int i, int b) { float f = 1.0f, r = 0.0f; while (i > 0) { f /= b; r += f * (i % b); i /= b; } return r; }

// Enable/disable TAA for the camera about to render. When enabled, advance the jitter (Halton 2,3; ±0.5 px) so the
// next beginCamera offsets the colour projection. Called by World::Render per camera before the prepass/beginCamera.
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

void NukeDiligent::Impl::CreateDebugResources()
{
	debugPSO.Release(); debugPSOBB.Release(); debugSRB.Release(); debugSRBBB.Release(); debugCB.Release();
	std::string vs = shaderSource("debug.vs"), ps = shaderSource("debug.ps");
	if (vs.empty() || ps.empty()) { cout << "[NukeDiligent]	debug-line shaders missing" << endl; return; }
	ShaderCreateInfo sci; sci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
	RefCntAutoPtr<IShader> v, p;
	sci.Desc = {"Debug VS", SHADER_TYPE_VERTEX, true}; sci.Source = vs.c_str(); device->CreateShader(sci, &v);
	sci.Desc = {"Debug PS", SHADER_TYPE_PIXEL, true};  sci.Source = ps.c_str(); device->CreateShader(sci, &p);
	if (!v || !p) return;

	BufferDesc cbd; cbd.Name = "DebugCB"; cbd.Size = sizeof(float4x4);
	cbd.Usage = USAGE_DYNAMIC; cbd.BindFlags = BIND_UNIFORM_BUFFER; cbd.CPUAccessFlags = CPU_ACCESS_WRITE;
	device->CreateBuffer(cbd, nullptr, &debugCB);

	// Post-last overlay: LDR targets, single-sample, NO depth (the scene depth is
	// multisampled and long resolved by this point; gizmos read on top of the image).
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
		device->CreateGraphicsPipelineState(ci, &pso);
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

void NukeDiligent::endCamera()
{
	m_impl->FlushSprites();     // draw any pending sprite batch WHILE the (MS) camera targets are still bound
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
	// 2) Custom effect chain (HDR): ping-pong the resolved scene through each post pipeline.
	ITextureView* chainSrc = m_impl->curPostSrc;
	if (!m_impl->postChain.empty() && m_impl->curPostSrc && m_impl->curRTW > 0 && m_impl->curRTH > 0)
	{
		m_impl->EnsureScratch(m_impl->curRTW, m_impl->curRTH);
		const int w = m_impl->curRTW, h = m_impl->curRTH;
		ITextureView* srcSRV = m_impl->curPostSrc;
		int idx = 0;
		for (auto& cs : m_impl->postChain)
		{
			auto pit = m_impl->postPipes.find(cs.pipeline);
			if (pit == m_impl->postPipes.end()) continue;
			if (!pit->second.pso && !pit->second.isRTRef) continue;   // RT reflections run a ray-tracing pipeline, not a graphics PSO
			Diligent::ITexture* dstTex = m_impl->scratch[idx % 2];
			if (!dstTex) break;
			ITextureView* dstRTV = dstTex->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET);
			if (pit->second.isSSR)     // built-in screen-space reflections (samples the prepass G-buffer + depth)
			{
				if (!m_impl->gbufActive) continue;   // no prepass ran -> skip this stage (src passes through unchanged)
				m_impl->RunSSR(pit->second, srcSRV, dstRTV, w, h, cs.params);
			}
			else if (pit->second.isRTRef)   // built-in ray-traced reflections (real DXR pipeline: rt_rgen/rmiss/rchit + SBT)
			{
				if (!m_impl->gbufActive) continue;   // needs the gbuffer prepass (reflector roughness/metalness); no TLAS -> passthrough inside
				m_impl->RunRTReflectPipeline(srcSRV, dstTex, w, h, cs.params);
			}
			else if (pit->second.isTAA)   // built-in temporal AA (jittered accumulation; needs the depth prepass)
			{
				if (!m_impl->gbufActive) continue;   // no depth prepass -> skip (src passes through)
				m_impl->RunTAA(pit->second, srcSRV, dstTex, w, h, cs.params);
			}
			else if (pit->second.isBloom)   // built-in multi-pass bloom (params: x=threshold, y=intensity)
			{
				float thr = cs.params.size() > 0 ? cs.params[0] : 1.0f;
				float inten = cs.params.size() > 1 ? cs.params[1] : 0.6f;
				m_impl->RunBloom(srcSRV, dstRTV, w, h, thr, inten);
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
	if (chainSrc && m_impl->curPostDst)
	{
		m_impl->RunPostPass(chainSrc, m_impl->curPostDst, m_impl->curRTW, m_impl->curRTH, m_impl->curTarget == 0);

		// Debug/gizmo lines LAST, over the final LDR image (target still bound by RunPostPass):
		// TAA has no velocity for lines and the RT-reflection composite overwrites them -
		// post-last dodges both. No depth here -> gizmos read on top (X-ray); fine for an editor.
		m_impl->DrawDebugLines(m_impl->curTarget == 0);
		m_impl->FlushScreenPost(m_impl->curTarget == 0);   // AfterPost screen-space canvas sprites (crisp HUD)
	}

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
