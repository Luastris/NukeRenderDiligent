#include "NukeDiligentImpl.h"


void NukeDiligent::Impl::CreateShadowResources()
{
	// Release prior objects first — re-callable (shadow-res change) without Diligent's overwrite assert.
	shadowTex.Release(); shadowCmpSampler.Release();
	for (auto& v : shadowSliceDSV) v.Release();
	shadowCubeTex.Release(); shadowCubeCmpSampler.Release();
	for (auto& v : cubeFaceDSV) v.Release();
	shadowVSCB.Release(); shadowPSCB.Release(); shadowPSO.Release(); shadowSRB.Release();
	shadowSRV = nullptr; shadowCubeSRV = nullptr;

	for (int s = 0; s < SHADOW_SLOTS; ++s) lightSlot[s] = -1;
	TextureDesc td; td.Name = "Shadow Maps"; td.Type = RESOURCE_DIM_TEX_2D_ARRAY;
	td.Width = shadowRes; td.Height = shadowRes; td.ArraySize = SHADOW_SLOTS; td.MipLevels = 1;
	td.Format = TEX_FORMAT_D32_FLOAT;
	td.BindFlags = BIND_DEPTH_STENCIL | BIND_SHADER_RESOURCE;
	device->CreateTexture(td, nullptr, &shadowTex);
	if (shadowTex)
	{
		shadowSRV = shadowTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);   // whole array
		// PCF comparison sampler must be attached to the SRV — combined-texture-samplers reads it there.
		SamplerDesc cmp; cmp.MinFilter = FILTER_TYPE_COMPARISON_LINEAR; cmp.MagFilter = FILTER_TYPE_COMPARISON_LINEAR;
		cmp.MipFilter = FILTER_TYPE_COMPARISON_POINT;
		cmp.AddressU = TEXTURE_ADDRESS_CLAMP; cmp.AddressV = TEXTURE_ADDRESS_CLAMP; cmp.AddressW = TEXTURE_ADDRESS_CLAMP;
		cmp.ComparisonFunc = COMPARISON_FUNC_LESS_EQUAL;
		device->CreateSampler(cmp, &shadowCmpSampler);
		if (shadowSRV && shadowCmpSampler) shadowSRV->SetSampler(shadowCmpSampler);
		for (int s = 0; s < SHADOW_SLOTS; ++s)
		{
			TextureViewDesc vd; vd.Name = "Shadow slice DSV"; vd.ViewType = TEXTURE_VIEW_DEPTH_STENCIL;
			vd.TextureDim = RESOURCE_DIM_TEX_2D_ARRAY; vd.FirstArraySlice = (Uint32)s; vd.NumArraySlices = 1;
			shadowTex->CreateView(vd, &shadowSliceDSV[s]);
		}
	}

	// Point-light shadows: a cube depth array (6 faces per cube), sampled by direction in the world pass.
	for (int i = 0; i < kMaxLights; ++i) lightCube[i] = -1;
	TextureDesc ctd; ctd.Name = "Shadow Cubes"; ctd.Type = RESOURCE_DIM_TEX_CUBE_ARRAY;
	ctd.Width = shadowRes / 2; ctd.Height = shadowRes / 2; ctd.ArraySize = MAX_POINT_SHADOWS * 6; ctd.MipLevels = 1;
	ctd.Format = TEX_FORMAT_D32_FLOAT; ctd.BindFlags = BIND_DEPTH_STENCIL | BIND_SHADER_RESOURCE;
	device->CreateTexture(ctd, nullptr, &shadowCubeTex);
	if (shadowCubeTex)
	{
		shadowCubeSRV = shadowCubeTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
		SamplerDesc cmp; cmp.MinFilter = FILTER_TYPE_COMPARISON_LINEAR; cmp.MagFilter = FILTER_TYPE_COMPARISON_LINEAR;
		cmp.MipFilter = FILTER_TYPE_COMPARISON_POINT;
		cmp.AddressU = TEXTURE_ADDRESS_CLAMP; cmp.AddressV = TEXTURE_ADDRESS_CLAMP; cmp.AddressW = TEXTURE_ADDRESS_CLAMP;
		cmp.ComparisonFunc = COMPARISON_FUNC_LESS_EQUAL;
		device->CreateSampler(cmp, &shadowCubeCmpSampler);
		if (shadowCubeSRV && shadowCubeCmpSampler) shadowCubeSRV->SetSampler(shadowCubeCmpSampler);
		for (int f = 0; f < MAX_POINT_SHADOWS * 6; ++f)   // a DSV per cube face (a single array slice)
		{
			TextureViewDesc vd; vd.Name = "Shadow cube face DSV"; vd.ViewType = TEXTURE_VIEW_DEPTH_STENCIL;
			vd.TextureDim = RESOURCE_DIM_TEX_2D_ARRAY; vd.FirstArraySlice = (Uint32)f; vd.NumArraySlices = 1;
			shadowCubeTex->CreateView(vd, &cubeFaceDSV[f]);
		}
	}

	BufferDesc cbd; cbd.Usage = USAGE_DYNAMIC; cbd.BindFlags = BIND_UNIFORM_BUFFER; cbd.CPUAccessFlags = CPU_ACCESS_WRITE;
	cbd.Name = "ShadowVSCB"; cbd.Size = sizeof(float4x4);    device->CreateBuffer(cbd, nullptr, &shadowVSCB);
	cbd.Name = "ShadowPSCB"; cbd.Size = sizeof(float) * 8;   device->CreateBuffer(cbd, nullptr, &shadowPSCB);

	std::string vs = shaderSource("shadow.vs"), ps = shaderSource("shadow.ps");
	if (vs.empty() || ps.empty()) { cout << "[NukeDiligent]\tshadow shaders missing" << endl; return; }
	ShaderCreateInfo sci; sci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
	sci.pShaderSourceStreamFactory = shaderFactory;   // resolves #include "nukebend.hlsl"
	RefCntAutoPtr<IShader> vsh, psh;
	sci.Desc = {"Shadow VS", SHADER_TYPE_VERTEX, true}; sci.Source = vs.c_str(); CreateShaderCached(sci, &vsh);
	sci.Desc = {"Shadow PS", SHADER_TYPE_PIXEL, true};  sci.Source = ps.c_str(); CreateShaderCached(sci, &psh);
	if (!vsh || !psh) return;

	GraphicsPipelineStateCreateInfo ci; ci.PSODesc.Name = "Shadow PSO";
	auto& gp = ci.GraphicsPipeline;
	gp.NumRenderTargets = 0;
	gp.RTVFormats[0]    = TEX_FORMAT_UNKNOWN;
	gp.DSVFormat        = TEX_FORMAT_D32_FLOAT;
	gp.PrimitiveTopology = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	gp.RasterizerDesc.CullMode        = CULL_MODE_NONE;
	gp.RasterizerDesc.DepthClipEnable = True;
	gp.DepthStencilDesc.DepthEnable      = True;
	gp.DepthStencilDesc.DepthWriteEnable = True;
	gp.DepthStencilDesc.DepthFunc        = COMPARISON_FUNC_LESS;
	LayoutElement layout[] = { {0, 0, 3, VT_FLOAT32}, {1, 1, 3, VT_FLOAT32}, {2, 2, 2, VT_FLOAT32} };
	gp.InputLayout.NumElements = 3; gp.InputLayout.LayoutElements = layout;
	ShaderResourceVariableDesc vars[] = {
		{SHADER_TYPE_VERTEX, "ShadowVSCB", SHADER_RESOURCE_VARIABLE_TYPE_STATIC},
		{SHADER_TYPE_PIXEL,  "ShadowPSCB", SHADER_RESOURCE_VARIABLE_TYPE_STATIC},
		{SHADER_TYPE_PIXEL,  "g_Tex",      SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
		{SHADER_TYPE_PIXEL,  "g_WipeMask", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},   // luma-wipe holes in shadows
	};
	ci.PSODesc.ResourceLayout.Variables = vars; ci.PSODesc.ResourceLayout.NumVariables = 4;
	SamplerDesc samp; samp.MinFilter = FILTER_TYPE_LINEAR; samp.MagFilter = FILTER_TYPE_LINEAR; samp.MipFilter = FILTER_TYPE_LINEAR;
	ImmutableSamplerDesc imm[] = {{SHADER_TYPE_PIXEL, "g_Tex", samp}, {SHADER_TYPE_PIXEL, "g_WipeMask", samp}};
	ci.PSODesc.ResourceLayout.ImmutableSamplers = imm; ci.PSODesc.ResourceLayout.NumImmutableSamplers = 2;
	ci.pVS = vsh; ci.pPS = psh;
	CreateGraphicsPipelineStateCached(ci, &shadowPSO);
	if (shadowPSO)
	{
		// Bind on BOTH the PSO (static) and the SRB (mutable) — reflection decides which one is real.
		if (auto* v = shadowPSO->GetStaticVariableByName(SHADER_TYPE_VERTEX, "ShadowVSCB")) v->Set(shadowVSCB);
		if (auto* v = shadowPSO->GetStaticVariableByName(SHADER_TYPE_PIXEL,  "ShadowPSCB")) v->Set(shadowPSCB);
		shadowPSO->CreateShaderResourceBinding(&shadowSRB, true);
		if (auto* v = shadowSRB->GetVariableByName(SHADER_TYPE_VERTEX, "ShadowVSCB")) v->Set(shadowVSCB);
		if (auto* v = shadowSRB->GetVariableByName(SHADER_TYPE_PIXEL,  "ShadowPSCB")) v->Set(shadowPSCB);
		shadowPsTexVar = shadowSRB->GetVariableByName(SHADER_TYPE_PIXEL, "g_Tex");
		shadowPsWipeVar = shadowSRB->GetVariableByName(SHADER_TYPE_PIXEL, "g_WipeMask");
	}

	// Instanced twin: same depth state/CBs, per-instance world rows in buffer slot 3; the CB then
	// carries the light's view*proj only (shadow.vs.hlsl NUKE_INSTANCED).
	shadowPSOInst.Release(); shadowSRBInst.Release(); shadowPsTexVarInst = nullptr; shadowPsWipeVarInst = nullptr;
	{
		const std::string vsI = "#define NUKE_INSTANCED 1\n" + vs;
		const std::string psI = "#define NUKE_INSTANCED 1\n" + ps;
		RefCntAutoPtr<IShader> vsiI, psiI;
		sci.Desc = {"Shadow VS (inst)", SHADER_TYPE_VERTEX, true}; sci.Source = vsI.c_str(); CreateShaderCached(sci, &vsiI);
		sci.Desc = {"Shadow PS (inst)", SHADER_TYPE_PIXEL, true};  sci.Source = psI.c_str(); CreateShaderCached(sci, &psiI);
		if (vsiI && psiI)
		{
			LayoutElement layoutI[] = {
				{0, 0, 3, VT_FLOAT32}, {1, 1, 3, VT_FLOAT32}, {2, 2, 2, VT_FLOAT32},
				{3, 3, 4, VT_FLOAT32, False, LAYOUT_ELEMENT_AUTO_OFFSET, LAYOUT_ELEMENT_AUTO_STRIDE, INPUT_ELEMENT_FREQUENCY_PER_INSTANCE},
				{4, 3, 4, VT_FLOAT32, False, LAYOUT_ELEMENT_AUTO_OFFSET, LAYOUT_ELEMENT_AUTO_STRIDE, INPUT_ELEMENT_FREQUENCY_PER_INSTANCE},
				{5, 3, 4, VT_FLOAT32, False, LAYOUT_ELEMENT_AUTO_OFFSET, LAYOUT_ELEMENT_AUTO_STRIDE, INPUT_ELEMENT_FREQUENCY_PER_INSTANCE},
				{6, 3, 4, VT_FLOAT32, False, LAYOUT_ELEMENT_AUTO_OFFSET, LAYOUT_ELEMENT_AUTO_STRIDE, INPUT_ELEMENT_FREQUENCY_PER_INSTANCE},
				{7, 3, 4, VT_FLOAT32, False, LAYOUT_ELEMENT_AUTO_OFFSET, LAYOUT_ELEMENT_AUTO_STRIDE, INPUT_ELEMENT_FREQUENCY_PER_INSTANCE},
			};
			ci.GraphicsPipeline.InputLayout.LayoutElements = layoutI;
			ci.GraphicsPipeline.InputLayout.NumElements    = 8;
			ci.PSODesc.Name = "Shadow PSO (inst)";
			ci.pVS = vsiI; ci.pPS = psiI;
			CreateGraphicsPipelineStateCached(ci, &shadowPSOInst);
			if (shadowPSOInst)
			{
				if (auto* v = shadowPSOInst->GetStaticVariableByName(SHADER_TYPE_VERTEX, "ShadowVSCB")) v->Set(shadowVSCB);
				if (auto* v = shadowPSOInst->GetStaticVariableByName(SHADER_TYPE_PIXEL,  "ShadowPSCB")) v->Set(shadowPSCB);
				if (auto* v = shadowPSOInst->GetStaticVariableByName(SHADER_TYPE_VERTEX, "BendCB"))     v->Set(bendCB);   // foliage bend follows into shadows (7.4)
				shadowPSOInst->CreateShaderResourceBinding(&shadowSRBInst, true);
				if (auto* v = shadowSRBInst->GetVariableByName(SHADER_TYPE_VERTEX, "ShadowVSCB")) v->Set(shadowVSCB);
				if (auto* v = shadowSRBInst->GetVariableByName(SHADER_TYPE_PIXEL,  "ShadowPSCB")) v->Set(shadowPSCB);
				// Vulkan: cbuffers may reflect MUTABLE, not static — an unbound BendCB invalidates the descriptor set.
				if (auto* v = shadowSRBInst->GetVariableByName(SHADER_TYPE_VERTEX, "BendCB"))     v->Set(bendCB);
				shadowPsTexVarInst = shadowSRBInst->GetVariableByName(SHADER_TYPE_PIXEL, "g_Tex");
				shadowPsWipeVarInst = shadowSRBInst->GetVariableByName(SHADER_TYPE_PIXEL, "g_WipeMask");
			}
		}
	}
	cout << "[NukeDiligent]\tshadow map " << shadowRes << "x" << shadowRes << (shadowPSO ? " ready" : " FAILED")
	     << (shadowPSOInst ? " +inst" : " NO-INST") << endl;
}

void NukeDiligent::setLights(const NukeLight* lights, int count)
{
	m_impl->lights.assign(lights ? lights : nullptr, (lights && count > 0) ? lights + count : nullptr);

	// Assign a shadow-map slot + world->light-clip to each shadow-casting dir/spot light (point = later).
	for (int i = 0; i < Impl::kMaxLights; ++i) m_impl->lightSlot[i] = -1;
	int slot = 0;
	for (int i = 0; i < count && slot < Impl::SHADOW_SLOTS; ++i)
	{
		const NukeLight& L = lights[i];
		if (!L.castShadows || L.type == 1) continue;   // 0 = directional, 2 = spot (point shadows: later)
		float3 d(L.dir[0], L.dir[1], L.dir[2]);
		float  len = length(d); if (len < 1e-4f) continue; d /= len;
		float3 P, F = d, U = ((d.y < 0.f ? -d.y : d.y) > 0.95f) ? float3(0, 0, 1) : float3(0, 1, 0);
		float4x4 proj;
		if (L.type == 0)   // directional: ortho centred on the origin
		{
			const float dist = m_impl->shadowDistance, extent = m_impl->shadowDistance;
			P = d * (-dist);
			proj = float4x4::Ortho(extent, extent, 0.1f, dist * 2.0f, false);
		}
		else               // spot: perspective from the light position, fov from the cone
		{
			P = float3(L.pos[0], L.pos[1], L.pos[2]);
			float fov = 2.0f * acosf(L.spotOuter < -1.f ? -1.f : (L.spotOuter > 1.f ? 1.f : L.spotOuter));
			if (fov < 0.2f) fov = 0.2f; if (fov > 3.0f) fov = 3.0f;
			float farZ = (L.range > 1.f) ? L.range : 50.0f;
			proj = float4x4::Projection(fov, 1.0f, 0.1f, farZ, false);
		}
		float3 R = normalize(cross(U, F)); U = cross(F, R);
		float4x4 view(
			R.x, U.x, F.x, 0.f,
			R.y, U.y, F.y, 0.f,
			R.z, U.z, F.z, 0.f,
			-dot(P, R), -dot(P, U), -dot(P, F), 1.f);
		m_impl->slotVP[slot] = view * proj;
		m_impl->lightSlot[i] = slot;
		++slot;
	}
	m_impl->numShadowSlots = m_impl->shadowPSO ? slot : 0;

	// Point lights: 6 perspective faces per cube (omnidirectional depth).
	for (int i = 0; i < Impl::kMaxLights; ++i) m_impl->lightCube[i] = -1;
	int cube = 0;
	static const float3 faceF[6] = { {1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1} };
	static const float3 faceU[6] = { {0,1,0}, {0,1,0}, {0,0,-1}, {0,0,1}, {0,1,0}, {0,1,0} };
	for (int i = 0; i < count && cube < Impl::MAX_POINT_SHADOWS; ++i)
	{
		const NukeLight& L = lights[i];
		if (!L.castShadows || L.type != 1) continue;   // point only
		float3 P(L.pos[0], L.pos[1], L.pos[2]);
		float  farZ = (L.range > 1.f) ? L.range : 50.0f;
		float4x4 proj = float4x4::Projection(1.5707963f, 1.0f, 0.1f, farZ, false);   // 90deg
		for (int f = 0; f < 6; ++f)
		{
			float3 F = faceF[f], U = faceU[f];
			float3 R = normalize(cross(U, F)); U = cross(F, R);
			float4x4 view(
				R.x, U.x, F.x, 0.f,
				R.y, U.y, F.y, 0.f,
				R.z, U.z, F.z, 0.f,
				-dot(P, R), -dot(P, U), -dot(P, F), 1.f);
			m_impl->cubeFaceVP[cube * 6 + f] = view * proj;
		}
		m_impl->lightCube[i] = cube;
		++cube;
	}
	m_impl->numCubes = (m_impl->shadowPSO && m_impl->shadowCubeTex) ? cube : 0;
}

int NukeDiligent::shadowPassCount() { return m_impl->numShadowSlots + m_impl->numCubes * 6; }

void NukeDiligent::beginShadowPass(int pass)
{
	m_impl->GpuPass("shadow");   // every cascade/cube face sums into one number
	++m_impl->passSerial;   // invalidate the per-draw redundancy gates (shared CBs re-map per pass)
	if (pass < 0 || pass >= m_impl->numShadowSlots + m_impl->numCubes * 6) return;
	IDeviceContext* ctx = m_impl->context;
	ITextureView* dsv; int res;
	if (pass < m_impl->numShadowSlots)   // a 2D (directional/spot) shadow slice
	{
		m_impl->curShadowVP = m_impl->slotVP[pass];
		dsv = m_impl->shadowSliceDSV[pass]; res = m_impl->shadowRes;
	}
	else                                 // a point-light cube face
	{
		int c = pass - m_impl->numShadowSlots;
		m_impl->curShadowVP = m_impl->cubeFaceVP[c];
		dsv = m_impl->cubeFaceDSV[c]; res = m_impl->shadowRes / 2;
	}
	if (!dsv) return;
	ctx->SetRenderTargets(0, nullptr, dsv, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	ctx->ClearDepthStencil(dsv, CLEAR_DEPTH_FLAG, 1.f, 0, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	Viewport vp; vp.TopLeftX = 0; vp.TopLeftY = 0;
	vp.Width = (float)res; vp.Height = (float)res; vp.MinDepth = 0; vp.MaxDepth = 1;
	ctx->SetViewports(1, &vp, res, res);
}

void NukeDiligent::renderShadowObject(Mesh* mesh, const float pos[3], const float quat[4], const float scale[3], Material* mat)
{
	if (!mesh) return;
	uint32_t first = 0, count = 0;
	m_impl->LodRange(mesh, m_impl->SelectLod(mesh, pos, scale), first, count);
	RenderShadowRange(mesh, mat, pos, quat, scale, first, count);
}

void NukeDiligent::renderShadowObjectMulti(Mesh* mesh, Material* const* mats, int matCount,
                                           const float pos[3], const float quat[4], const float scale[3])
{
	if (!mesh) return;
	if (mesh->numIndices <= 0 || mesh->sections.empty())
	{
		renderShadowObject(mesh, pos, quat, scale, matCount > 0 ? mats[0] : nullptr);
		return;
	}
	MeshLOD L = mesh->Lod(m_impl->SelectLod(mesh, pos, scale));
	for (int s = 0; s < L.sectionCount; ++s)
	{
		MeshSection sec = mesh->Section(L.firstSection + s);
		Material* m = (sec.slot >= 0 && sec.slot < matCount && mats[sec.slot]) ? mats[sec.slot]
		            : (matCount > 0 ? mats[0] : nullptr);
		if (m && !m->castShadows) continue;   // per-section cast gate
		RenderShadowRange(mesh, m, pos, quat, scale, sec.firstIndex, sec.indexCount);
	}
}

void NukeDiligent::RenderShadowRange(Mesh* mesh, Material* mat,
                                     const float pos[3], const float quat[4], const float scale[3],
                                     uint32_t firstIndex, uint32_t indexCount)
{
	if (!m_impl->shadowPSO || indexCount == 0) return;
	Impl::MeshGPU* gp = m_impl->GetMeshGPU(mesh); if (!gp) return;
	++m_impl->statDraws;                              // frame stats (status bar)
	m_impl->statTris += (int)indexCount / 3;
	Impl::MeshGPU& g = *gp;
	float4x4 world = float4x4::Scale(scale[0], scale[1], scale[2])
	               * Diligent::Quaternion<float>(quat[0], quat[1], quat[2], quat[3]).ToMatrix()
	               * float4x4::Translation(pos[0], pos[1], pos[2]);
	float4x4 wvp = world * m_impl->curShadowVP;

	ITextureView* base = (mat && mat->diff) ? m_impl->GetTexSRV(mat->diff) : nullptr;
	{
		MapHelper<float4x4> cb(m_impl->context, m_impl->shadowVSCB, MAP_WRITE, MAP_FLAG_DISCARD);
		*cb = wvp;
	}
	{
		MapHelper<float> cb(m_impl->context, m_impl->shadowPSCB, MAP_WRITE, MAP_FLAG_DISCARD);
		cb[0] = mat ? (float)mat->color.a : 1.0f;
		cb[1] = base ? 1.0f : 0.0f;
		// LiveMaterial: cutout threshold + wipe threshold + UV transform mirror the color pass.
		cb[2] = (mat && mat->blendMode == 3) ? mat->alphaCutoff : 0.0f;
		cb[3] = (mat && mat->wipe && mat->wipeThreshold > 0.0f) ? mat->wipeThreshold : 0.0f;
		cb[4] = mat ? (float)mat->uvTiling.x : 0.0f; cb[5] = mat ? (float)mat->uvTiling.y : 0.0f;
		cb[6] = mat ? (float)mat->uvOffset.x + mat->uvAnim[0] : 0.0f;
		cb[7] = mat ? (float)mat->uvOffset.y + mat->uvAnim[1] : 0.0f;
	}
	if (m_impl->shadowPsTexVar)
		m_impl->shadowPsTexVar->Set(base ? base : m_impl->whiteTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
	if (m_impl->shadowPsWipeVar)
		m_impl->shadowPsWipeVar->Set(mat && mat->wipe ? (IDeviceObject*)m_impl->GetTexSRV(mat->wipe)
		                                              : (IDeviceObject*)m_impl->whiteTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));

	IDeviceContext* ctx = m_impl->context;
	IBuffer* vbs[] = { g.PosBuf(), g.NrmBuf(), g.UVBuf() };
	Uint64   offs[] = { g.PosOfs(), g.NrmOfs(), g.UVOfs() };
	ctx->SetVertexBuffers(0, 3, vbs, offs, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, SET_VERTEX_BUFFERS_FLAG_RESET);
	ctx->SetPipelineState(m_impl->shadowPSO);
	ctx->CommitShaderResources(m_impl->shadowSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	if (g.IdxBuf())
	{
		ctx->SetIndexBuffer(g.IdxBuf(), g.IdxOfs(), RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
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
}

// Draw the [first, first+count) range of `instBuf` into the current shadow slice. The CB carries
// the light's view*proj only; instance rows supply the world matrices.
void NukeDiligent::renderShadowInstanced(Mesh* mesh, uint64_t instBuf, int first, int count, Material* mat)
{
	if (!m_impl->shadowPSOInst || count <= 0) return;
	auto bit = m_impl->instBufs.find(instBuf);
	if (bit == m_impl->instBufs.end() || !bit->second.buf) return;
	if (first < 0 || first + count > bit->second.count) return;
	Impl::MeshGPU* gp = m_impl->GetMeshGPU(mesh); if (!gp) return;
	++m_impl->statDraws;
	m_impl->statTris += mesh ? mesh->TriCount() * count : 0;
	Impl::MeshGPU& g = *gp;

	ITextureView* base = (mat && mat->diff) ? m_impl->GetTexSRV(mat->diff) : nullptr;
	{
		MapHelper<float4x4> cb(m_impl->context, m_impl->shadowVSCB, MAP_WRITE, MAP_FLAG_DISCARD);
		if (cb == nullptr) return;
		*cb = m_impl->curShadowVP;
	}
	{
		MapHelper<float> cb(m_impl->context, m_impl->shadowPSCB, MAP_WRITE, MAP_FLAG_DISCARD);
		if (cb == nullptr) return;
		cb[0] = mat ? (float)mat->color.a : 1.0f;
		cb[1] = base ? 1.0f : 0.0f;
		cb[2] = (mat && mat->blendMode == 3) ? mat->alphaCutoff : 0.0f;
		cb[3] = (mat && mat->wipe && mat->wipeThreshold > 0.0f) ? mat->wipeThreshold : 0.0f;
		cb[4] = mat ? (float)mat->uvTiling.x : 0.0f; cb[5] = mat ? (float)mat->uvTiling.y : 0.0f;
		cb[6] = mat ? (float)mat->uvOffset.x + mat->uvAnim[0] : 0.0f;
		cb[7] = mat ? (float)mat->uvOffset.y + mat->uvAnim[1] : 0.0f;
	}
	if (m_impl->shadowPsTexVarInst)
		m_impl->shadowPsTexVarInst->Set(base ? base : m_impl->whiteTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));
	if (m_impl->shadowPsWipeVarInst)
		m_impl->shadowPsWipeVarInst->Set(mat && mat->wipe ? (IDeviceObject*)m_impl->GetTexSRV(mat->wipe)
		                                                  : (IDeviceObject*)m_impl->whiteTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE));

	IDeviceContext* ctx = m_impl->context;
	IBuffer* vbs[]  = { g.PosBuf(), g.NrmBuf(), g.UVBuf(), bit->second.buf };
	Uint64   offs[] = { g.PosOfs(), g.NrmOfs(), g.UVOfs(), 0 };
	ctx->SetVertexBuffers(0, 4, vbs, offs, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, SET_VERTEX_BUFFERS_FLAG_RESET);
	ctx->SetPipelineState(m_impl->shadowPSOInst);
	ctx->CommitShaderResources(m_impl->shadowSRBInst, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	if (g.IdxBuf())
	{
		// LOD0 range only: the whole IB also carries the appended LOD shells.
		uint32_t l0First = 0, l0Count = 0;
		m_impl->LodRange(mesh, 0, l0First, l0Count);
		ctx->SetIndexBuffer(g.IdxBuf(), g.IdxOfs(), RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
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
}

void NukeDiligent::endShadowPass() { /* the next beginCamera rebinds the camera targets */ }

void NukeDiligent::setShadowSettings(int resolution, float distance, float depthBias, float normalBias, float softness)
{
	int res = resolution < 256 ? 256 : (resolution > 8192 ? 8192 : resolution);
	// LAST writer this frame wins (the apply site skips a rebuild when it matches the current
	// res) — a first-differing-writer pending would let two worlds oscillate the map forever.
	m_impl->pendingShadowRes = res;
	m_impl->shadowDistance   = distance > 1.0f ? distance : 1.0f;
	m_impl->shadowDepthBias  = depthBias;
	m_impl->shadowNormalBias = normalBias;
	m_impl->shadowSoftness   = softness > 0.0f ? softness : 0.0f;
}
