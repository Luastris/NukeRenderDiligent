#include "NukeDiligentImpl.h"

// Screen-space decals: rasterize the decal box, reconstruct the surface from the depth prepass,
// project the texture onto it. Two PSOs (alpha blend / additive); requires gbufDepth.

namespace {
struct DecalCBData
{
	Diligent::float4x4 wvp, invWorld, invViewProj;
	Diligent::float4   tint, params, projAxis, res;
};
}  // namespace

void NukeDiligent::Impl::CreateDecalResources()
{
	decalPSO.Release(); decalPSOAdd.Release(); decalPSOMod.Release();
	decalSRB.Release(); decalSRBAdd.Release(); decalSRBMod.Release(); decalCB.Release();
	decalTexVar = decalDepthVar = decalTexVarAdd = decalDepthVarAdd = decalTexVarMod = decalDepthVarMod = nullptr;
	std::string vs = shaderSource("decal.vs"), ps = shaderSource("decal.ps");
	if (vs.empty() || ps.empty()) { std::cout << "[NukeDiligent]\tdecal shaders missing" << std::endl; return; }

	ShaderCreateInfo sci; sci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
	RefCntAutoPtr<IShader> v, p;
	sci.Desc = {"Decal VS", SHADER_TYPE_VERTEX, true}; sci.Source = vs.c_str(); CreateShaderCached(sci, &v);
	sci.Desc = {"Decal PS", SHADER_TYPE_PIXEL, true};  sci.Source = ps.c_str(); CreateShaderCached(sci, &p);
	if (!v || !p) return;

	BufferDesc cbd; cbd.Name = "DecalCB"; cbd.Size = sizeof(DecalCBData);
	cbd.Usage = USAGE_DYNAMIC; cbd.BindFlags = BIND_UNIFORM_BUFFER; cbd.CPUAccessFlags = CPU_ACCESS_WRITE;
	device->CreateBuffer(cbd, nullptr, &decalCB);

	// Unit cube [-0.5,0.5]^3, 36 verts (position only).
	if (!decalVB)
	{
		const float h = 0.5f;
		const float C[8][3] = { {-h,-h,-h},{h,-h,-h},{h,h,-h},{-h,h,-h},{-h,-h,h},{h,-h,h},{h,h,h},{-h,h,h} };
		const int   F[6][4] = { {4,5,6,7},{1,0,3,2},{5,1,2,6},{0,4,7,3},{7,6,2,3},{0,1,5,4} };
		const int   tri[6]  = { 0,1,2, 0,2,3 };
		float cube[36 * 3]; int vi = 0;
		for (int f = 0; f < 6; ++f) for (int t = 0; t < 6; ++t)
		{ const float* c = C[F[f][tri[t]]]; cube[vi*3+0]=c[0]; cube[vi*3+1]=c[1]; cube[vi*3+2]=c[2]; ++vi; }
		BufferDesc bd; bd.Name = "Decal Cube VB"; bd.BindFlags = BIND_VERTEX_BUFFER; bd.Usage = USAGE_IMMUTABLE;
		bd.Size = sizeof(cube); BufferData bdata{ cube, sizeof(cube) };
		device->CreateBuffer(bd, &bdata, &decalVB);
	}

	LayoutElement layout[] = { {0, 0, 3, VT_FLOAT32, False} };   // position only
	SamplerDesc smp; smp.MinFilter = FILTER_TYPE_LINEAR; smp.MagFilter = FILTER_TYPE_LINEAR; smp.MipFilter = FILTER_TYPE_LINEAR;
	smp.AddressU = TEXTURE_ADDRESS_CLAMP; smp.AddressV = TEXTURE_ADDRESS_CLAMP; smp.AddressW = TEXTURE_ADDRESS_CLAMP;
	ShaderResourceVariableDesc vars[] = {
		{SHADER_TYPE_PIXEL, "g_DecalTex", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
		{SHADER_TYPE_PIXEL, "g_Depth",    SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
	};
	ImmutableSamplerDesc imms[] = {
		{SHADER_TYPE_PIXEL, "g_DecalTex", smp},
		{SHADER_TYPE_PIXEL, "g_Depth",    smp},
	};

	auto build = [&](int blendMode, const char* nm, RefCntAutoPtr<IPipelineState>& pso,
	                 RefCntAutoPtr<IShaderResourceBinding>& srb, IShaderResourceVariable*& tv, IShaderResourceVariable*& dv)
	{
		GraphicsPipelineStateCreateInfo ci; ci.PSODesc.Name = nm;
		auto& gp = ci.GraphicsPipeline;
		gp.NumRenderTargets = 1; gp.RTVFormats[0] = SceneFmt();
		gp.DSVFormat = TEX_FORMAT_D32_FLOAT;                 // a depth buffer is bound; we just don't test/write it
		gp.PrimitiveTopology = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		gp.RasterizerDesc.CullMode = CULL_MODE_FRONT;        // back faces -> box footprint from any camera position
		gp.DepthStencilDesc.DepthEnable = False;
		gp.DepthStencilDesc.DepthWriteEnable = False;
		gp.SmplDesc.Count = samples;
		auto& rt = gp.BlendDesc.RenderTargets[0];
		rt.BlendEnable = True; rt.BlendOp = BLEND_OPERATION_ADD; rt.BlendOpAlpha = BLEND_OPERATION_ADD;
		// PS outputs PREMULTIPLIED color. Albedo = classic alpha on top (bright decals stay
		// bright); Additive = glow; Stain = dest * lerp(1, tex.rgb, a) - tints the LIT
		// surface, so lighting and shadows show through it.
		if      (blendMode == 1) { rt.SrcBlend = BLEND_FACTOR_ONE;        rt.DestBlend = BLEND_FACTOR_ONE; }
		else if (blendMode == 2) { rt.SrcBlend = BLEND_FACTOR_DEST_COLOR; rt.DestBlend = BLEND_FACTOR_INV_SRC_ALPHA; }
		else                     { rt.SrcBlend = BLEND_FACTOR_ONE;        rt.DestBlend = BLEND_FACTOR_INV_SRC_ALPHA; }
		rt.SrcBlendAlpha = BLEND_FACTOR_ZERO; rt.DestBlendAlpha = BLEND_FACTOR_ONE;
		gp.InputLayout.LayoutElements = layout; gp.InputLayout.NumElements = 1;
		ci.pVS = v; ci.pPS = p;
		ci.PSODesc.ResourceLayout.Variables            = vars; ci.PSODesc.ResourceLayout.NumVariables         = 2;
		ci.PSODesc.ResourceLayout.ImmutableSamplers    = imms; ci.PSODesc.ResourceLayout.NumImmutableSamplers = 2;
		ci.PSODesc.ResourceLayout.DefaultVariableType  = SHADER_RESOURCE_VARIABLE_TYPE_STATIC;
		CreateGraphicsPipelineStateCached(ci, &pso);
		if (pso)
		{
			if (auto* sv = pso->GetStaticVariableByName(SHADER_TYPE_VERTEX, "DecalCB")) sv->Set(decalCB);
			if (auto* sp = pso->GetStaticVariableByName(SHADER_TYPE_PIXEL,  "DecalCB")) sp->Set(decalCB);
			pso->CreateShaderResourceBinding(&srb, true);
			if (srb) { tv = srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_DecalTex"); dv = srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Depth"); }
		}
	};
	build(0, "Decal PSO",       decalPSO,    decalSRB,    decalTexVar,    decalDepthVar);
	build(1, "Decal PSO Add",   decalPSOAdd, decalSRBAdd, decalTexVarAdd, decalDepthVarAdd);
	build(2, "Decal PSO Stain", decalPSOMod, decalSRBMod, decalTexVarMod, decalDepthVarMod);

	// Target-filtered variant: the TARGET MESH re-draws with the projection (position stream
	// only, depth LEQUAL against its own surface) — only that surface can catch the stain.
	decalMeshPSO.Release(); decalMeshPSOAdd.Release(); decalMeshPSOMod.Release();
	decalMeshSRB.Release(); decalMeshSRBAdd.Release(); decalMeshSRBMod.Release();
	decalMeshTexVar = decalMeshTexVarAdd = decalMeshTexVarMod = nullptr;
	std::string mvs = shaderSource("decal_mesh.vs"), mps = shaderSource("decal_mesh.ps");
	if (!mvs.empty() && !mps.empty())
	{
		RefCntAutoPtr<IShader> v2, p2;
		sci.Desc = {"Decal Mesh VS", SHADER_TYPE_VERTEX, true}; sci.Source = mvs.c_str(); CreateShaderCached(sci, &v2);
		sci.Desc = {"Decal Mesh PS", SHADER_TYPE_PIXEL, true};  sci.Source = mps.c_str(); CreateShaderCached(sci, &p2);
		if (v2 && p2)
		{
			ShaderResourceVariableDesc mvars[] = {
				{SHADER_TYPE_PIXEL, "g_DecalTex", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
			};
			ImmutableSamplerDesc mimms[] = {
				{SHADER_TYPE_PIXEL, "g_DecalTex", smp},
			};
			auto buildMesh = [&](int blendMode, const char* nm, RefCntAutoPtr<IPipelineState>& pso,
			                     RefCntAutoPtr<IShaderResourceBinding>& srb, IShaderResourceVariable*& tv)
			{
				GraphicsPipelineStateCreateInfo ci; ci.PSODesc.Name = nm;
				auto& gp = ci.GraphicsPipeline;
				gp.NumRenderTargets = 1; gp.RTVFormats[0] = SceneFmt();
				gp.DSVFormat = TEX_FORMAT_D32_FLOAT;
				gp.PrimitiveTopology = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
				gp.RasterizerDesc.CullMode = CULL_MODE_BACK;
				gp.DepthStencilDesc.DepthEnable = True;          // occluded like the surface itself
				gp.DepthStencilDesc.DepthFunc = COMPARISON_FUNC_LESS_EQUAL;
				gp.DepthStencilDesc.DepthWriteEnable = False;
				gp.SmplDesc.Count = samples;
				auto& rt = gp.BlendDesc.RenderTargets[0];
				rt.BlendEnable = True; rt.BlendOp = BLEND_OPERATION_ADD; rt.BlendOpAlpha = BLEND_OPERATION_ADD;
				if      (blendMode == 1) { rt.SrcBlend = BLEND_FACTOR_ONE;        rt.DestBlend = BLEND_FACTOR_ONE; }
				else if (blendMode == 2) { rt.SrcBlend = BLEND_FACTOR_DEST_COLOR; rt.DestBlend = BLEND_FACTOR_INV_SRC_ALPHA; }
				else                     { rt.SrcBlend = BLEND_FACTOR_ONE;        rt.DestBlend = BLEND_FACTOR_INV_SRC_ALPHA; }
				rt.SrcBlendAlpha = BLEND_FACTOR_ZERO; rt.DestBlendAlpha = BLEND_FACTOR_ONE;
				gp.InputLayout.LayoutElements = layout; gp.InputLayout.NumElements = 1;
				ci.pVS = v2; ci.pPS = p2;
				ci.PSODesc.ResourceLayout.Variables            = mvars; ci.PSODesc.ResourceLayout.NumVariables         = 1;
				ci.PSODesc.ResourceLayout.ImmutableSamplers    = mimms; ci.PSODesc.ResourceLayout.NumImmutableSamplers = 1;
				ci.PSODesc.ResourceLayout.DefaultVariableType  = SHADER_RESOURCE_VARIABLE_TYPE_STATIC;
				CreateGraphicsPipelineStateCached(ci, &pso);
				if (pso)
				{
					if (auto* sv = pso->GetStaticVariableByName(SHADER_TYPE_VERTEX, "DecalCB")) sv->Set(decalCB);
					if (auto* sp = pso->GetStaticVariableByName(SHADER_TYPE_PIXEL,  "DecalCB")) sp->Set(decalCB);
					pso->CreateShaderResourceBinding(&srb, true);
					if (srb) tv = srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_DecalTex");
				}
			};
			buildMesh(0, "Decal Mesh PSO",       decalMeshPSO,    decalMeshSRB,    decalMeshTexVar);
			buildMesh(1, "Decal Mesh PSO Add",   decalMeshPSOAdd, decalMeshSRBAdd, decalMeshTexVarAdd);
			buildMesh(2, "Decal Mesh PSO Stain", decalMeshPSOMod, decalMeshSRBMod, decalMeshTexVarMod);
		}
	}
	std::cout << "[NukeDiligent]\tdecal pipeline" << (decalPSO ? " ready" : " FAILED")
	          << (decalMeshPSO ? " (+mesh-target)" : "") << std::endl;
}

void NukeDiligent::drawDecal(Texture* tex, const float pos[3], const float quat[4], const float scale[3],
                             const float tint[4], float intensity, float angleFade, int mode,
                             float appear, int appearMode)
{
	m_impl->lastInstBind.pso = nullptr;   // decal pipeline replaces the instanced VB/PSO state
	if (!m_impl->decalPSO || !m_impl->decalStamp.current(m_impl->samples, m_impl->SceneFmt()) || !tex) return;
	ITextureView* dtex = m_impl->GetTexSRV(tex);
	if (!dtex || !m_impl->gbufDepthSRV) return;   // needs the depth prepass (engine forces it for decals)

	float4x4 world = float4x4::Scale(scale[0], scale[1], scale[2])
	               * Diligent::Quaternion<float>(quat[0], quat[1], quat[2], quat[3]).ToMatrix()
	               * float4x4::Translation(pos[0], pos[1], pos[2]);
	float4x4 vp = m_impl->curView * m_impl->curProj;

	DecalCBData cb;
	cb.wvp         = world * vp;
	cb.invWorld    = world.Inverse();
	cb.invViewProj = vp.Inverse();
	cb.tint        = float4(tint[0], tint[1], tint[2], tint[3]);
	cb.params      = float4(intensity, angleFade, appear, (float)appearMode);
	// projection axis = the box's local +Z in world (row 2 of the world matrix), normalized.
	float3 axis(world.m[2][0], world.m[2][1], world.m[2][2]);
	float  al = length(axis); if (al < 1e-6f) al = 1.f; axis = axis / al;
	cb.projAxis    = float4(axis.x, axis.y, axis.z, 0.f);
	cb.res         = float4((float)(m_impl->curRTW > 0 ? m_impl->curRTW : 1), (float)(m_impl->curRTH > 0 ? m_impl->curRTH : 1), 0.f, 0.f);
	{ MapHelper<DecalCBData> m(m_impl->context, m_impl->decalCB, MAP_WRITE, MAP_FLAG_DISCARD); *m = cb; }

	IPipelineState*         pso = (mode == 1) ? m_impl->decalPSOAdd      : (mode == 2) ? m_impl->decalPSOMod      : m_impl->decalPSO;
	IShaderResourceBinding* srb = (mode == 1) ? m_impl->decalSRBAdd      : (mode == 2) ? m_impl->decalSRBMod      : m_impl->decalSRB;
	IShaderResourceVariable* tv = (mode == 1) ? m_impl->decalTexVarAdd   : (mode == 2) ? m_impl->decalTexVarMod   : m_impl->decalTexVar;
	IShaderResourceVariable* dv = (mode == 1) ? m_impl->decalDepthVarAdd : (mode == 2) ? m_impl->decalDepthVarMod : m_impl->decalDepthVar;
	if (!pso || !srb) return;
	if (tv) tv->Set(dtex);
	if (dv) dv->Set(m_impl->gbufDepthSRV);

	IDeviceContext* ctx = m_impl->context;
	Uint64 offset = 0; IBuffer* vbs[] = { m_impl->decalVB };
	ctx->SetVertexBuffers(0, 1, vbs, &offset, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, SET_VERTEX_BUFFERS_FLAG_RESET);
	ctx->SetPipelineState(pso);
	ctx->CommitShaderResources(srb, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	DrawAttribs da; da.NumVertices = 36; da.Flags = DRAW_FLAG_VERIFY_ALL;
	ctx->Draw(da);
}

void NukeDiligent::drawDecalMesh(Texture* tex, const float pos[3], const float quat[4], const float scale[3],
                                 const float tint[4], float intensity, float angleFade, int mode,
                                 float appear, int appearMode, Mesh* target,
                                 const float tPos[3], const float tQuat[4], const float tScale[3])
{
	m_impl->lastInstBind.pso = nullptr;   // decal pipeline replaces the instanced VB/PSO state
	if (!target || !tex) return;
	if (!m_impl->decalMeshPSO || !m_impl->decalStamp.current(m_impl->samples, m_impl->SceneFmt())) return;
	ITextureView* dtex = m_impl->GetTexSRV(tex);
	if (!dtex) return;
	Impl::MeshGPU* g = m_impl->GetMeshGPU(target);
	if (!g || !g->PosBuf()) return;

	float4x4 boxWorld = float4x4::Scale(scale[0], scale[1], scale[2])
	                  * Diligent::Quaternion<float>(quat[0], quat[1], quat[2], quat[3]).ToMatrix()
	                  * float4x4::Translation(pos[0], pos[1], pos[2]);
	float4x4 meshWorld = float4x4::Scale(tScale[0], tScale[1], tScale[2])
	                   * Diligent::Quaternion<float>(tQuat[0], tQuat[1], tQuat[2], tQuat[3]).ToMatrix()
	                   * float4x4::Translation(tPos[0], tPos[1], tPos[2]);
	float4x4 vp = m_impl->curView * m_impl->curProj;

	DecalCBData cb;
	cb.wvp         = meshWorld * vp;
	cb.invWorld    = boxWorld.Inverse();
	cb.invViewProj = meshWorld;   // the mesh variant reads this slot as g_MeshWorld
	cb.tint        = float4(tint[0], tint[1], tint[2], tint[3]);
	cb.params      = float4(intensity, angleFade, appear, (float)appearMode);
	float3 axis(boxWorld.m[2][0], boxWorld.m[2][1], boxWorld.m[2][2]);
	float  al = length(axis); if (al < 1e-6f) al = 1.f; axis = axis / al;
	cb.projAxis    = float4(axis.x, axis.y, axis.z, 0.f);
	cb.res         = float4((float)(m_impl->curRTW > 0 ? m_impl->curRTW : 1), (float)(m_impl->curRTH > 0 ? m_impl->curRTH : 1), 0.f, 0.f);
	{ MapHelper<DecalCBData> m(m_impl->context, m_impl->decalCB, MAP_WRITE, MAP_FLAG_DISCARD); *m = cb; }

	IPipelineState*          pso = (mode == 1) ? m_impl->decalMeshPSOAdd    : (mode == 2) ? m_impl->decalMeshPSOMod    : m_impl->decalMeshPSO;
	IShaderResourceBinding*  srb = (mode == 1) ? m_impl->decalMeshSRBAdd    : (mode == 2) ? m_impl->decalMeshSRBMod    : m_impl->decalMeshSRB;
	IShaderResourceVariable* tv  = (mode == 1) ? m_impl->decalMeshTexVarAdd : (mode == 2) ? m_impl->decalMeshTexVarMod : m_impl->decalMeshTexVar;
	if (!pso || !srb) return;
	if (tv) tv->Set(dtex);

	// The target's active LOD range — the projection re-draws exactly what the world drew.
	const int lod = m_impl->SelectLod(target, tPos, tScale);
	uint32_t first = 0, count = 0;
	m_impl->LodRange(target, lod, first, count);
	if (!count) return;

	IDeviceContext* ctx = m_impl->context;
	IBuffer* vbs[] = { g->PosBuf() };
	Uint64   offs[] = { g->PosOfs() };
	ctx->SetVertexBuffers(0, 1, vbs, offs, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, SET_VERTEX_BUFFERS_FLAG_RESET);
	ctx->SetPipelineState(pso);
	ctx->CommitShaderResources(srb, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	if (g->IdxBuf() && g->numIndices > 0)
	{
		ctx->SetIndexBuffer(g->IdxBuf(), g->IdxOfs(), RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		DrawIndexedAttribs da{(Uint32)count, VT_UINT32, DRAW_FLAG_VERIFY_STATES};
		da.FirstIndexLocation = (Uint32)first;
		ctx->DrawIndexed(da);
	}
	else
	{
		DrawAttribs da{(Uint32)count, DRAW_FLAG_VERIFY_STATES};
		da.StartVertexLocation = (Uint32)first;
		ctx->Draw(da);
	}
}
