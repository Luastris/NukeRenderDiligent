#include "NukeDiligentImpl.h"
#include <cstring>

// Sprite pipeline: unlit textured quads (iRender::drawSprite). Alpha-blended, single texture per
// draw, depth-tested but no depth write — drawn IN the camera pass (SceneFmt + MSAA) after opaque
// geometry, so sprites get tonemap/post like everything else. Double-sided (CULL_NONE), so the
// engine's quad winding is irrelevant. Immediate per-sprite draw (batching is a later optimisation).

void NukeDiligent::Impl::CreateSpriteResources()
{
	spritePSO.Release(); spriteSRB.Release(); spriteCB.Release(); spriteTexVar = nullptr;
	std::string vs = shaderSource("sprite.vs"), ps = shaderSource("sprite.ps");
	if (vs.empty() || ps.empty()) { std::cout << "[NukeDiligent]\tsprite shaders missing" << std::endl; return; }

	ShaderCreateInfo sci; sci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
	RefCntAutoPtr<IShader> v, p;
	sci.Desc = {"Sprite VS", SHADER_TYPE_VERTEX, true}; sci.Source = vs.c_str(); device->CreateShader(sci, &v);
	sci.Desc = {"Sprite PS", SHADER_TYPE_PIXEL, true};  sci.Source = ps.c_str(); device->CreateShader(sci, &p);
	if (!v || !p) return;

	BufferDesc cbd; cbd.Name = "SpriteCB"; cbd.Size = sizeof(float4x4);
	cbd.Usage = USAGE_DYNAMIC; cbd.BindFlags = BIND_UNIFORM_BUFFER; cbd.CPUAccessFlags = CPU_ACCESS_WRITE;
	device->CreateBuffer(cbd, nullptr, &spriteCB);
	// The vertex buffer is created/grown on demand in FlushSprites (it survives PSO rebuilds).

	GraphicsPipelineStateCreateInfo ci; ci.PSODesc.Name = "Sprite PSO";
	auto& gp = ci.GraphicsPipeline;
	gp.NumRenderTargets = 1; gp.RTVFormats[0] = SceneFmt();   // composites into the (MS) HDR camera target
	gp.DSVFormat = TEX_FORMAT_D32_FLOAT;
	gp.PrimitiveTopology = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	gp.RasterizerDesc.CullMode = CULL_MODE_NONE;             // double-sided
	gp.DepthStencilDesc.DepthEnable = True;                  // occluded by opaque geometry...
	gp.DepthStencilDesc.DepthWriteEnable = False;            // ...but transparent: no depth write
	gp.SmplDesc.Count = samples;                            // MSAA: match the camera target
	auto& rt = gp.BlendDesc.RenderTargets[0];
	rt.BlendEnable   = True;
	rt.SrcBlend      = BLEND_FACTOR_SRC_ALPHA; rt.DestBlend      = BLEND_FACTOR_INV_SRC_ALPHA; rt.BlendOp      = BLEND_OPERATION_ADD;
	rt.SrcBlendAlpha = BLEND_FACTOR_ONE;       rt.DestBlendAlpha = BLEND_FACTOR_INV_SRC_ALPHA; rt.BlendOpAlpha = BLEND_OPERATION_ADD;

	LayoutElement layout[] = {
		{0, 0, 3, VT_FLOAT32, False},   // pos
		{1, 0, 2, VT_FLOAT32, False},   // uv
		{2, 0, 4, VT_FLOAT32, False},   // tint
	};
	gp.InputLayout.LayoutElements = layout; gp.InputLayout.NumElements = 3;
	ci.pVS = v; ci.pPS = p;

	// g_Sprite: dynamic PS texture + a linear-clamp immutable sampler (combined-sampler convention).
	ShaderResourceVariableDesc vars[] = { {SHADER_TYPE_PIXEL, "g_Sprite", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC} };
	SamplerDesc samp;
	samp.MinFilter = FILTER_TYPE_LINEAR; samp.MagFilter = FILTER_TYPE_LINEAR; samp.MipFilter = FILTER_TYPE_LINEAR;
	samp.AddressU = TEXTURE_ADDRESS_CLAMP; samp.AddressV = TEXTURE_ADDRESS_CLAMP; samp.AddressW = TEXTURE_ADDRESS_CLAMP;
	ImmutableSamplerDesc imms[] = { {SHADER_TYPE_PIXEL, "g_Sprite", samp} };
	ci.PSODesc.ResourceLayout.Variables            = vars; ci.PSODesc.ResourceLayout.NumVariables         = 1;
	ci.PSODesc.ResourceLayout.ImmutableSamplers    = imms; ci.PSODesc.ResourceLayout.NumImmutableSamplers = 1;
	ci.PSODesc.ResourceLayout.DefaultVariableType  = SHADER_RESOURCE_VARIABLE_TYPE_STATIC;

	device->CreateGraphicsPipelineState(ci, &spritePSO);
	if (spritePSO)
	{
		if (auto* sv = spritePSO->GetStaticVariableByName(SHADER_TYPE_VERTEX, "SpriteCB")) sv->Set(spriteCB);
		spritePSO->CreateShaderResourceBinding(&spriteSRB, true);
		if (spriteSRB) spriteTexVar = spriteSRB->GetVariableByName(SHADER_TYPE_PIXEL, "g_Sprite");
	}
	std::cout << "[NukeDiligent]\tsprite pipeline" << (spritePSO ? " ready" : " FAILED") << std::endl;
}

// Accumulate one quad. The batch flushes when the texture changes (a new run) or at endCamera —
// so consecutive same-texture sprites collapse to a single draw call.
void NukeDiligent::drawSprite(Texture* tex, const float center[3], const float right[3], const float up[3],
                              const float uv[4], const float tint[4])
{
	if (!m_impl->spritePSO || !tex) return;
	if (m_impl->spriteBatchTex && tex != m_impl->spriteBatchTex) m_impl->FlushSprites();   // texture changed -> new batch
	m_impl->spriteBatchTex = tex;

	// center + half-extent vectors right/up (already scaled + pivoted engine-side); ±1 = full quad.
	auto push = [&](float sx, float sy, float u, float vv)
	{
		std::vector<float>& b = m_impl->spriteBatchVerts;
		b.push_back(center[0] + sx * right[0] + sy * up[0]);
		b.push_back(center[1] + sx * right[1] + sy * up[1]);
		b.push_back(center[2] + sx * right[2] + sy * up[2]);
		b.push_back(u); b.push_back(vv);
		b.push_back(tint[0]); b.push_back(tint[1]); b.push_back(tint[2]); b.push_back(tint[3]);
	};
	const float u0 = uv[0], v0 = uv[1], u1 = uv[2], v1 = uv[3];
	push(-1.f,  1.f, u0, v0); push( 1.f,  1.f, u1, v0); push( 1.f, -1.f, u1, v1);   // TL, TR, BR
	push(-1.f,  1.f, u0, v0); push( 1.f, -1.f, u1, v1); push(-1.f, -1.f, u0, v1);   // TL, BR, BL
}

// Draw the accumulated batch (one texture) in a single call. Called on a texture change and at
// endCamera (before the MSAA resolve, while the camera targets are still bound).
void NukeDiligent::Impl::FlushSprites()
{
	if (!spritePSO || spriteBatchVerts.empty() || !spriteBatchTex) { spriteBatchVerts.clear(); spriteBatchTex = nullptr; return; }
	ITextureView* srv = GetTexSRV(spriteBatchTex);
	if (!srv) { spriteBatchVerts.clear(); spriteBatchTex = nullptr; return; }

	const int vertCount = (int)(spriteBatchVerts.size() / 9);
	if (!spriteVB || spriteVBSize < vertCount)
	{
		spriteVB.Release();
		while (spriteVBSize < vertCount) spriteVBSize = spriteVBSize ? spriteVBSize * 2 : 384;   // 384 = 64 quads
		BufferDesc bd; bd.Name = "Sprite VB"; bd.BindFlags = BIND_VERTEX_BUFFER;
		bd.Usage = USAGE_DYNAMIC; bd.CPUAccessFlags = CPU_ACCESS_WRITE; bd.Size = (Uint64)spriteVBSize * 9 * sizeof(float);
		device->CreateBuffer(bd, nullptr, &spriteVB);
		if (!spriteVB) { spriteBatchVerts.clear(); spriteBatchTex = nullptr; return; }
	}
	{ MapHelper<float>    mv(context, spriteVB, MAP_WRITE, MAP_FLAG_DISCARD); std::memcpy(mv, spriteBatchVerts.data(), spriteBatchVerts.size() * sizeof(float)); }
	{ MapHelper<float4x4> cb(context, spriteCB, MAP_WRITE, MAP_FLAG_DISCARD); *cb = curView * curProj; }
	if (spriteTexVar) spriteTexVar->Set(srv);

	Uint64 offset = 0; IBuffer* vbs[] = { spriteVB };
	context->SetVertexBuffers(0, 1, vbs, &offset, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, SET_VERTEX_BUFFERS_FLAG_RESET);
	context->SetPipelineState(spritePSO);
	context->CommitShaderResources(spriteSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	DrawAttribs da; da.NumVertices = (Uint32)vertCount; da.Flags = DRAW_FLAG_VERIFY_ALL;
	context->Draw(da);

	spriteBatchVerts.clear();
	spriteBatchTex = nullptr;
}
