#include "NukeDiligentImpl.h"
#include <cstring>

// Sprite pipeline: unlit textured quads (iRender::drawSprite). Alpha-blended, single texture per
// draw, depth-tested but no depth write — drawn IN the camera pass (SceneFmt + MSAA) after opaque
// geometry, so sprites get tonemap/post like everything else. Double-sided (CULL_NONE), so the
// engine's quad winding is irrelevant. Immediate per-sprite draw (batching is a later optimisation).

void NukeDiligent::Impl::CreateSpriteResources()
{
	spritePSO.Release(); spriteSRB.Release(); spriteCB.Release(); spriteTexVar = nullptr;
	spriteScreenPSO.Release(); spriteScreenSRB.Release(); spriteScreenTexVar = nullptr;
	spriteScreenPSOBB.Release(); spriteScreenSRBBB.Release(); spriteScreenTexVarBB = nullptr;
	std::string vs = shaderSource("sprite.vs"), ps = shaderSource("sprite.ps");
	if (vs.empty() || ps.empty()) { std::cout << "[NukeDiligent]\tsprite shaders missing" << std::endl; return; }

	ShaderCreateInfo sci; sci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
	RefCntAutoPtr<IShader> v, p;
	sci.Desc = {"Sprite VS", SHADER_TYPE_VERTEX, true}; sci.Source = vs.c_str(); CreateShaderCached(sci, &v);
	sci.Desc = {"Sprite PS", SHADER_TYPE_PIXEL, true};  sci.Source = ps.c_str(); CreateShaderCached(sci, &p);
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

	CreateGraphicsPipelineStateCached(ci, &spritePSO);
	if (spritePSO)
	{
		if (auto* sv = spritePSO->GetStaticVariableByName(SHADER_TYPE_VERTEX, "SpriteCB")) sv->Set(spriteCB);
		spritePSO->CreateShaderResourceBinding(&spriteSRB, true);
		if (spriteSRB) spriteTexVar = spriteSRB->GetVariableByName(SHADER_TYPE_PIXEL, "g_Sprite");
	}

	// After-post screen variants: same sprite shaders, but output-format RTV, single-sample, NO depth
	// (drawn over the final image). One for the RT post texture (RGBA8), one for the backbuffer.
	auto buildScreen = [&](TEXTURE_FORMAT fmt, const char* nm, RefCntAutoPtr<IPipelineState>& pso,
	                       RefCntAutoPtr<IShaderResourceBinding>& srb, IShaderResourceVariable*& tvar)
	{
		GraphicsPipelineStateCreateInfo si; si.PSODesc.Name = nm;
		auto& g = si.GraphicsPipeline;
		g.NumRenderTargets = 1; g.RTVFormats[0] = fmt;
		g.DSVFormat = TEX_FORMAT_UNKNOWN;
		g.PrimitiveTopology = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		g.RasterizerDesc.CullMode = CULL_MODE_NONE;
		g.DepthStencilDesc.DepthEnable = False;
		g.SmplDesc.Count = 1;
		auto& b = g.BlendDesc.RenderTargets[0];
		b.BlendEnable = True; b.SrcBlend = BLEND_FACTOR_SRC_ALPHA; b.DestBlend = BLEND_FACTOR_INV_SRC_ALPHA; b.BlendOp = BLEND_OPERATION_ADD;
		b.SrcBlendAlpha = BLEND_FACTOR_ONE; b.DestBlendAlpha = BLEND_FACTOR_INV_SRC_ALPHA; b.BlendOpAlpha = BLEND_OPERATION_ADD;
		g.InputLayout.LayoutElements = layout; g.InputLayout.NumElements = 3;
		si.pVS = v; si.pPS = p;
		si.PSODesc.ResourceLayout.Variables            = vars; si.PSODesc.ResourceLayout.NumVariables         = 1;
		si.PSODesc.ResourceLayout.ImmutableSamplers    = imms; si.PSODesc.ResourceLayout.NumImmutableSamplers = 1;
		si.PSODesc.ResourceLayout.DefaultVariableType  = SHADER_RESOURCE_VARIABLE_TYPE_STATIC;
		CreateGraphicsPipelineStateCached(si, &pso);
		if (pso)
		{
			if (auto* sv = pso->GetStaticVariableByName(SHADER_TYPE_VERTEX, "SpriteCB")) sv->Set(spriteCB);
			pso->CreateShaderResourceBinding(&srb, true);
			if (srb) tvar = srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Sprite");
		}
	};
	buildScreen(TEX_FORMAT_RGBA8_UNORM, "Sprite Screen PSO", spriteScreenPSO, spriteScreenSRB, spriteScreenTexVar);
	TEXTURE_FORMAT bbFmt = swapChain ? swapChain->GetDesc().ColorBufferFormat : TEX_FORMAT_RGBA8_UNORM;
	buildScreen(bbFmt, "Sprite Screen PSO BB", spriteScreenPSOBB, spriteScreenSRBBB, spriteScreenTexVarBB);

	// LIT variant (drawSpriteRunLit): sprite_lit shaders — diffuse+normal, Lambert from the
	// shared worldFrameCB, per-batch plane TBN. Same blend/depth/MSAA as the unlit sprite PSO.
	spriteLitPSO.Release(); spriteLitCB.Release(); spriteLitSRBs.clear();
	std::string lvs = shaderSource("sprite_lit.vs"), lps = shaderSource("sprite_lit.ps");
	if (!lvs.empty() && !lps.empty() && worldFrameCB)
	{
		RefCntAutoPtr<IShader> lv, lp;
		sci.Desc = {"SpriteLit VS", SHADER_TYPE_VERTEX, true}; sci.Source = lvs.c_str(); CreateShaderCached(sci, &lv);
		sci.Desc = {"SpriteLit PS", SHADER_TYPE_PIXEL, true};  sci.Source = lps.c_str(); CreateShaderCached(sci, &lp);
		if (lv && lp)
		{
			BufferDesc lcb; lcb.Name = "SpriteLitCB"; lcb.Size = sizeof(float) * 12;
			lcb.Usage = USAGE_DYNAMIC; lcb.BindFlags = BIND_UNIFORM_BUFFER; lcb.CPUAccessFlags = CPU_ACCESS_WRITE;
			device->CreateBuffer(lcb, nullptr, &spriteLitCB);

			GraphicsPipelineStateCreateInfo li; li.PSODesc.Name = "SpriteLit PSO";
			auto& lg = li.GraphicsPipeline;
			lg = gp;   // same targets/blend/depth/MSAA/topology/cull as the unlit sprite PSO
			lg.InputLayout.LayoutElements = layout; lg.InputLayout.NumElements = 3;
			li.pVS = lv; li.pPS = lp;
			ShaderResourceVariableDesc lvars[] = {
				{SHADER_TYPE_PIXEL, "g_Sprite", SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
				{SHADER_TYPE_PIXEL, "g_Normal", SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE} };
			ImmutableSamplerDesc limms[] = { {SHADER_TYPE_PIXEL, "g_Sprite", samp},
			                                 {SHADER_TYPE_PIXEL, "g_Normal", samp} };
			li.PSODesc.ResourceLayout.Variables            = lvars; li.PSODesc.ResourceLayout.NumVariables         = 2;
			li.PSODesc.ResourceLayout.ImmutableSamplers    = limms; li.PSODesc.ResourceLayout.NumImmutableSamplers = 2;
			li.PSODesc.ResourceLayout.DefaultVariableType  = SHADER_RESOURCE_VARIABLE_TYPE_STATIC;
			CreateGraphicsPipelineStateCached(li, &spriteLitPSO);
			if (spriteLitPSO)
			{
				if (auto* sv = spriteLitPSO->GetStaticVariableByName(SHADER_TYPE_VERTEX, "SpriteCB"))     sv->Set(spriteCB);
				if (auto* sv = spriteLitPSO->GetStaticVariableByName(SHADER_TYPE_PIXEL,  "SpriteLitCB"))  sv->Set(spriteLitCB);
				if (auto* sv = spriteLitPSO->GetStaticVariableByName(SHADER_TYPE_PIXEL,  "FrameCB"))      sv->Set(worldFrameCB);
			}
		}
	}

	std::cout << "[NukeDiligent]\tsprite pipeline" << (spritePSO ? " ready" : " FAILED")
	          << (spriteLitPSO ? " (+lit)" : "") << std::endl;
}

// Accumulate one quad. The batch flushes when the texture changes (a new run) or at endCamera —
// so consecutive same-texture sprites collapse to a single draw call.
void NukeDiligent::drawSprite(Texture* tex, const float center[3], const float right[3], const float up[3],
                              const float uv[4], const float tint[4])
{
	if (!m_impl->spritePSO || !tex) return;
	if (!m_impl->cameraPassActive) return;   // no camera targets bound -> nowhere valid to draw (see Impl flag)
	if (m_impl->spriteLitTex) m_impl->FlushSpritesLit();   // kind switch: keep paint order
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

// BULK append (tilemap chunks): pre-baked quads in the batch's exact vertex layout —
// one memcpy instead of thousands of drawSprite calls. Same per-texture run semantics.
void NukeDiligent::drawSpriteRun(Texture* tex, const float* verts, int vertCount)
{
	if (!m_impl->spritePSO || !tex || !verts || vertCount <= 0) return;
	if (!m_impl->cameraPassActive) return;   // no camera targets bound -> nowhere valid to draw
	if (m_impl->spriteLitTex) m_impl->FlushSpritesLit();   // kind switch: keep paint order
	if (m_impl->spriteBatchTex && tex != m_impl->spriteBatchTex) m_impl->FlushSprites();
	m_impl->spriteBatchTex = tex;
	std::vector<float>& b = m_impl->spriteBatchVerts;
	b.insert(b.end(), verts, verts + (size_t)vertCount * 9);
}

// LIT bulk append: same layout, drawn with the normal-mapped Lambert pipeline. Falls back
// to the unlit run when the lit PSO/normal map is unavailable (backends stay honest).
void NukeDiligent::drawSpriteRunLit(Texture* tex, Texture* normal, const float* verts, int vertCount,
                                    bool normalFlipY)
{
	if (!m_impl->spriteLitPSO || !normal) { drawSpriteRun(tex, verts, vertCount); return; }
	if (!tex || !verts || vertCount <= 0) return;
	if (!m_impl->cameraPassActive) return;
	if (m_impl->spriteBatchTex) m_impl->FlushSprites();    // kind switch: keep paint order
	if (m_impl->spriteLitTex && (tex != m_impl->spriteLitTex || normal != m_impl->spriteLitNormal))
		m_impl->FlushSpritesLit();
	m_impl->spriteLitTex = tex; m_impl->spriteLitNormal = normal; m_impl->spriteLitFlipY = normalFlipY;
	std::vector<float>& b = m_impl->spriteLitVerts;
	b.insert(b.end(), verts, verts + (size_t)vertCount * 9);
}

// Draw the accumulated batch (one texture) in a single call. Called on a texture change and at
// endCamera (before the MSAA resolve, while the camera targets are still bound).
void NukeDiligent::Impl::FlushSprites()
{
	if (!spritePSO || spriteBatchVerts.empty() || !spriteBatchTex) { spriteBatchVerts.clear(); spriteBatchTex = nullptr; return; }
	// Outside a camera pass the bound target has no (matching) depth buffer — the sprite PSO needs
	// D32. Drop the batch instead of spamming D3D12 with format-mismatch draws.
	if (!cameraPassActive) { spriteBatchVerts.clear(); spriteBatchTex = nullptr; return; }
	ITextureView* srv = GetTexSRV(spriteBatchTex);
	// Draw-path diagnostics (NUKE_TM_DIAG=1): a dropped batch is invisible art with no error.
	static const bool diag = []{ const char* e = std::getenv("NUKE_TM_DIAG"); return e && *e == '1'; }();
	if (diag)
	{
		static int n = 0;
		if (n < 6)
		{
			++n;
			std::cout << "[NukeDiligent]\tDIAG sprite flush " << spriteBatchVerts.size() / 9
			          << " verts, srv " << (srv ? "ok" : "NULL") << std::endl;
			const std::vector<float>& b = spriteBatchVerts;
			for (size_t v = 0; v + 9 <= b.size() && v < 27; v += 9)
				std::cout << "[NukeDiligent]\tDIAG v" << v / 9 << " pos(" << b[v] << "," << b[v+1] << "," << b[v+2]
				          << ") uv(" << b[v+3] << "," << b[v+4] << ") col(" << b[v+5] << "," << b[v+6]
				          << "," << b[v+7] << "," << b[v+8] << ")" << std::endl;
			if (b.size() >= 9 * 9)
			{
				size_t v = ((b.size() / 9) - 1) * 9;   // last vert (the sprite's, when appended after tiles)
				std::cout << "[NukeDiligent]\tDIAG vLAST pos(" << b[v] << "," << b[v+1] << "," << b[v+2]
				          << ") uv(" << b[v+3] << "," << b[v+4] << ")" << std::endl;
			}
		}
	}
	if (!srv) { spriteBatchVerts.clear(); spriteBatchTex = nullptr; return; }

	const int vertCount = (int)(spriteBatchVerts.size() / 9);
	if (!spriteVB || spriteVBSize < vertCount)
	{
		Trash(spriteVB);   // grows mid-frame; earlier draws this frame reference the old buffer
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

// Draw the accumulated LIT batch (one diffuse+normal pair). Plane TBN comes from the first
// quad's corners (a run shares one plane: TL,TR,BR — T = TL->TR, B = BR->TR ⊥ish, N = T×B).
void NukeDiligent::Impl::FlushSpritesLit()
{
	if (spriteLitVerts.empty() || !spriteLitTex) { spriteLitVerts.clear(); spriteLitTex = nullptr; spriteLitNormal = nullptr; return; }
	auto drop = [&]{ spriteLitVerts.clear(); spriteLitTex = nullptr; spriteLitNormal = nullptr; };
	if (!spriteLitPSO || !cameraPassActive) { drop(); return; }
	ITextureView* srv  = GetTexSRV(spriteLitTex);
	ITextureView* nsrv = GetTexSRV(spriteLitNormal);
	if (!srv || !nsrv) { drop(); return; }

	// Per-batch TBN from the first quad (verts: TL, TR, BR, ... — 9 floats each).
	{
		const float* v = spriteLitVerts.data();
		float3 tl(v[0], v[1], v[2]), tr(v[9], v[10], v[11]), br(v[18], v[19], v[20]);
		float3 T = tr - tl, B = tr - br;
		float tl2 = length(T), bl2 = length(B);
		T = (tl2 > 1e-6f) ? T / tl2 : float3(1, 0, 0);
		B = (bl2 > 1e-6f) ? B / bl2 : float3(0, 1, 0);
		float3 N = cross(T, B);
		float nl = length(N); N = (nl > 1e-6f) ? N / nl : float3(0, 0, 1);
		MapHelper<float> cb(context, spriteLitCB, MAP_WRITE, MAP_FLAG_DISCARD);
		if (cb)
		{
			float* d = cb;
			d[0] = T.x; d[1] = T.y; d[2]  = T.z; d[3]  = 0;
			d[4] = B.x; d[5] = B.y; d[6]  = B.z; d[7]  = 0;
			d[8] = N.x; d[9] = N.y; d[10] = N.z; d[11] = spriteLitFlipY ? 1.0f : -1.0f;
		}
	}

	const int vertCount = (int)(spriteLitVerts.size() / 9);
	if (!spriteVB || spriteVBSize < vertCount)   // shared VB with the unlit flush (sequential use)
	{
		Trash(spriteVB);
		spriteVB.Release();
		while (spriteVBSize < vertCount) spriteVBSize = spriteVBSize ? spriteVBSize * 2 : 384;
		BufferDesc bd; bd.Name = "Sprite VB"; bd.BindFlags = BIND_VERTEX_BUFFER;
		bd.Usage = USAGE_DYNAMIC; bd.CPUAccessFlags = CPU_ACCESS_WRITE; bd.Size = (Uint64)spriteVBSize * 9 * sizeof(float);
		device->CreateBuffer(bd, nullptr, &spriteVB);
		if (!spriteVB) { drop(); return; }
	}
	{ MapHelper<float>    mv(context, spriteVB, MAP_WRITE, MAP_FLAG_DISCARD); std::memcpy(mv, spriteLitVerts.data(), spriteLitVerts.size() * sizeof(float)); }
	{ MapHelper<float4x4> cb(context, spriteCB, MAP_WRITE, MAP_FLAG_DISCARD); *cb = curView * curProj; }

	// SRB per (diffuse, normal) pair — MUTABLE vars set once (no dynamic-descriptor churn).
	RefCntAutoPtr<IShaderResourceBinding>& srb = spriteLitSRBs[{srv, nsrv}];
	if (!srb)
	{
		spriteLitPSO->CreateShaderResourceBinding(&srb, true);
		if (!srb) { spriteLitSRBs.erase({srv, nsrv}); drop(); return; }
		if (auto* v = srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Sprite")) v->Set(srv);
		if (auto* v = srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Normal")) v->Set(nsrv);
	}

	Uint64 offset = 0; IBuffer* vbs[] = { spriteVB };
	context->SetVertexBuffers(0, 1, vbs, &offset, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, SET_VERTEX_BUFFERS_FLAG_RESET);
	context->SetPipelineState(spriteLitPSO);
	context->CommitShaderResources(srb, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	DrawAttribs da; da.NumVertices = (Uint32)vertCount; da.Flags = DRAW_FLAG_VERIFY_ALL;
	context->Draw(da);
	drop();
}

// ---- screen-space (Canvas HUD) sprites --------------------------------------------------------

void NukeDiligent::drawSpriteScreen(Texture* tex, const float rect[4], const float refSize[2],
                                    const float uv[4], const float tint[4], int afterPost)
{
	if (!tex) return;
	drawSpriteScreenEx(tex, rect, refSize, uv, tint, afterPost, 0);
}

void NukeDiligent::drawSpriteScreenEx(Texture* tex, const float rect[4], const float refSize[2],
                                      const float uv[4], const float tint[4], int afterPost, int scaleMode)
{
	if (!tex) return;
	if (!m_impl->cameraPassActive) return;   // canvas sprites need the camera targets (see Impl flag)
	if (afterPost) m_impl->AppendScreenSprite(m_impl->spriteScrPostVerts, m_impl->spriteScrPostRuns, tex, rect, refSize, uv, tint, scaleMode);
	else           m_impl->AppendScreenSprite(m_impl->spriteScrPreVerts,  m_impl->spriteScrPreRuns,  tex, rect, refSize, uv, tint, scaleMode);
}

// Build one NDC quad from a reference-pixel rect (centre + size), mapped to the current target per
// the canvas scale mode, and append it (+ a per-texture run marker) to a screen batch.
void NukeDiligent::Impl::AppendScreenSprite(std::vector<float>& verts, std::vector<SprRun>& runs, Texture* tex,
                                            const float rect[4], const float refSize[2], const float uv[4], const float tint[4],
                                            int scaleMode)
{
	const float tw = (float)(curRTW > 0 ? curRTW : 1), th = (float)(curRTH > 0 ? curRTH : 1);
	const float refW = refSize[0] > 1.f ? refSize[0] : 1.f, refH = refSize[1] > 1.f ? refSize[1] : 1.f;
	const float sw = tw / refW, sh = th / refH;   // per-axis reference->target scale
	float sx, sy;
	switch (scaleMode)
	{
		default:
		case 0: sx = sy = (sh < sw ? sh : sw); break;   // Fit: uniform, whole canvas visible (letterboxed)
		case 1: sx = sw; sy = sh;              break;   // Stretch: canvas corners = screen corners
		case 2: sx = sy = (sh > sw ? sh : sw); break;   // Expand: uniform, covers the screen (may crop)
		case 3: sx = sy = sw;                  break;   // FitWidth
		case 4: sx = sy = sh;                  break;   // FitHeight
	}
	const float cx = rect[0] * sx / (tw * 0.5f), cy = rect[1] * sy / (th * 0.5f);
	const float hw = rect[2] * 0.5f * sx / (tw * 0.5f), hh = rect[3] * 0.5f * sy / (th * 0.5f);
	const float u0 = uv[0], v0 = uv[1], u1 = uv[2], v1 = uv[3];
	auto push = [&](float x, float y, float u, float vv)
	{
		verts.push_back(x); verts.push_back(y); verts.push_back(0.0f);
		verts.push_back(u); verts.push_back(vv);
		verts.push_back(tint[0]); verts.push_back(tint[1]); verts.push_back(tint[2]); verts.push_back(tint[3]);
	};
	push(cx - hw, cy + hh, u0, v0); push(cx + hw, cy + hh, u1, v0); push(cx + hw, cy - hh, u1, v1);   // TL,TR,BR
	push(cx - hw, cy + hh, u0, v0); push(cx + hw, cy - hh, u1, v1); push(cx - hw, cy - hh, u0, v1);   // TL,BR,BL
	if (runs.empty() || runs.back().tex != tex) runs.push_back({ tex, 0 });
	runs.back().count += 6;
}

// Replay a screen batch: identity transform (verts are already NDC), one draw per texture run.
void NukeDiligent::Impl::FlushScreen(std::vector<float>& verts, std::vector<SprRun>& runs, IPipelineState* pso,
                                     IShaderResourceBinding* srb, IShaderResourceVariable* texVar)
{
	if (!pso || verts.empty() || runs.empty()) { verts.clear(); runs.clear(); return; }
	const int vertCount = (int)(verts.size() / 9);
	if (!spriteVB || spriteVBSize < vertCount)
	{
		Trash(spriteVB);   // grows mid-frame; earlier draws this frame reference the old buffer
		spriteVB.Release();
		while (spriteVBSize < vertCount) spriteVBSize = spriteVBSize ? spriteVBSize * 2 : 384;
		BufferDesc bd; bd.Name = "Sprite VB"; bd.BindFlags = BIND_VERTEX_BUFFER;
		bd.Usage = USAGE_DYNAMIC; bd.CPUAccessFlags = CPU_ACCESS_WRITE; bd.Size = (Uint64)spriteVBSize * 9 * sizeof(float);
		device->CreateBuffer(bd, nullptr, &spriteVB);
		if (!spriteVB) { verts.clear(); runs.clear(); return; }
	}
	{ MapHelper<float>    mv(context, spriteVB, MAP_WRITE, MAP_FLAG_DISCARD); std::memcpy(mv, verts.data(), verts.size() * sizeof(float)); }
	{ MapHelper<float4x4> cb(context, spriteCB, MAP_WRITE, MAP_FLAG_DISCARD); *cb = float4x4::Identity(); }
	Uint64 offset = 0; IBuffer* vbs[] = { spriteVB };
	context->SetVertexBuffers(0, 1, vbs, &offset, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, SET_VERTEX_BUFFERS_FLAG_RESET);
	context->SetPipelineState(pso);
	int base = 0;
	for (const SprRun& r : runs)
	{
		ITextureView* srv = r.tex ? GetTexSRV(r.tex) : nullptr;
		if (srv && texVar)
		{
			texVar->Set(srv);
			context->CommitShaderResources(srb, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
			DrawAttribs da; da.NumVertices = (Uint32)r.count; da.StartVertexLocation = (Uint32)base; da.Flags = DRAW_FLAG_VERIFY_ALL;
			context->Draw(da);
		}
		base += r.count;
	}
	verts.clear(); runs.clear();
}

// Before-post screen sprites reuse the in-scene sprite PSO (NDC z=0 => drawn on top of the scene).
void NukeDiligent::Impl::FlushScreenPre()
{
	// Pre-post canvas sprites reuse the depth-tested sprite PSO -> camera targets required.
	if (!cameraPassActive) { spriteScrPreVerts.clear(); spriteScrPreRuns.clear(); return; }
	FlushScreen(spriteScrPreVerts, spriteScrPreRuns, spritePSO, spriteSRB, spriteTexVar);
}

void NukeDiligent::Impl::FlushScreenPost(bool toBackbuffer)
{
	if (toBackbuffer) FlushScreen(spriteScrPostVerts, spriteScrPostRuns, spriteScreenPSOBB, spriteScreenSRBBB, spriteScreenTexVarBB);
	else              FlushScreen(spriteScrPostVerts, spriteScrPostRuns, spriteScreenPSO,   spriteScreenSRB,   spriteScreenTexVar);
}
