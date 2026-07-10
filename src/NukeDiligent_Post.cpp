#include "NukeDiligentImpl.h"


// Final post-process pipeline: fullscreen pass that tonemaps the HDR scene into an LDR target.
// After swap-chain creation: if the monitor is in HDR mode, switch the swap chain to the HDR10 (PQ, Rec2020)
// colour space so the backbuffer drives the display as real HDR. Falls back to plain (SDR) on failure.
void NukeDiligent::Impl::SetupHDROutput()
{
	hdr10Active = false;
#ifdef _WIN32   // DXGI HDR10 path — Windows (D3D11 or D3D12)
	IDXGISwapChain* dxgi = nullptr;
	if (useD3D12)
	{
		RefCntAutoPtr<ISwapChainD3D12> sc12(swapChain.RawPtr(), IID_SwapChainD3D12);
		if (!sc12) return;
		dxgi = sc12->GetDXGISwapChain();
	}
	else
	{
		RefCntAutoPtr<ISwapChainD3D11> sc11(swapChain.RawPtr(), IID_SwapChainD3D11);
		if (!sc11) return;
		dxgi = sc11->GetDXGISwapChain();
	}
	if (!dxgi) return;
	IDXGISwapChain3* sc3 = nullptr;
	if (FAILED(dxgi->QueryInterface(__uuidof(IDXGISwapChain3), (void**)&sc3)) || !sc3) return;

	bool displayHDR = false;
	IDXGIOutput* out = nullptr;
	if (SUCCEEDED(sc3->GetContainingOutput(&out)) && out)
	{
		IDXGIOutput6* out6 = nullptr;
		if (SUCCEEDED(out->QueryInterface(__uuidof(IDXGIOutput6), (void**)&out6)) && out6)
		{
			DXGI_OUTPUT_DESC1 od{};
			if (SUCCEEDED(out6->GetDesc1(&od)))
				displayHDR = (od.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020);
			out6->Release();
		}
		out->Release();
	}

	if (displayHDR)
	{
		UINT support = 0;
		if (SUCCEEDED(sc3->CheckColorSpaceSupport(DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020, &support)) &&
		    (support & DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT) &&
		    SUCCEEDED(sc3->SetColorSpace1(DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020)))
		{
			hdr10Active = true;
			IDXGISwapChain4* sc4 = nullptr;
			if (SUCCEEDED(sc3->QueryInterface(__uuidof(IDXGISwapChain4), (void**)&sc4)) && sc4)
			{
				DXGI_HDR_METADATA_HDR10 md{};   // Rec2020 primaries + D65 white, 1000-nit peak (chromaticity * 50000)
				md.RedPrimary[0]   = 34000; md.RedPrimary[1]   = 16000;
				md.GreenPrimary[0] = 13250; md.GreenPrimary[1] = 34500;
				md.BluePrimary[0]  =  7500; md.BluePrimary[1]  =  3000;
				md.WhitePoint[0]   = 15635; md.WhitePoint[1]   = 16450;
				md.MaxMasteringLuminance = 1000 * 10000; md.MinMasteringLuminance = 1;
				md.MaxContentLightLevel = 1000; md.MaxFrameAverageLightLevel = 400;
				sc4->SetHDRMetaData(DXGI_HDR_METADATA_TYPE_HDR10, sizeof(md), &md);
				sc4->Release();
			}
		}
	}
	sc3->Release();
	cout << "[NukeDiligent]\tHDR10 output " << (hdr10Active ? "ACTIVE" : "off (display not in HDR mode / unsupported)") << endl;
#endif   // _WIN32
}

void NukeDiligent::Impl::CreatePostResources()
{
	postPSO.Release(); postSRB.Release(); postPSOBB.Release(); postSRBBB.Release(); postCB.Release();   // rebuild-safe
	std::string vs = shaderSource("post.vs"), ps = shaderSource("post.ps");
	if (vs.empty() || ps.empty()) { cout << "[NukeDiligent]\tpost shaders missing" << endl; return; }
	ShaderCreateInfo sci; sci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
	RefCntAutoPtr<IShader> v, p;
	sci.Desc = {"Post VS", SHADER_TYPE_VERTEX, true}; sci.Source = vs.c_str(); device->CreateShader(sci, &v);
	sci.Desc = {"Post PS", SHADER_TYPE_PIXEL, true};  sci.Source = ps.c_str(); device->CreateShader(sci, &p);
	if (!v || !p) return;

	BufferDesc cbd; cbd.Name = "PostCB"; cbd.Size = sizeof(float) * 8; cbd.Usage = USAGE_DYNAMIC;   // g_Post + g_Grade
	cbd.BindFlags = BIND_UNIFORM_BUFFER; cbd.CPUAccessFlags = CPU_ACCESS_WRITE;
	device->CreateBuffer(cbd, nullptr, &postCB);
	// Shared cbuffers for custom post-effect pipelines: PostParams (per-effect params, packed by the engine)
	// and PostFrame (resolution / time, for blur kernels & animated effects).
	if (!postParamsCB) { BufferDesc d; d.Name = "PostParams"; d.Size = 256; d.Usage = USAGE_DYNAMIC; d.BindFlags = BIND_UNIFORM_BUFFER; d.CPUAccessFlags = CPU_ACCESS_WRITE; device->CreateBuffer(d, nullptr, &postParamsCB); }
	if (!postFrameCB)  { BufferDesc d; d.Name = "PostFrame";  d.Size = sizeof(float) * 8; d.Usage = USAGE_DYNAMIC; d.BindFlags = BIND_UNIFORM_BUFFER; d.CPUAccessFlags = CPU_ACCESS_WRITE; device->CreateBuffer(d, nullptr, &postFrameCB); }
	// SSR matrices (view/proj/invProj + resolution), filled per camera in RunSSR.
	if (!ssrCB) { BufferDesc d; d.Name = "SSRCB"; d.Size = sizeof(float) * (16 * 4 + 4); d.Usage = USAGE_DYNAMIC; d.BindFlags = BIND_UNIFORM_BUFFER; d.CPUAccessFlags = CPU_ACCESS_WRITE; device->CreateBuffer(d, nullptr, &ssrCB); }   // view/proj/invProj/invView + res
	if (!rtRefCB) { BufferDesc d; d.Name = "RTRefCB"; d.Size = sizeof(float) * (16 + 4 * 8); d.Usage = USAGE_DYNAMIC; d.BindFlags = BIND_UNIFORM_BUFFER; d.CPUAccessFlags = CPU_ACCESS_WRITE; device->CreateBuffer(d, nullptr, &rtRefCB); }
	if (!taaCB) { BufferDesc d; d.Name = "TAACB"; d.Size = sizeof(float) * (16 * 4 + 4 * 2); d.Usage = USAGE_DYNAMIC; d.BindFlags = BIND_UNIFORM_BUFFER; d.CPUAccessFlags = CPU_ACCESS_WRITE; device->CreateBuffer(d, nullptr, &taaCB); }
	BuildGBufferPipe();   // gbuffer.ps + world.vs (shares worldCB/worldMatCB) — for the SSR prepass

	// Build one post PSO per output format: RGBA8 for RT targets, swap-chain format for the backbuffer.
	auto buildPost = [&](TEXTURE_FORMAT fmt, const char* name,
	                     RefCntAutoPtr<IPipelineState>& pso, RefCntAutoPtr<IShaderResourceBinding>& srb, IShaderResourceVariable*& var)
	{
		GraphicsPipelineStateCreateInfo ci; ci.PSODesc.Name = name;
		auto& gp = ci.GraphicsPipeline;
		gp.NumRenderTargets = 1; gp.RTVFormats[0] = fmt;
		gp.DSVFormat = TEX_FORMAT_UNKNOWN;
		gp.PrimitiveTopology = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		gp.RasterizerDesc.CullMode = CULL_MODE_NONE;
		gp.DepthStencilDesc.DepthEnable = False;
		gp.InputLayout.NumElements = 0;
		ShaderResourceVariableDesc vars[] = {{SHADER_TYPE_PIXEL, "g_HDR", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC}};
		ci.PSODesc.ResourceLayout.Variables = vars; ci.PSODesc.ResourceLayout.NumVariables = 1;
		SamplerDesc samp; samp.MinFilter = FILTER_TYPE_LINEAR; samp.MagFilter = FILTER_TYPE_LINEAR; samp.MipFilter = FILTER_TYPE_LINEAR;
		samp.AddressU = TEXTURE_ADDRESS_CLAMP; samp.AddressV = TEXTURE_ADDRESS_CLAMP;
		ImmutableSamplerDesc imm[] = {{SHADER_TYPE_PIXEL, "g_HDR", samp}};
		ci.PSODesc.ResourceLayout.ImmutableSamplers = imm; ci.PSODesc.ResourceLayout.NumImmutableSamplers = 1;
		ci.pVS = v; ci.pPS = p;
		device->CreateGraphicsPipelineState(ci, &pso);
		if (pso)
		{
			if (auto* c = pso->GetStaticVariableByName(SHADER_TYPE_PIXEL, "PostCB")) c->Set(postCB);
			pso->CreateShaderResourceBinding(&srb, true);
			var = srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_HDR");
		}
	};
	buildPost(TEX_FORMAT_RGBA8_UNORM, "Post PSO", postPSO, postSRB, postHdrVar);
	TEXTURE_FORMAT bbFmt = swapChain ? swapChain->GetDesc().ColorBufferFormat : TEX_FORMAT_RGBA8_UNORM;
	buildPost(bbFmt, "Post PSO BB", postPSOBB, postSRBBB, postHdrVarBB);

	// --- Built-in bloom pipelines (bright-pass / blur / composite), all fullscreen HDR ----------------
	bloomBrightPSO.Release(); bloomBlurPSO.Release(); bloomCompPSO.Release(); bloomCB.Release();
	if (!bloomCB) { BufferDesc d; d.Name = "BloomCB"; d.Size = sizeof(float) * 8; d.Usage = USAGE_DYNAMIC; d.BindFlags = BIND_UNIFORM_BUFFER; d.CPUAccessFlags = CPU_ACCESS_WRITE; device->CreateBuffer(d, nullptr, &bloomCB); }
	auto bloomPSO = [&](const char* psName, const char* dbg, bool twoTex,
	                    RefCntAutoPtr<IPipelineState>& pso, RefCntAutoPtr<IShaderResourceBinding>& srb,
	                    IShaderResourceVariable*& srcV, IShaderResourceVariable** bloomV)
	{
		std::string ps = shaderSource(psName);
		if (vs.empty() || ps.empty()) { cout << "[NukeDiligent]\tbloom shader missing: " << psName << endl; return; }
		ShaderCreateInfo s; s.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
		RefCntAutoPtr<IShader> vv, pp; s.Desc = {dbg, SHADER_TYPE_VERTEX, true}; s.Source = vs.c_str(); device->CreateShader(s, &vv);
		s.Desc = {dbg, SHADER_TYPE_PIXEL, true}; s.Source = ps.c_str(); device->CreateShader(s, &pp);
		if (!vv || !pp) return;
		GraphicsPipelineStateCreateInfo ci; ci.PSODesc.Name = dbg;
		auto& gp = ci.GraphicsPipeline;
		gp.NumRenderTargets = 1; gp.RTVFormats[0] = HDR_FMT; gp.DSVFormat = TEX_FORMAT_UNKNOWN;
		gp.PrimitiveTopology = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; gp.RasterizerDesc.CullMode = CULL_MODE_NONE;
		gp.DepthStencilDesc.DepthEnable = False; gp.InputLayout.NumElements = 0;
		std::vector<ShaderResourceVariableDesc> vars = {{SHADER_TYPE_PIXEL, "g_Source", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC}};
		std::vector<ImmutableSamplerDesc> imms;
		SamplerDesc sm; sm.MinFilter = FILTER_TYPE_LINEAR; sm.MagFilter = FILTER_TYPE_LINEAR; sm.MipFilter = FILTER_TYPE_LINEAR; sm.AddressU = TEXTURE_ADDRESS_CLAMP; sm.AddressV = TEXTURE_ADDRESS_CLAMP;
		imms.push_back({SHADER_TYPE_PIXEL, "g_Source", sm});
		if (twoTex) { vars.push_back({SHADER_TYPE_PIXEL, "g_Bloom", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC}); imms.push_back({SHADER_TYPE_PIXEL, "g_Bloom", sm}); }
		ci.PSODesc.ResourceLayout.Variables = vars.data(); ci.PSODesc.ResourceLayout.NumVariables = (Uint32)vars.size();
		ci.PSODesc.ResourceLayout.ImmutableSamplers = imms.data(); ci.PSODesc.ResourceLayout.NumImmutableSamplers = (Uint32)imms.size();
		ci.pVS = vv; ci.pPS = pp;
		device->CreateGraphicsPipelineState(ci, &pso);
		if (!pso) { cout << "[NukeDiligent]\tbloom PSO failed: " << dbg << endl; return; }
		if (auto* c = pso->GetStaticVariableByName(SHADER_TYPE_PIXEL, "BloomCB")) c->Set(bloomCB);
		pso->CreateShaderResourceBinding(&srb, true);
		srcV = srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Source");
		if (bloomV) *bloomV = srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Bloom");
	};
	bloomPSO("bloom_bright.ps", "Bloom Bright", false, bloomBrightPSO, bloomBrightSRB, bbSrc, nullptr);
	bloomPSO("bloom_blur.ps",   "Bloom Blur",   false, bloomBlurPSO,   bloomBlurSRB,   blSrc, nullptr);
	bloomPSO("bloom_comp.ps",   "Bloom Comp",   true,  bloomCompPSO,   bloomCompSRB,   bcSrc, &bcBloom);
}

// Tonemap HDR -> output into dstRTV. toBackbuffer picks the backbuffer PSO + (when HDR10 is live) PQ encoding.
void NukeDiligent::Impl::RunPostPass(ITextureView* hdrSRV, ITextureView* dstRTV, int w, int h, bool toBackbuffer)
{
	IPipelineState*          pso = toBackbuffer ? postPSOBB : postPSO;
	IShaderResourceBinding*  srb = toBackbuffer ? postSRBBB : postSRB;
	IShaderResourceVariable* var = toBackbuffer ? postHdrVarBB : postHdrVar;
	if (!pso || !srb || !hdrSRV || !dstRTV) return;
	const float mode = !hdr ? 0.0f : ((toBackbuffer && hdr10Active) ? 2.0f : 1.0f);   // 0=passthrough,1=sRGB SDR,2=HDR10 PQ
	{
		MapHelper<float> cb(context, postCB, MAP_WRITE, MAP_FLAG_DISCARD);   // final pass = tonemap/encode ONLY
		for (int k = 0; k < 8; ++k) cb[k] = 0.0f;
		cb[1] = mode;
		cb[2] = hdrPaperWhite; cb[3] = hdrPeak;   // HDR10 mapping (post.ps mode 2)
		cb[4] = toneExposure;   // g_Grade.x = exposure
		cb[5] = toneWhite;      // g_Grade.y = tonemap white point (linear value -> pure white). TODO: expose in World Settings
	}
	context->SetRenderTargets(1, &dstRTV, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	Viewport vp; vp.TopLeftX = 0; vp.TopLeftY = 0; vp.Width = (float)w; vp.Height = (float)h; vp.MinDepth = 0; vp.MaxDepth = 1;
	context->SetViewports(1, &vp, w, h);
	if (var) var->Set(hdrSRV);
	context->SetPipelineState(pso);
	context->CommitShaderResources(srb, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	DrawAttribs da{3, DRAW_FLAG_VERIFY_STATES};
	context->Draw(da);
}

// Build a custom post-effect pipeline from a fullscreen PS (samples g_Source, params in PostParams).
uint64_t NukeDiligent::Impl::CreatePostPipe(const std::string& name, const std::string& ps)
{
	if (name == "rtreflect")   // built-in RT reflections: a real ray-tracing pipeline (rt_rgen/rmiss/rchit + SBT), not a post PS
	{
		if (!rtSupported || !BuildRTPipeline()) return 0;
		PostPipe pp; pp.isRTRef = true;
		uint64_t h = nextShaderHandle++; postPipes[h] = std::move(pp);
		return h;
	}
	std::string vs = shaderSource("post.vs");
	if (vs.empty() || ps.empty()) return 0;
	ShaderCreateInfo sci; sci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
	RefCntAutoPtr<IShader> v, p;
	sci.Desc = {"Post Effect VS", SHADER_TYPE_VERTEX, true}; sci.Source = vs.c_str(); device->CreateShader(sci, &v);
	sci.Desc = {"Post Effect PS", SHADER_TYPE_PIXEL, true};  sci.Source = ps.c_str(); device->CreateShader(sci, &p);
	if (!v || !p) return 0;

	GraphicsPipelineStateCreateInfo ci; ci.PSODesc.Name = "Post Effect PSO";
	auto& gp = ci.GraphicsPipeline;
	gp.NumRenderTargets = 1; gp.RTVFormats[0] = HDR_FMT;   // the chain runs in HDR, single-sample scratch
	gp.DSVFormat = TEX_FORMAT_UNKNOWN;
	gp.PrimitiveTopology = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	gp.RasterizerDesc.CullMode = CULL_MODE_NONE;
	gp.DepthStencilDesc.DepthEnable = False;
	gp.InputLayout.NumElements = 0;
	// "musicvis" (audio-reactive ghost overlays) shares SSR's resource layout: G-buffer
	// normals + prepass depth + camera matrices (SSRCB), PLUS the generic per-object id
	// target. Same isSSR path end to end — the difference is in the pixel shader.
	const bool mvis = (name == "musicvis");
	const bool ssr = (name == "ssr" || mvis);   // built-in: also samples the G-buffer + depth + camera matrices (SSRCB)
	const bool taa = (name == "taa");   // built-in: samples depth + a history texture + camera matrices (TAACB)
	SamplerDesc samp; samp.MinFilter = FILTER_TYPE_LINEAR; samp.MagFilter = FILTER_TYPE_LINEAR; samp.MipFilter = FILTER_TYPE_LINEAR;
	samp.AddressU = TEXTURE_ADDRESS_CLAMP; samp.AddressV = TEXTURE_ADDRESS_CLAMP;
	// POINT sampler for depth + G-buffer: linear filtering of depth/normal across a curved surface (sphere) blends
	// neighbouring values -> wrong reconstructed position (drifts INSIDE the sphere -> self-intersection). Flat
	// surfaces (mirror) have ~constant depth so linear looked fine. Point sampling = exact per-pixel value.
	SamplerDesc psamp; psamp.MinFilter = FILTER_TYPE_POINT; psamp.MagFilter = FILTER_TYPE_POINT; psamp.MipFilter = FILTER_TYPE_POINT;
	psamp.AddressU = TEXTURE_ADDRESS_CLAMP; psamp.AddressV = TEXTURE_ADDRESS_CLAMP;
	std::vector<ShaderResourceVariableDesc> vars = {{SHADER_TYPE_PIXEL, "g_Source", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC}};
	std::vector<ImmutableSamplerDesc>       imms = {{SHADER_TYPE_PIXEL, "g_Source", samp}};
	if (ssr)
	{
		vars.push_back({SHADER_TYPE_PIXEL, "g_GBuffer", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC});
		vars.push_back({SHADER_TYPE_PIXEL, "g_Depth",   SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC});
		imms.push_back({SHADER_TYPE_PIXEL, "g_GBuffer", psamp});
		imms.push_back({SHADER_TYPE_PIXEL, "g_Depth",   psamp});
	}
	if (mvis)   // generic per-object id (gbuffer RT2; POINT — ids must not blend)
	{
		vars.push_back({SHADER_TYPE_PIXEL, "g_ObjId", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC});
		imms.push_back({SHADER_TYPE_PIXEL, "g_ObjId", psamp});
	}
	if (taa)
	{
		vars.push_back({SHADER_TYPE_PIXEL, "g_Depth",    SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC});
		vars.push_back({SHADER_TYPE_PIXEL, "g_History",  SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC});
		vars.push_back({SHADER_TYPE_PIXEL, "g_Velocity", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC});
		imms.push_back({SHADER_TYPE_PIXEL, "g_Depth",    psamp});   // point: exact depth for reprojection
		imms.push_back({SHADER_TYPE_PIXEL, "g_History",  samp});    // linear: bilinear history sampling
		imms.push_back({SHADER_TYPE_PIXEL, "g_Velocity", psamp});   // point: exact motion vector
	}
	ci.PSODesc.ResourceLayout.Variables = vars.data(); ci.PSODesc.ResourceLayout.NumVariables = (Uint32)vars.size();
	ci.PSODesc.ResourceLayout.ImmutableSamplers = imms.data(); ci.PSODesc.ResourceLayout.NumImmutableSamplers = (Uint32)imms.size();
	ci.pVS = v; ci.pPS = p;
	PostPipe pp;
	device->CreateGraphicsPipelineState(ci, &pp.pso);
	if (!pp.pso) { cout << "[NukeDiligent]\tpost effect PSO build failed" << endl; return 0; }
	if (auto* c = pp.pso->GetStaticVariableByName(SHADER_TYPE_PIXEL, "PostParams")) c->Set(postParamsCB);
	if (auto* f = pp.pso->GetStaticVariableByName(SHADER_TYPE_PIXEL, "PostFrame"))  f->Set(postFrameCB);
	if (ssr) if (auto* s = pp.pso->GetStaticVariableByName(SHADER_TYPE_PIXEL, "SSRCB")) s->Set(ssrCB);
	if (taa) if (auto* s = pp.pso->GetStaticVariableByName(SHADER_TYPE_PIXEL, "TAACB")) s->Set(taaCB);
	pp.pso->CreateShaderResourceBinding(&pp.srb, true);
	pp.srcVar = pp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Source");
	if (ssr) { pp.gbufVar = pp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_GBuffer"); pp.depthVar = pp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Depth"); pp.isSSR = true; }
	if (mvis) pp.objIdVar = pp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_ObjId");
	if (taa) { pp.depthVar = pp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Depth"); pp.histVar = pp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_History"); pp.velVar = pp.srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Velocity"); pp.isTAA = true; }
	pp.isBloom = (name == "bloom");   // built-in multi-pass effect; the renderer runs the passes itself
	uint64_t h = nextShaderHandle++;
	postPipes[h] = std::move(pp);
	return h;
}

// Equal formats -> plain CopyTexture. Different formats -> fullscreen blit: D3D12 only
// allows copies inside a format family, and the chain legitimately crosses families
// (RGBA8 scene color with HDR off vs the RGBA16F chain scratch). One tiny PSO per
// destination format, built lazily.
void NukeDiligent::Impl::BlitTexture(ITextureView* srcSRV, ITexture* dstTex)
{
	if (!srcSRV || !dstTex) return;
	ITexture* srcTex = srcSRV->GetTexture();
	if (srcTex && srcTex->GetDesc().Format == dstTex->GetDesc().Format)
	{
		context->CopyTexture(CopyTextureAttribs{srcTex, RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
		                                        dstTex, RESOURCE_STATE_TRANSITION_MODE_TRANSITION});
		return;
	}
	const TEXTURE_FORMAT fmt = dstTex->GetDesc().Format;
	auto it = blitPipes.find(fmt);
	if (it == blitPipes.end())
	{
		static const char* kBlitPS =
			"Texture2D g_Source; SamplerState g_Source_sampler;\n"
			"struct PSIn { float4 pos : SV_POSITION; float2 uv : TEXCOORD0; };\n"
			"float4 main(in PSIn i) : SV_TARGET { return g_Source.Sample(g_Source_sampler, i.uv); }\n";
		std::string vs = shaderSource("post.vs");
		if (vs.empty()) return;
		ShaderCreateInfo sci; sci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
		RefCntAutoPtr<IShader> v, p;
		sci.Desc = {"Blit VS", SHADER_TYPE_VERTEX, true}; sci.Source = vs.c_str();  device->CreateShader(sci, &v);
		sci.Desc = {"Blit PS", SHADER_TYPE_PIXEL,  true}; sci.Source = kBlitPS;     device->CreateShader(sci, &p);
		if (!v || !p) return;
		GraphicsPipelineStateCreateInfo ci; ci.PSODesc.Name = "Blit PSO";
		auto& gp = ci.GraphicsPipeline;
		gp.NumRenderTargets = 1; gp.RTVFormats[0] = fmt;
		gp.DSVFormat = TEX_FORMAT_UNKNOWN;
		gp.PrimitiveTopology = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		gp.RasterizerDesc.CullMode = CULL_MODE_NONE;
		gp.DepthStencilDesc.DepthEnable = False;
		gp.InputLayout.NumElements = 0;
		SamplerDesc samp; samp.MinFilter = FILTER_TYPE_POINT; samp.MagFilter = FILTER_TYPE_POINT; samp.MipFilter = FILTER_TYPE_POINT;
		samp.AddressU = TEXTURE_ADDRESS_CLAMP; samp.AddressV = TEXTURE_ADDRESS_CLAMP;
		ShaderResourceVariableDesc var{SHADER_TYPE_PIXEL, "g_Source", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC};
		ImmutableSamplerDesc       imm{SHADER_TYPE_PIXEL, "g_Source", samp};
		ci.PSODesc.ResourceLayout.Variables = &var; ci.PSODesc.ResourceLayout.NumVariables = 1;
		ci.PSODesc.ResourceLayout.ImmutableSamplers = &imm; ci.PSODesc.ResourceLayout.NumImmutableSamplers = 1;
		ci.pVS = v; ci.pPS = p;
		RefCntAutoPtr<IPipelineState> pso;
		device->CreateGraphicsPipelineState(ci, &pso);
		if (!pso) return;
		RefCntAutoPtr<IShaderResourceBinding> srb;
		pso->CreateShaderResourceBinding(&srb, true);
		it = blitPipes.emplace(fmt, std::make_pair(pso, srb)).first;
	}
	IPipelineState* pso = it->second.first;
	IShaderResourceBinding* srb = it->second.second;
	if (!pso || !srb) return;
	const TextureDesc& dd = dstTex->GetDesc();
	ITextureView* rtv = dstTex->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET);
	if (!rtv) return;
	context->SetRenderTargets(1, &rtv, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	Viewport vp; vp.TopLeftX = 0; vp.TopLeftY = 0; vp.Width = (float)dd.Width; vp.Height = (float)dd.Height; vp.MinDepth = 0; vp.MaxDepth = 1;
	context->SetViewports(1, &vp, dd.Width, dd.Height);
	if (auto* v = srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Source")) v->Set(srcSRV);
	context->SetPipelineState(pso);
	context->CommitShaderResources(srb, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	DrawAttribs da{3, DRAW_FLAG_VERIFY_STATES};
	context->Draw(da);
	context->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE);
}

void NukeDiligent::Impl::EnsureScratch(int w, int h)
{
	if (w <= 0 || h <= 0) return;
	if (scratch[0] && scratchW == w && scratchH == h) return;
	scratchW = w; scratchH = h;
	for (int i = 0; i < 2; ++i)
	{
		scratch[i].Release();
		TextureDesc td; td.Name = "Post Scratch"; td.Type = RESOURCE_DIM_TEX_2D; td.Width = (Uint32)w; td.Height = (Uint32)h;
		td.Format = HDR_FMT; td.BindFlags = BIND_RENDER_TARGET | BIND_SHADER_RESOURCE;
		device->CreateTexture(td, nullptr, &scratch[i]);
	}
}

void NukeDiligent::Impl::EnsureBloom(int w, int h)
{
	int bw = w > 1 ? w / 2 : 1, bh = h > 1 ? h / 2 : 1;
	if (bloomTex[0] && bloomW == bw && bloomH == bh) return;
	bloomW = bw; bloomH = bh;
	for (int i = 0; i < 2; ++i)
	{
		bloomTex[i].Release();
		TextureDesc td; td.Name = "Bloom"; td.Type = RESOURCE_DIM_TEX_2D; td.Width = (Uint32)bw; td.Height = (Uint32)bh;
		td.Format = HDR_FMT; td.BindFlags = BIND_RENDER_TARGET | BIND_SHADER_RESOURCE;
		device->CreateTexture(td, nullptr, &bloomTex[i]);
	}
}

// Built-in bloom: bright-pass (-> half-res A) -> separable blur (A<->B, a few iterations) -> composite
// (scene + A*intensity -> dst).
void NukeDiligent::Impl::RunBloom(ITextureView* srcSRV, ITextureView* dstRTV, int w, int h, float threshold, float intensity)
{
	if (!bloomBrightPSO || !bloomBlurPSO || !bloomCompPSO || !srcSRV || !dstRTV) return;
	EnsureBloom(w, h);
	if (!bloomTex[0] || !bloomTex[1]) return;
	const int bw = bloomW, bh = bloomH;
	auto setCB = [&](float dx, float dy)
	{
		MapHelper<float> cb(context, bloomCB, MAP_WRITE, MAP_FLAG_DISCARD);
		cb[0] = threshold; cb[1] = intensity; cb[2] = 0; cb[3] = 0; cb[4] = dx; cb[5] = dy; cb[6] = 0; cb[7] = 0;
	};
	auto pass = [&](IPipelineState* pso, IShaderResourceBinding* srb, IShaderResourceVariable* sv, ITextureView* in, ITextureView* out, int vw, int vh)
	{
		context->SetRenderTargets(1, &out, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		Viewport vp; vp.TopLeftX = 0; vp.TopLeftY = 0; vp.Width = (float)vw; vp.Height = (float)vh; vp.MinDepth = 0; vp.MaxDepth = 1;
		context->SetViewports(1, &vp, vw, vh);
		if (sv) sv->Set(in);
		context->SetPipelineState(pso);
		context->CommitShaderResources(srb, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		DrawAttribs da{3, DRAW_FLAG_VERIFY_STATES}; context->Draw(da);
	};
	ITextureView* aRTV = bloomTex[0]->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET);
	ITextureView* aSRV = bloomTex[0]->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
	ITextureView* bRTV = bloomTex[1]->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET);
	ITextureView* bSRV = bloomTex[1]->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
	setCB(0, 0);                                          // 1) bright-pass: scene -> A (half res)
	pass(bloomBrightPSO, bloomBrightSRB, bbSrc, srcSRV, aRTV, bw, bh);
	const float tx = bw ? 1.0f / bw : 0.0f, ty = bh ? 1.0f / bh : 0.0f;
	for (int it = 0; it < 3; ++it)                        // 2) separable Gaussian blur (H then V)
	{
		setCB(tx, 0.0f); pass(bloomBlurPSO, bloomBlurSRB, blSrc, aSRV, bRTV, bw, bh);
		setCB(0.0f, ty); pass(bloomBlurPSO, bloomBlurSRB, blSrc, bSRV, aRTV, bw, bh);
	}
	setCB(0, 0);                                          // 3) composite: scene + A*intensity -> dst (full res)
	context->SetRenderTargets(1, &dstRTV, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	Viewport vp; vp.TopLeftX = 0; vp.TopLeftY = 0; vp.Width = (float)w; vp.Height = (float)h; vp.MinDepth = 0; vp.MaxDepth = 1;
	context->SetViewports(1, &vp, w, h);
	if (bcSrc)   bcSrc->Set(srcSRV);
	if (bcBloom) bcBloom->Set(aSRV);
	context->SetPipelineState(bloomCompPSO);
	context->CommitShaderResources(bloomCompSRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	DrawAttribs da{3, DRAW_FLAG_VERIFY_STATES}; context->Draw(da);
}

void NukeDiligent::setMSAA(int s)
{
	int req = (s >= 8) ? 8 : (s >= 4) ? 4 : (s >= 2) ? 2 : 1;   // snap to a power of two (1 = off)
	if (!m_impl->device) { m_impl->samples = (Uint8)req; return; }   // before init: Setup will clamp + build
	Uint32 colorSC = (Uint32)m_impl->device->GetTextureFormatInfoExt(TEX_FORMAT_RGBA8_UNORM).SampleCounts;
	Uint32 depthSC = (Uint32)m_impl->device->GetTextureFormatInfoExt(TEX_FORMAT_D32_FLOAT).SampleCounts;
	while (req > 1 && !((colorSC & req) && (depthSC & req))) req >>= 1;
	// Defer the rebuild to the start of the next frame — doing it now (mid ImGui frame) would free the RT
	// textures the current frame's draw data still references (UI ImGui::Image), crashing renderDrawLists.
	m_impl->pendingSamples = req;
}

int NukeDiligent::getMSAA() { return m_impl->pendingSamples > 0 ? m_impl->pendingSamples : (int)m_impl->samples; }

void NukeDiligent::setHDR(bool on)
{
	if (!m_impl->device) { m_impl->hdr = on; return; }   // before init: Setup builds at this setting
	m_impl->pendingHDR = on ? 1 : 0;                       // applied at the start of the next frame (see render())
}
bool NukeDiligent::getHDR() { return m_impl->pendingHDR >= 0 ? (m_impl->pendingHDR != 0) : m_impl->hdr; }

uint64_t NukeDiligent::createPostPipeline(const char* name, const char* ps) { return m_impl->CreatePostPipe(name ? name : "", ps ? ps : ""); }

void NukeDiligent::setPostChain(const NukePostStage* stages, int count)
{
	m_impl->postChain.clear();
	for (int i = 0; i < count; ++i)
	{
		Impl::ChainStage cs; cs.pipeline = stages[i].pipeline;
		int n = stages[i].paramFloats > 0 ? stages[i].paramFloats : 0;
		if (stages[i].params && n > 0) cs.params.assign(stages[i].params, stages[i].params + n);
		m_impl->postChain.push_back(std::move(cs));
	}
}

void NukeDiligent::setHDROutput(bool on)
{
#ifdef _WIN32
	m_impl->hdrOutput = on;   // honoured at init (swap-chain format + HDR10 colour space)
#else
	(void)on;                 // no D3D11/DXGI off Windows — HDR output stays off
#endif
}
bool NukeDiligent::getHDROutput() { return m_impl->hdr10Active; }

void NukeDiligent::setHDRNits(float paperWhite, float peak)
{
	m_impl->hdrPaperWhite = paperWhite > 1.0f ? paperWhite : 1.0f;
	m_impl->hdrPeak = peak > m_impl->hdrPaperWhite ? peak : m_impl->hdrPaperWhite;
}
