#include "NukeDiligentImpl.h"
#include <cmath>

// Honest wide FOV: the cylindrical Panini projection as a camera property (NukeCameraDesc::panini).
// A rectilinear image stretches its edges as the FOV grows; Panini keeps verticals straight and
// edge objects their size, so a wider FOV WIDENS the view instead. Done in two halves:
//   1) SetCameraViewProj (Scene.cpp) renders the pass rectilinear, over-scanned vertically by the
//      factor the remap needs at the corners (the horizontal edge angle is the same as rectilinear);
//   2) endCamera tonemaps into a scratch LDR frame and RunPanini remaps it into the real output
//      (Catmull-Rom resampling), before the screen-space HUD. The mapping is the one in panini.ps;
//      Camera::ScreenRayDir (engine) inverts the same formula for picking / screen rays.

using namespace Diligent;

void NukeDiligent::Impl::EnsurePaniniPipes()
{
	if (PaniniReady() || paniniFailed || paniniBuilding.exchange(true)) return;
	EnqueueBuild([this]
	{
		std::string vs = shaderSource("post.vs"), ps = shaderSource("panini.ps");
		if (vs.empty() || ps.empty()) { paniniFailed = true; return; }
		if (!paniniCB)
		{
			BufferDesc cbd; cbd.Name = "PaniniCB"; cbd.Size = sizeof(float) * 8;
			cbd.Usage = USAGE_DYNAMIC; cbd.BindFlags = BIND_UNIFORM_BUFFER; cbd.CPUAccessFlags = CPU_ACCESS_WRITE;
			device->CreateBuffer(cbd, nullptr, &paniniCB);
			if (!paniniCB) { paniniFailed = true; return; }
		}
		ShaderCreateInfo sci; sci.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
		sci.pShaderSourceStreamFactory = ShaderFactory();
		RefCntAutoPtr<IShader> v, p;
		sci.Desc = {"Panini VS", SHADER_TYPE_VERTEX, true}; sci.Source = vs.c_str(); CreateShaderCached(sci, &v);
		sci.Desc = {"Panini PS", SHADER_TYPE_PIXEL, true};  sci.Source = ps.c_str(); CreateShaderCached(sci, &p);
		if (!v || !p) { paniniFailed = true; return; }
		auto make = [&](const char* dbg, TEXTURE_FORMAT fmt, RefCntAutoPtr<IPipelineState>& pso,
		                RefCntAutoPtr<IShaderResourceBinding>& srb, IShaderResourceVariable*& var) -> bool
		{
			GraphicsPipelineStateCreateInfo ci; ci.PSODesc.Name = dbg;
			auto& gp = ci.GraphicsPipeline;
			gp.NumRenderTargets = 1; gp.RTVFormats[0] = fmt; gp.DSVFormat = TEX_FORMAT_UNKNOWN;
			gp.PrimitiveTopology = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; gp.RasterizerDesc.CullMode = CULL_MODE_NONE;
			gp.DepthStencilDesc.DepthEnable = False; gp.SmplDesc.Count = 1; gp.InputLayout.NumElements = 0;
			ci.pVS = v; ci.pPS = p;
			// The remap samples bilinear taps of a Catmull-Rom kernel: a linear clamp sampler.
			SamplerDesc s; s.MinFilter = FILTER_TYPE_LINEAR; s.MagFilter = FILTER_TYPE_LINEAR; s.MipFilter = FILTER_TYPE_LINEAR;
			s.AddressU = TEXTURE_ADDRESS_CLAMP; s.AddressV = TEXTURE_ADDRESS_CLAMP; s.AddressW = TEXTURE_ADDRESS_CLAMP;
			ShaderResourceVariableDesc vars[] = {{SHADER_TYPE_PIXEL, "g_Source", SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC}};
			ImmutableSamplerDesc imms[] = {{SHADER_TYPE_PIXEL, "g_Source", s}};
			ci.PSODesc.ResourceLayout.Variables = vars; ci.PSODesc.ResourceLayout.NumVariables = 1;
			ci.PSODesc.ResourceLayout.ImmutableSamplers = imms; ci.PSODesc.ResourceLayout.NumImmutableSamplers = 1;
			ci.PSODesc.ResourceLayout.DefaultVariableType = SHADER_RESOURCE_VARIABLE_TYPE_STATIC;
			CreateGraphicsPipelineStateCached(ci, &pso);
			if (!pso) return false;
			if (auto* c = pso->GetStaticVariableByName(SHADER_TYPE_PIXEL, "PaniniCB")) c->Set(paniniCB);
			pso->CreateShaderResourceBinding(&srb, true);
			if (!srb) return false;
			var = srb->GetVariableByName(SHADER_TYPE_PIXEL, "g_Source");
			return true;
		};
		const TEXTURE_FORMAT bbFmt = swapChain ? swapChain->GetDesc().ColorBufferFormat : TEX_FORMAT_RGBA8_UNORM;
		if (!make("Panini PS", TEX_FORMAT_RGBA8_UNORM, paniniPSO, paniniSRB, paniniSrcVar) ||
		    !make("Panini PS BB", bbFmt, paniniPSOBB, paniniSRBBB, paniniSrcVarBB))
		{ paniniFailed = true; std::cout << "[NukeDiligent]\tpanini pipelines FAILED" << std::endl; }
		else std::cout << "[NukeDiligent]\tpanini remap ready" << std::endl;
	}, [this] { paniniBuilding = false; }, kPrioExtra, "panini remap");
}

// The un-remapped LDR frame of one camera target (tonemap + gizmo lines draw into it).
NukeDiligent::Impl::PaniniScratch* NukeDiligent::Impl::PaniniTarget(uint64_t target, int w, int h, TEXTURE_FORMAT fmt)
{
	PaniniScratch& s = paniniScratch[target];
	if (s.tex && s.w == w && s.h == h && s.fmt == fmt) return &s;
	Trash(s.tex); s.tex.Release(); s.rtv = s.srv = nullptr;
	TextureDesc td; td.Name = "Panini scratch"; td.Type = RESOURCE_DIM_TEX_2D;
	td.Width = (Uint32)w; td.Height = (Uint32)h; td.Format = fmt; td.BindFlags = BIND_RENDER_TARGET | BIND_SHADER_RESOURCE;
	device->CreateTexture(td, nullptr, &s.tex);
	if (!s.tex) return nullptr;
	s.rtv = s.tex->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET); s.srv = s.tex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
	s.w = w; s.h = h; s.fmt = fmt;
	return &s;
}

void NukeDiligent::Impl::RunPanini(ITextureView* src, ITextureView* dstRTV, int w, int h, bool toBackbuffer)
{
	IPipelineState* pso = toBackbuffer ? paniniPSOBB : paniniPSO;
	IShaderResourceBinding* srb = toBackbuffer ? paniniSRBBB : paniniSRB;
	IShaderResourceVariable* var = toBackbuffer ? paniniSrcVarBB : paniniSrcVar;
	if (!pso || !srb || !src || !dstRTV) return;
	{
		MapHelper<float> cb(context, paniniCB, MAP_WRITE, MAP_FLAG_DISCARD);
		if (cb == nullptr) return;
		cb[0] = curPanini; cb[1] = panXMax; cb[2] = panTanV; cb[3] = panVertS;
		cb[4] = panSrcTanH; cb[5] = panSrcTanV; cb[6] = w ? 1.0f / w : 0.0f; cb[7] = h ? 1.0f / h : 0.0f;
	}
	context->SetRenderTargets(1, &dstRTV, nullptr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	Viewport vp; vp.TopLeftX = 0; vp.TopLeftY = 0; vp.Width = (float)w; vp.Height = (float)h; vp.MinDepth = 0; vp.MaxDepth = 1;
	context->SetViewports(1, &vp, w, h);
	if (var) var->Set(src);
	context->SetPipelineState(pso);
	context->CommitShaderResources(srb, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	DrawAttribs da{3, DRAW_FLAG_VERIFY_STATES};
	context->Draw(da);
}
