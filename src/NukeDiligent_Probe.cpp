#include "NukeDiligentImpl.h"


// (Re)create a probe cube's GPU resources at the CURRENT scene format (SceneFmt) so capture into it matches
// the geometry PSOs' RTV format. Rebuild-safe (releases prior objects -> no Diligent overwrite assert).
void NukeDiligent::Impl::BuildCube(CubeRT& c, int res)
{
	res = res < 16 ? 16 : (res > 1024 ? 1024 : res);
	// A rebuild (HDR toggle / res change) can land mid-frame while the world PSOs still sample the old
	// cube SRV — park everything first (centralized lifetime rule).
	Trash(c.color); Trash(c.depth); Trash(c.dsv); Trash(c.msColor); Trash(c.msDepth);
	for (auto& v : c.faceRTV) Trash(v);
	c.color.Release(); c.depth.Release(); c.dsv.Release(); c.msColor.Release(); c.msDepth.Release();
	for (auto& v : c.faceRTV) v.Release();
	c.srv = nullptr;
	c.res = res; c.fmtHdr = hdr; c.msSamples = (int)samples;
	int mips = 1; { int s = res; while (s > 1) { s >>= 1; ++mips; } } c.mips = mips;

	TextureDesc cd; cd.Name = "Probe Cube"; cd.Type = RESOURCE_DIM_TEX_CUBE; cd.Width = (Uint32)res; cd.Height = (Uint32)res;
	cd.ArraySize = 6; cd.MipLevels = (Uint32)mips; cd.Format = SceneFmt();   // match the world/Shader PSO RTV
	cd.BindFlags = BIND_RENDER_TARGET | BIND_SHADER_RESOURCE; cd.MiscFlags = MISC_TEXTURE_FLAG_GENERATE_MIPS;
	device->CreateTexture(cd, nullptr, &c.color);
	if (!c.color) return;
	c.srv = c.color->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
	if (c.srv && probeSampler) c.srv->SetSampler(probeSampler);   // combined-texture-samplers reads it from the view
	for (int f = 0; f < 6; ++f)
	{
		TextureViewDesc vd; vd.Name = "Probe face RTV"; vd.ViewType = TEXTURE_VIEW_RENDER_TARGET;
		vd.TextureDim = RESOURCE_DIM_TEX_2D_ARRAY; vd.FirstArraySlice = (Uint32)f; vd.NumArraySlices = 1;
		vd.MostDetailedMip = 0; vd.NumMipLevels = 1;
		c.color->CreateView(vd, &c.faceRTV[f]);
	}
	TextureDesc dd; dd.Name = "Probe Depth"; dd.Type = RESOURCE_DIM_TEX_2D; dd.Width = (Uint32)res; dd.Height = (Uint32)res;
	dd.Format = TEX_FORMAT_D32_FLOAT; dd.BindFlags = BIND_DEPTH_STENCIL;
	device->CreateTexture(dd, nullptr, &c.depth);
	if (c.depth) c.dsv = c.depth->GetDefaultView(TEXTURE_VIEW_DEPTH_STENCIL);

	// MSAA intermediates: the geometry/sky PSOs are MSAA-only when MSAA is on — capture
	// renders here and RESOLVES into the cube face (single-sample direct rendering with an
	// MSAA PSO device-losts Vulkan).
	if (samples > 1)
	{
		TextureDesc md; md.Name = "Probe MS color"; md.Type = RESOURCE_DIM_TEX_2D;
		md.Width = (Uint32)res; md.Height = (Uint32)res; md.Format = SceneFmt();
		md.SampleCount = samples; md.MipLevels = 1; md.BindFlags = BIND_RENDER_TARGET;
		device->CreateTexture(md, nullptr, &c.msColor);
		TextureDesc mdd; mdd.Name = "Probe MS depth"; mdd.Type = RESOURCE_DIM_TEX_2D;
		mdd.Width = (Uint32)res; mdd.Height = (Uint32)res; mdd.Format = TEX_FORMAT_D32_FLOAT;
		mdd.SampleCount = samples; mdd.MipLevels = 1; mdd.BindFlags = BIND_DEPTH_STENCIL;
		device->CreateTexture(mdd, nullptr, &c.msDepth);
		if (!c.msColor || !c.msDepth) { c.msColor.Release(); c.msDepth.Release(); }   // pair or nothing
	}
}

uint64_t NukeDiligent::createReflectionCube(int resolution)
{
	if (!m_impl->device) return 0;
	uint64_t id = ++m_impl->rtCounter;
	Impl::CubeRT& c = m_impl->cubes[id];
	m_impl->BuildCube(c, resolution);
	if (!c.color) { m_impl->cubes.erase(id); return 0; }
	return id;
}

void NukeDiligent::beginCubeFace(uint64_t cube, int face, const float pos[3], float nearZ, float farZ)
{
	auto it = m_impl->cubes.find(cube);
	if (it == m_impl->cubes.end() || face < 0 || face > 5) return;
	Impl::CubeRT& c = it->second;
	// HDR or MSAA changed -> rebuild so capture targets match the geometry PSOs exactly.
	if (c.fmtHdr != m_impl->hdr || c.msSamples != (int)m_impl->samples) m_impl->BuildCube(c, c.res);
	if (!c.faceRTV[face] || !c.dsv) return;
	m_impl->probeActive = false;   // never sample the probe while capturing it (analytic IBL) -> no feedback
	m_impl->curTarget = 0;

	// D3D cube-face orientations (look dir + up), so the captured faces match TextureCube sampling.
	static const float3 F6[6] = { { 1,0,0}, {-1,0,0}, {0, 1,0}, {0,-1,0}, {0,0, 1}, {0,0,-1} };
	static const float3 U6[6] = { { 0,1,0}, { 0,1,0}, {0,0,-1}, {0,0, 1}, {0,1, 0}, {0,1, 0} };
	float3 P(pos[0], pos[1], pos[2]);
	float3 F = F6[face], U = U6[face], R = normalize(cross(U, F)); U = cross(F, R);
	m_impl->curView = float4x4(R.x,U.x,F.x,0, R.y,U.y,F.y,0, R.z,U.z,F.z,0, -dot(P,R),-dot(P,U),-dot(P,F),1);
	m_impl->curProj = float4x4::Projection(1.5707963f, 1.0f, nearZ, farZ, false);   // 90deg, square
	m_impl->curCamPos[0] = P.x; m_impl->curCamPos[1] = P.y; m_impl->curCamPos[2] = P.z;
	// MSAA on: sky/world PSOs are MULTISAMPLED — render the face into the MS intermediate
	// and resolve into the cube slice in endCubeFace. Rendering an MSAA PSO into the
	// single-sample face RTV is a Vulkan render-pass incompatibility (DEVICE_LOST).
	const bool ms = c.msColor && c.msDepth;
	ITextureView* rtv = ms ? c.msColor->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET) : c.faceRTV[face].RawPtr();
	ITextureView* dsv = ms ? c.msDepth->GetDefaultView(TEXTURE_VIEW_DEPTH_STENCIL) : c.dsv.RawPtr();
	if (!rtv || !dsv) return;
	m_impl->curRTV = rtv; m_impl->curDSV = dsv; m_impl->curRTW = c.res; m_impl->curRTH = c.res;

	IDeviceContext* ctx = m_impl->context;
	ctx->SetRenderTargets(1, &rtv, dsv, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	const float clr[4] = { 0, 0, 0, 1 };
	ctx->ClearRenderTarget(rtv, clr, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	ctx->ClearDepthStencil(dsv, CLEAR_DEPTH_FLAG, 1.f, 0, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	Viewport vp; vp.TopLeftX = 0; vp.TopLeftY = 0; vp.Width = (float)c.res; vp.Height = (float)c.res; vp.MinDepth = 0; vp.MaxDepth = 1;
	ctx->SetViewports(1, &vp, c.res, c.res);

	m_impl->WriteFrameCB(P);   // probe off (probeActive=false) -> analytic IBL during capture
	m_impl->DrawSky();
}

void NukeDiligent::endCubeFace(uint64_t cube, int face)
{
	auto it = m_impl->cubes.find(cube);
	if (it == m_impl->cubes.end() || face < 0 || face > 5) return;
	Impl::CubeRT& c = it->second;
	// MSAA capture: resolve the multisampled face into this cube slice.
	if (c.msColor && c.color)
	{
		ResolveTextureSubresourceAttribs ra;
		ra.SrcTextureTransitionMode = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
		ra.DstTextureTransitionMode = RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
		ra.Format      = m_impl->SceneFmt();
		ra.DstSlice    = (Uint32)face;
		ra.DstMipLevel = 0;
		m_impl->context->ResolveTextureSubresource(c.msColor, c.color, ra);
	}
	if (face != 5) return;   // all six faces captured -> build the mip chain for rough reflections
	if (c.srv) m_impl->context->GenerateMips(c.srv);
}

void NukeDiligent::setReflectionProbe(uint64_t cube, const float pos[3], float intensity, float farZ, const float boxHalf[3])
{
	(void)farZ;
	auto it = (cube != 0) ? m_impl->cubes.find(cube) : m_impl->cubes.end();
	if (it != m_impl->cubes.end() && it->second.srv)
	{
		m_impl->probeActive = true;
		m_impl->probeCubeSRV = it->second.srv;
		m_impl->probePos[0] = pos[0]; m_impl->probePos[1] = pos[1]; m_impl->probePos[2] = pos[2];
		m_impl->probeIntensity = intensity;
		m_impl->probeMaxMip = (float)(it->second.mips - 1);
		m_impl->probeBoxHalf[0] = boxHalf ? boxHalf[0] : 0.f; m_impl->probeBoxHalf[1] = boxHalf ? boxHalf[1] : 0.f; m_impl->probeBoxHalf[2] = boxHalf ? boxHalf[2] : 0.f;
	}
	else { m_impl->probeActive = false; m_impl->probeCubeSRV = nullptr; m_impl->probeBoxHalf[0] = m_impl->probeBoxHalf[1] = m_impl->probeBoxHalf[2] = 0.f; }
}
