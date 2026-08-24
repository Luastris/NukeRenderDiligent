#include "NukeDiligentImpl.h"
#include "../include/NukeDiligentNative.h"

// Implementation of the native escape hatch (NukeDiligentNative.h) plus the generic ortho
// bottom-depth capture on iRender. The active Impl comes from the NukeDiligent::nativeImpl static.

namespace nukediligent {

static WaterHooks g_hooks;

const WaterHooks& ActiveWaterHooks() { return g_hooks; }   // renderer-internal (Scene.cpp)

bool GetFrame(Frame& out)
{
	NukeDiligent::Impl* d = NukeDiligent::nativeImpl;
	if (!d || !d->device || !d->context) return false;
	out.device = d->device;
	out.context = d->context;
	out.shaderFactory = d->ShaderFactory();   // raw for the seam; retired factories outlive the run
	out.cameraPassActive = d->cameraPassActive;
	out.camIsEditor = d->curCamEditor;
	out.curRTV = d->curRTV;
	out.curDSV = d->curDSV;
	out.curRTW = d->curRTW; out.curRTH = d->curRTH;
	out.samples = d->samples;
	out.sceneFmt = d->SceneFmt();
	memcpy(out.view, &d->curView, sizeof(float) * 16);
	memcpy(out.proj, &d->curProj, sizeof(float) * 16);
	memcpy(out.projNoJitter, &d->curProjNoJitter, sizeof(float) * 16);
	memcpy(out.camPos, d->curCamPos, sizeof(float) * 3);
	memcpy(out.camFwd, d->curCamFwd, sizeof(float) * 3);
	out.nearZ = d->curNear; out.farZ = d->curFar;
	out.sceneMSAAColor = d->curMSAA ? d->curResolveSrc : nullptr;
	out.scenePostSRV = d->curPostSrc;
	out.frameId = d->frameId;
	out.passSerial = d->passSerial;
	out.curTarget = d->curTarget;
	out.sceneDepthSRV = d->gbufDepthSRV;
	out.gbufActive = d->gbufActive;
	out.whiteSRV = d->whiteTex ? d->whiteTex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE) : nullptr;
	out.fallbackCubeSRV = d->fallbackCubeSRV;
	out.probeCubeSRV = d->probeCubeSRV;
	out.probeActive = d->probeActive;
	out.shadowSRV = d->shadowSRV;
	out.worldFrameCB = d->worldFrameCB;
	out.rtSupported = d->rtSupported;
	out.lights = d->lights.empty() ? nullptr : d->lights.data();
	out.lightCount = (int)d->lights.size();
	return true;
}

bool GetShaderSource(const char* name, std::string& out)
{
	if (!NukeDiligent::nativeImpl || !name) return false;
	out = NukeDiligent::nativeImpl->shaderSource(name);
	return !out.empty();
}

void CreateShaderCached(Diligent::ShaderCreateInfo& sci, Diligent::IShader** out)
{
	if (NukeDiligent::nativeImpl && out) NukeDiligent::nativeImpl->CreateShaderCached(sci, out);
}

void CreateGraphicsPSOCached(Diligent::GraphicsPipelineStateCreateInfo& ci, Diligent::IPipelineState** out)
{
	if (NukeDiligent::nativeImpl && out) NukeDiligent::nativeImpl->CreateGraphicsPipelineStateCached(ci, out);
}

void EnqueueBuild(const boost::function<void()>& build, const boost::function<void()>& adopt, int prio, const char* name)
{
	if (NukeDiligent::nativeImpl) NukeDiligent::nativeImpl->EnqueueBuild(build, adopt, prio, name ? name : "");
	else { if (build) build(); if (adopt) adopt(); }   // no renderer: run inline, same contract
}

void AddPipelineWarmup(const char* name, WarmupFn fn, void* user)
{
	NukeDiligent::Impl* d = NukeDiligent::nativeImpl;
	if (!d || !fn) return;
	for (auto& e : d->warmups) if (e.user == user && e.fn == fn) return;   // idempotent
	NukeDiligent::Impl::WarmEntry e;
	e.name = name ? name : "pipelines"; e.fn = fn; e.user = user;
	d->warmups.push_back(e);
}

void RearmPipelineWarmup(void* user)
{
	NukeDiligent::Impl* d = NukeDiligent::nativeImpl;
	if (!d) return;
	for (auto& e : d->warmups) if (e.user == user) e.done = false;
}

void RemovePipelineWarmup(void* user)
{
	NukeDiligent::Impl* d = NukeDiligent::nativeImpl;
	if (!d) return;
	for (size_t i = 0; i < d->warmups.size(); )
	{
		if (d->warmups[i].user == user) d->warmups.erase(d->warmups.begin() + i);
		else ++i;
	}
}

void Trash(Diligent::IDeviceObject* obj)
{
	if (NukeDiligent::nativeImpl && obj) NukeDiligent::nativeImpl->Trash(obj);
}

void FlushBatches()
{
	if (!NukeDiligent::nativeImpl) return;
	NukeDiligent::nativeImpl->FlushSprites();
	NukeDiligent::nativeImpl->FlushSpritesLit();
}

void NoteDraw(int tris)
{
	if (!NukeDiligent::nativeImpl) return;
	++NukeDiligent::nativeImpl->statDraws;
	NukeDiligent::nativeImpl->statTris += tris;
	NukeDiligent::nativeImpl->lastInstBind.pso = nullptr;   // raw pass work invalidated the bind cache
}

Diligent::ITextureView* GetPostScratchRTV(int idx, int w, int h)
{
	if (!NukeDiligent::nativeImpl || idx < 0 || idx > 1) return nullptr;
	NukeDiligent::nativeImpl->EnsureScratch(w, h);
	if (!NukeDiligent::nativeImpl->scratch[idx]) return nullptr;
	return NukeDiligent::nativeImpl->scratch[idx]->GetDefaultView(TEXTURE_VIEW_RENDER_TARGET);
}

void SetWaterHooks(const WaterHooks* hooks)
{
	if (hooks) g_hooks = *hooks;
	else       g_hooks = WaterHooks();
}

void SetRTWaterState(float level, float on, float fade, const float scatter[3], const float absorb[3])
{
	NukeDiligent::Impl* d = NukeDiligent::nativeImpl;
	if (!d) return;
	d->rtWaterOcc[0] = level; d->rtWaterOcc[1] = on; d->rtWaterOcc[2] = fade; d->rtWaterOcc[3] = 0.0f;
	if (scatter) memcpy(d->rtWaterCol, scatter, sizeof(float) * 3);
	if (absorb)  memcpy(d->rtWaterAbs, absorb, sizeof(float) * 3);
}

}  // namespace nukediligent

// Ortho top-down depth capture over a rect, riding the shadow draw path: begin binds the D32
// target, the World submits via renderShadowObject, end copies to staging, fetch maps it a few
// frames later as meters-below-level. One capture in flight.

static const int   kBottomCapN    = 128;     // must equal WaterBody::kDepthN (the fetch guard)
static const float kBottomCapEye  = 50.0f;   // ortho eye height above the reference level
static const float kBottomCapFar  = 120.0f;  // the level plus ~70 m of depth below it

void NukeDiligent::beginWaterBottomPass(const float pos[3], const float quat[4], float sizeX, float sizeZ,
                                        float aboveY)
{
	Impl* d = m_impl;
	if (d->capPending >= 0 || d->capActive) return;   // one capture in flight
	if (!d->capDepth)
	{
		TextureDesc td; td.Name = "Bottom Capture"; td.Type = RESOURCE_DIM_TEX_2D;
		td.Width = kBottomCapN; td.Height = kBottomCapN; td.Format = TEX_FORMAT_D32_FLOAT;
		td.MipLevels = 1; td.BindFlags = BIND_DEPTH_STENCIL;
		d->device->CreateTexture(td, nullptr, &d->capDepth);
		TextureDesc sd; sd.Name = "Bottom Capture Staging"; sd.Type = RESOURCE_DIM_TEX_2D;
		sd.Width = kBottomCapN; sd.Height = kBottomCapN; sd.Format = TEX_FORMAT_D32_FLOAT;
		sd.MipLevels = 1; sd.BindFlags = BIND_NONE;
		sd.Usage = USAGE_STAGING; sd.CPUAccessFlags = CPU_ACCESS_READ;
		d->device->CreateTexture(sd, nullptr, &d->capStaging);
	}
	if (!d->capDepth || !d->capStaging || !d->shadowPSO) return;

	// World -> capture space: X along the rect's local +X, Y along local +Z, Z straight down.
	float4x4 R = Diligent::Quaternion<float>(quat[0], quat[1], quat[2], quat[3]).ToMatrix();
	float3 ax(R.m[0][0], R.m[0][1], R.m[0][2]);
	float3 az(R.m[2][0], R.m[2][1], R.m[2][2]);
	float3 dn(0.0f, -1.0f, 0.0f);
	float3 eye(pos[0], pos[1] + kBottomCapEye, pos[2]);
	float4x4 view(
		ax.x, az.x, dn.x, 0.0f,
		ax.y, az.y, dn.y, 0.0f,
		ax.z, az.z, dn.z, 0.0f,
		-dot(eye, ax), -dot(eye, az), -dot(eye, dn), 1.0f);
	// The near plane decides how much of the world ABOVE the reference level takes part.
	// aboveY = 0 clips at the level itself: the result is the bottom of the water column only,
	// so nothing hanging in the air can be mistaken for a shoal. A positive aboveY lets terrain
	// that rises out of the water through, which the consumers that read this as raw height need.
	const float capNear = std::max(kBottomCapEye - std::max(aboveY, 0.0f), 0.05f);
	float4x4 proj = float4x4::Ortho(std::max(sizeX, 1.0f), std::max(sizeZ, 1.0f), capNear, kBottomCapFar, false);
	++d->passSerial;
	d->curShadowVP = view * proj;
	d->capLevel = pos[1];
	d->capEyeY  = eye.y;
	d->capNear  = capNear;

	IDeviceContext* ctx = d->context;
	ITextureView* dsv = d->capDepth->GetDefaultView(TEXTURE_VIEW_DEPTH_STENCIL);
	ctx->SetRenderTargets(0, nullptr, dsv, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	ctx->ClearDepthStencil(dsv, CLEAR_DEPTH_FLAG, 1.f, 0, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	Viewport vp; vp.TopLeftX = 0; vp.TopLeftY = 0;
	vp.Width = (float)kBottomCapN; vp.Height = (float)kBottomCapN; vp.MinDepth = 0; vp.MaxDepth = 1;
	ctx->SetViewports(1, &vp, kBottomCapN, kBottomCapN);
	d->capActive = true;
}

void NukeDiligent::endWaterBottomPass()
{
	Impl* d = m_impl;
	if (!d->capActive) return;
	d->capActive = false;
	// Unbind first: the copy transitions the DSV to COPY_SOURCE, which warns while still bound.
	d->context->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE);
	CopyTextureAttribs cp(d->capDepth, RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
	                      d->capStaging, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	d->context->CopyTexture(cp);
	d->capPending = 3;   // let the copy retire before mapping (DO_NOT_WAIT covers stragglers)
}

bool NukeDiligent::fetchWaterBottom(float* out, int n)
{
	Impl* d = m_impl;
	if (!out || n != kBottomCapN || d->capPending < 0) return false;
	if (d->capPending > 0) { --d->capPending; return false; }
	MappedTextureSubresource m;
	d->context->MapTextureSubresource(d->capStaging, 0, 0, MAP_READ, MAP_FLAG_DO_NOT_WAIT, nullptr, m);
	if (!m.pData) return false;   // not ready — retry next frame
	for (int j = 0; j < kBottomCapN; ++j)
	{
		const float* src = (const float*)((const uint8_t*)m.pData + (size_t)j * m.Stride);
		// Ortho NDC y is UP, texture rows go DOWN: row j maps to the rect's -Z half at j max.
		float* dst = out + (size_t)(kBottomCapN - 1 - j) * kBottomCapN;
		for (int i = 0; i < kBottomCapN; ++i)
		{
			const float z = src[i];
			float d2;
			if (z >= 0.9999f) d2 = 60.0f;   // nothing under this texel: deep
			else
			{
				const float worldY = d->capEyeY - (d->capNear + z * (kBottomCapFar - d->capNear));
				d2 = d->capLevel - worldY;
			}
			// -40 floor: consumers reuse this as raw terrain, so hills above the reference must survive.
			dst[i] = std::max(std::min(d2, 60.0f), -40.0f);
		}
	}
	d->context->UnmapTextureSubresource(d->capStaging, 0, 0);
	d->capPending = -1;
	return true;
}
