#include "NukeDiligentImpl.h"
#include "../include/NukeDiligentNative.h"

// The NATIVE ESCAPE HATCH (NukeDiligentNative.h): the exported surface a backend-specific
// companion module (NukeWater) drives its own GPU passes through, plus the GENERIC ortho
// bottom-depth capture that stayed a renderer capability (it rides the shadow path and any
// module may capture top-down depth with it — begin/end/fetchWaterBottom on iRender).

// The active renderer's Impl comes from the public static (set by ctor/dtor).

namespace nukediligent {

static WaterHooks g_hooks;

const WaterHooks& ActiveWaterHooks() { return g_hooks; }   // renderer-internal (Scene.cpp)

bool GetFrame(Frame& out)
{
	NukeDiligent::Impl* d = NukeDiligent::nativeImpl;
	if (!d || !d->device || !d->context) return false;
	out.device = d->device;
	out.context = d->context;
	out.shaderFactory = d->shaderFactory;
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

// ---- generic ortho bottom-depth capture (iRender seam; rides the shadow draw path) ----------
// An ortho top-down DEPTH render over a rect: begin sets curShadowVP + binds the small D32
// target, the World submits opaque meshes with renderShadowObject, end copies to staging;
// fetch maps it a few frames later and converts to meters-below-level. One in flight.

static const int   kBottomCapN    = 128;     // must equal WaterBody::kDepthN (the fetch guard)
static const float kBottomCapNear = 0.1f;
static const float kBottomCapFar  = 120.0f;  // 50 up + up to ~60 below the level
static const float kBottomCapEye  = 50.0f;   // ortho eye height above the reference level

void NukeDiligent::beginWaterBottomPass(const float pos[3], const float quat[4], float sizeX, float sizeZ)
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
	float4x4 proj = float4x4::Ortho(std::max(sizeX, 1.0f), std::max(sizeZ, 1.0f), kBottomCapNear, kBottomCapFar, false);
	++d->passSerial;
	d->curShadowVP = view * proj;
	d->capLevel = pos[1];
	d->capEyeY  = eye.y;

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
				const float worldY = d->capEyeY - (kBottomCapNear + z * (kBottomCapFar - kBottomCapNear));
				d2 = d->capLevel - worldY;
			}
			// -40 floor: consumers reuse this capture as raw TERRAIN around the reference —
			// hills well above it must survive (shore math only eats the shallow band).
			dst[i] = std::max(std::min(d2, 60.0f), -40.0f);
		}
	}
	d->context->UnmapTextureSubresource(d->capStaging, 0, 0);
	d->capPending = -1;
	return true;
}
