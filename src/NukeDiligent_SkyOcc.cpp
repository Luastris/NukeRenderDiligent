// W4 weather: the sky-occlusion capture. A top-down orthographic depth of the opaque world
// around the camera, drawn through the shadow path from a world render hook before the camera
// passes; world.ps / gbuffer.ps (skyocc.hlsli) gate from-sky conditions (rain wet, snow, dust)
// by it, so roofs shelter what is under them. Runs only while some condition is flagged
// from-sky (Surface::AnyConditionFromSky) - zero cost otherwise.
#include "NukeDiligentImpl.h"
#include <API/Model/Surface.h>
#include <interface/WorldHooks.h>

using namespace Diligent;

namespace {
struct SkyOccHook : nuke::WorldRenderHook
{
	void preRender(nuke::iRender*, const boost::function<void()>& submitOpaques) override
	{
		if (!NukeDiligent::nativeImpl) return;
		NukeDiligent::nativeImpl->RunSkyOcclusion(submitOpaques);
		NukeDiligent::nativeImpl->RunTrails();   // W5: the carve map steps with the same gate
	}
};
SkyOccHook gSkyOccHook;

const int   kSkyOccN    = 1024;    // texels per side
const float kSkyOccEye  = 300.0f;  // ortho eye height above the camera
const float kSkyOccNear = 1.0f;
const float kSkyOccFar  = 700.0f;  // 300 m above the camera .. 400 m below
const float kSkyOccBias = 0.25f;   // m: the point itself never shades; > 2 biases above blocks fully
}

namespace nukediligent {
void RegisterSkyOccHook()   { nuke::RegisterWorldRenderHook(&gSkyOccHook); }
void UnregisterSkyOccHook() { nuke::UnregisterWorldRenderHook(&gSkyOccHook); }
}

void NukeDiligent::Impl::RunSkyOcclusion(const boost::function<void()>& submitOpaques)
{
	skyOccSRV = nullptr;
	auto writeCB = [&](const float p[8])
	{
		if (skyOccCB) context->UpdateBuffer(skyOccCB, 0, 32, p, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	};
	if (!nuke::Surface::AnyConditionFromSky() || !shadowPSO || !submitOpaques || !skyOccCB)
	{
		if (!skyOccCBZero) { const float z[8] = {}; writeCB(z); skyOccCBZero = true; }
		return;
	}
	if (!skyOccDepth)
	{
		TextureDesc td; td.Name = "Sky Occlusion"; td.Type = RESOURCE_DIM_TEX_2D;
		td.Width = td.Height = (Uint32)kSkyOccN; td.Format = TEX_FORMAT_D32_FLOAT; td.MipLevels = 1;
		td.BindFlags = BIND_DEPTH_STENCIL | BIND_SHADER_RESOURCE;
		device->CreateTexture(td, nullptr, &skyOccDepth);
		if (!skyOccDepth) return;
	}
	GpuPass("skyocc");
	// The square around the camera, snapped to its texel grid so a moving camera never makes
	// the gate swim; its size follows the shadow distance (where surface detail is looked at).
	const float size  = std::max(shadowDistance * 4.0f, 64.0f);
	const float texel = size / (float)kSkyOccN;
	const float ox = std::floor((curCamPos[0] - size * 0.5f) / texel) * texel;
	const float oz = std::floor((curCamPos[2] - size * 0.5f) / texel) * texel;
	const float3 eye(ox + size * 0.5f, curCamPos[1] + kSkyOccEye, oz + size * 0.5f);
	// World -> capture: X along +X, Y along +Z, Z straight down (the water bottom capture's frame).
	const float3 ax(1, 0, 0), az(0, 0, 1), dn(0, -1, 0);
	float4x4 view(
		ax.x, az.x, dn.x, 0.0f,
		ax.y, az.y, dn.y, 0.0f,
		ax.z, az.z, dn.z, 0.0f,
		-dot(eye, ax), -dot(eye, az), -dot(eye, dn), 1.0f);
	float4x4 proj = float4x4::Ortho(size, size, kSkyOccNear, kSkyOccFar, false);
	++passSerial;   // the shared shadow CBs re-map per pass
	curShadowVP = view * proj;

	ITextureView* dsv = skyOccDepth->GetDefaultView(TEXTURE_VIEW_DEPTH_STENCIL);
	context->SetRenderTargets(0, nullptr, dsv, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	context->ClearDepthStencil(dsv, CLEAR_DEPTH_FLAG, 1.f, 0, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
	Viewport vp; vp.TopLeftX = 0; vp.TopLeftY = 0;
	vp.Width = (float)kSkyOccN; vp.Height = (float)kSkyOccN; vp.MinDepth = 0; vp.MaxDepth = 1;
	context->SetViewports(1, &vp, kSkyOccN, kSkyOccN);
	submitOpaques();
	context->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE);

	const float range = kSkyOccFar - kSkyOccNear;
	const float p[8] = { ox, oz, 1.0f / size, 1.0f, eye.y, kSkyOccNear, range, kSkyOccBias / range };
	writeCB(p); skyOccCBZero = false;
	skyOccSRV = skyOccDepth->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
}
