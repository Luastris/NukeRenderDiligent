#pragma once
#ifndef NUKEDILIGENT_NATIVE_H
#define NUKEDILIGENT_NATIVE_H

// ---------------------------------------------------------------------------------------------
// The NATIVE ESCAPE HATCH of the Diligent renderer: a narrow exported API for modules that
// CHOOSE to be backend-specific and drive their own GPU passes (the water module). This is the
// companion-module pattern (NukeTilemapEditor -> NukeTilemap): the client includes this header
// (plus the vendored DiligentCore headers) and links NukeRenderDiligent's import library.
//
// WHAT LIVES HERE ON PURPOSE (generic renderer capabilities, not water):
//   - the per-camera/per-frame state snapshot (device, targets, matrices, engine resources)
//   - the disk-cached shader/PSO factories (includeEpoch-aware — setShaderSource keys them)
//   - the GPU lifetime manager (Trash) and batch flushing around raw pass work
//   - pipeline hook points (camera begin / post-resolve) for module-owned passes
//   - the RT volume-attenuation input (tiny POD the ray shaders consume)
// Everything ELSE about water — sims, draws, post — lives in NukeWater.
//
// Threading: every call is MAIN-THREAD only (the render thread), same as iRender draws.
// ---------------------------------------------------------------------------------------------

#include <cstdint>

// Bare Diligent interface includes — the consumer adds the DiligentCore interface dirs to
// its include path (see NukeWater/CMakeLists.txt for the canonical list).
#include "RenderDevice.h"
#include "DeviceContext.h"
#include "PipelineState.h"
#include "Shader.h"

#ifdef NUKEDILIGENT_BUILD
#define NUKEDLG_API __declspec(dllexport)
#else
#define NUKEDLG_API __declspec(dllimport)
#endif

namespace nuke { struct NukeLight; }

namespace nukediligent {

// Per-call state snapshot. Pointers are owned by the renderer and valid for the CURRENT
// frame/camera only — never cache them across frames (the device pointer is the exception:
// cache it to DETECT renderer swaps and rebuild your resources).
struct Frame
{
	Diligent::IRenderDevice*  device = nullptr;
	Diligent::IDeviceContext* context = nullptr;
	Diligent::IShaderSourceInputStreamFactory* shaderFactory = nullptr;   // setShaderSource sources

	// Current camera pass (valid while cameraPassActive).
	bool cameraPassActive = false;
	bool camIsEditor = false;
	Diligent::ITextureView* curRTV = nullptr;
	Diligent::ITextureView* curDSV = nullptr;
	int      curRTW = 0, curRTH = 0;
	uint32_t samples = 1;
	Diligent::TEXTURE_FORMAT sceneFmt = Diligent::TEX_FORMAT_UNKNOWN;
	float view[16], proj[16], projNoJitter[16];     // row-major float4x4 payloads
	float camPos[3], camFwd[3];
	float nearZ = 0.1f, farZ = 1000.0f;
	// The scene color for grabs: resolve sceneMSAAColor when non-null, else copy scenePostSRV.
	Diligent::ITexture*     sceneMSAAColor = nullptr;   // null when the scene is single-sample
	Diligent::ITextureView* scenePostSRV = nullptr;     // single-sample HDR scene SRV

	// Frame bookkeeping.
	uint64_t frameId = 0;
	uint32_t passSerial = 0;
	uint64_t curTarget = 0;      // render-target id bound by beginCamera (per-target ownership)

	// Engine-owned resources (may be null — always fall back).
	Diligent::ITextureView* sceneDepthSRV = nullptr;    // single-sample prepass depth (gbuf)
	bool gbufActive = false;
	Diligent::ITextureView* whiteSRV = nullptr;
	Diligent::ITextureView* fallbackCubeSRV = nullptr;
	Diligent::ITextureView* probeCubeSRV = nullptr;     // null when no probe captured
	bool probeActive = false;
	Diligent::ITextureView* shadowSRV = nullptr;        // directional/spot shadow array
	Diligent::IBuffer* worldFrameCB = nullptr;          // FrameCB (lights/sky/probe/wind)
	bool rtSupported = false;

	// Scene lights (renderer's CPU list; valid this frame).
	const nuke::NukeLight* lights = nullptr;
	int lightCount = 0;
};

// Fill `out` with the current state. Returns false before init / after shutdown.
NUKEDLG_API bool GetFrame(Frame& out);

// A pushed shader source by name (setShaderSource map: engine loose shaders + every module's
// pushes). Modules with their own embed tables use this as the fallback for SHARED shaders
// (the fullscreen post.vs and friends). Returns false when the name is unknown.
NUKEDLG_API bool GetShaderSource(const char* name, std::string& out);

// Disk-cached shader compilation (the cache keys on source + includeEpoch, so setShaderSource
// pushes invalidate dependents correctly). Same helpers the renderer uses internally.
NUKEDLG_API void CreateShaderCached(Diligent::ShaderCreateInfo& sci, Diligent::IShader** out);
NUKEDLG_API void CreateGraphicsPSOCached(Diligent::GraphicsPipelineStateCreateInfo& ci,
                                         Diligent::IPipelineState** out);

// GPU lifetime manager: NEVER Release() a live device object inline — hand it here (deferred
// destruction after the GPU is done with it).
NUKEDLG_API void Trash(Diligent::IDeviceObject* obj);

// Flush the renderer's pending sprite batches (call BEFORE unbinding the camera targets for
// raw compute/offscreen work mid-pass — pending batches belong to the pre-water scene).
NUKEDLG_API void FlushBatches();

// Stats + state-cache poke: report `tris` drawn by a module pass and invalidate the
// renderer's instancing bind cache (call after raw SetPipelineState/Draw work).
NUKEDLG_API void NoteDraw(int tris);

// Post scratch RTV (HDR, single-sample, screen-sized): index 0/1 ping-pong shared with the
// post chain — valid for module passes that run INSIDE the hook points below.
NUKEDLG_API Diligent::ITextureView* GetPostScratchRTV(int idx, int w, int h);

// Pipeline hook points. `user` is passed back verbatim. Pass null to unregister.
//   onCameraBegin: start of every camera pass (reset per-camera module state).
//   onCameraPost:  after the MSAA resolve, before the user post chain — return a replacement
//                  scene SRV (your pass output) or null to leave the scene untouched.
struct WaterHooks
{
	void* user = nullptr;
	void (*onCameraBegin)(void* user) = nullptr;
	Diligent::ITextureView* (*onCameraPost)(void* user, Diligent::ITextureView* scene) = nullptr;
};
NUKEDLG_API void SetWaterHooks(const WaterHooks* hooks);

// RT volume attenuation input (the ray shaders swallow radiance crossing a water surface):
// level = world Y, on = 0/1 this frame, fade = 1/opacityDepth, scatter/absorb per channel.
NUKEDLG_API void SetRTWaterState(float level, float on, float fade,
                                 const float scatter[3], const float absorb[3]);

}  // namespace nukediligent

#endif // NUKEDILIGENT_NATIVE_H
