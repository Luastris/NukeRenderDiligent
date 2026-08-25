#pragma once
#ifndef NUKEDILIGENT_NATIVE_H
#define NUKEDILIGENT_NATIVE_H

// Native escape hatch of the Diligent renderer: a narrow exported API for backend-specific
// modules that drive their own GPU passes. Link NukeRenderDiligent's import library.
// All calls are main-thread (render thread) only.

#include <cstdint>
#include <boost/function.hpp>   // EnqueueBuild callbacks

// The consumer must add the DiligentCore interface dirs to its include path.
#include "RenderDevice.h"
#include "DeviceContext.h"
#include "PipelineState.h"
#include "Shader.h"

#ifdef _WIN32
  #ifdef NUKEDILIGENT_BUILD
  #define NUKEDLG_API __declspec(dllexport)
  #else
  #define NUKEDLG_API __declspec(dllimport)
  #endif
#else
  #define NUKEDLG_API __attribute__((visibility("default")))
#endif

namespace nuke { struct NukeLight; }

namespace nukediligent {

// Per-call state snapshot. Renderer-owned pointers, valid for the current frame/camera only —
// never cache them (except `device`, cached to detect a renderer swap).
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

	// Active debug view (iRender::setDebugView): 0 = off, 1 = mesh cost. A module whose pass
	// draws visible geometry should render a cost proxy instead (DrawCostProxy). ABI: appended.
	int debugView = 0;
};

// Fill `out` with the current state. Returns false before init / after shutdown.
NUKEDLG_API bool GetFrame(Frame& out);

// Look up a pushed shader source by name (engine loose shaders + module pushes); the fallback
// for shared shaders such as post.vs. Returns false when the name is unknown.
NUKEDLG_API bool GetShaderSource(const char* name, std::string& out);

// Disk-cached shader/PSO compilation. Cache keys on source + includeEpoch, so setShaderSource
// pushes invalidate dependents.
// Background build: `build` runs on the renderer's builder thread (device-object creation
// ONLY — never the device context), `adopt` on the render thread once it finished. Modules
// keep their draws gated on a flag that `adopt` flips, so a cold start never freezes a frame.
// `prio`: lower runs sooner (renderer: 0 = world base pipes, 5 = G-buffer, 10 = blend variants,
// 12 = module default, 20 = extras, 25 = RT, 30 = wireframe). `name` shows in the status bar.
NUKEDLG_API void EnqueueBuild(const boost::function<void()>& build, const boost::function<void()>& adopt,
                              int prio = 12, const char* name = "");
NUKEDLG_API void CreateShaderCached(Diligent::ShaderCreateInfo& sci, Diligent::IShader** out);
NUKEDLG_API void CreateGraphicsPSOCached(Diligent::GraphicsPipelineStateCreateInfo& ci,
                                         Diligent::IPipelineState** out);

// Pipeline warm-up. Register every pipeline builder here instead of compiling on the draw path:
// the renderer runs the pending ones once per frame under a shared time budget, so a cold shader
// cache costs a few frames of a missing effect rather than one long freeze, and it re-arms them
// when the sample count or the scene format changes. `fn` returns true when its pipelines are
// ready and false to be called again next frame; it must be able to stop half-way and resume.
// Unregister before the owner dies.
typedef bool (*WarmupFn)(void* user);
NUKEDLG_API void AddPipelineWarmup(const char* name, WarmupFn fn, void* user);
// Ask for a finished builder to be called again — a feature turned on at runtime needs
// pipelines that were not worth compiling before.
NUKEDLG_API void RearmPipelineWarmup(void* user);
NUKEDLG_API void RemovePipelineWarmup(void* user);

// Deferred destruction. Never Release() a live device object inline — hand it here.
NUKEDLG_API void Trash(Diligent::IDeviceObject* obj);

// Flush pending sprite batches. Call before unbinding the camera targets for raw
// compute/offscreen work mid-pass.
NUKEDLG_API void FlushBatches();

// Report `tris` drawn by a module pass and invalidate the instancing bind cache; call after
// raw SetPipelineState/Draw work.
NUKEDLG_API void NoteDraw(int tris);

// Post scratch RTV (HDR, single-sample, screen-sized), index 0/1 ping-pong shared with the post
// chain. Only valid for module passes running inside the hook points below.
NUKEDLG_API Diligent::ITextureView* GetPostScratchRTV(int idx, int w, int h);

// Pipeline hook points; `user` is passed back verbatim. onCameraBegin runs at the start of every
// camera pass; onCameraPost runs after the MSAA resolve and returns a replacement scene SRV or null.
struct WaterHooks
{
	void* user = nullptr;
	void (*onCameraBegin)(void* user) = nullptr;
	Diligent::ITextureView* (*onCameraPost)(void* user, Diligent::ITextureView* scene) = nullptr;
};
NUKEDLG_API void SetWaterHooks(const WaterHooks* hooks);

// RT volume attenuation input consumed by the ray shaders: level = world Y, on = 0/1 this
// frame, fade = 1/opacityDepth, scatter/absorb per channel.
NUKEDLG_API void SetRTWaterState(float level, float on, float fade,
                                 const float scatter[3], const float absorb[3]);

// Mesh-cost debug view (Frame.debugView == 1): draw a translucent box at pos/quat/size,
// colored by `tris` on the cost ramp — the stand-in for a module pass's own geometry.
NUKEDLG_API void DrawCostProxy(const float pos[3], const float quat[4], const float size[3], double tris);

}  // namespace nukediligent

#endif // NUKEDILIGENT_NATIVE_H
