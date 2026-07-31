# NukeRenderDiligent

The main renderer of [NukeEngine](https://github.com/Luastris/NukeEngine-Eco), built on
[Diligent Engine](https://github.com/DiligentGraphics/DiligentEngine) (vendored in
`deps/`). A render **module**: implements the engine's `iRender` seam and is chosen per
project — the engine core contains zero graphics code.

## Features

- Three backends from one HLSL source: **Vulkan** (the editor's default — native ImGui
  multi-viewport, hardware RT via `VK_KHR_ray_tracing_pipeline`, own SPIR-V disk cache
  `config/shadercache_vk/`), **D3D12** (the packaged-game default — RT, DComp transparency,
  HDR10) and **D3D11** (legacy fallback, no RT). Regular shaders go through glslang on
  Vulkan; SM 6.x shaders compile through the ONE vendored DXC on both backends
  (`dxcompiler.dll`/`dxil.dll` deploy next to the exe) — DXIL for D3D12, SPIR-V for Vulkan.
  Switching the editor off Vulkan is not recommended (see the
  [root README](https://github.com/Luastris/NukeEngine-Eco)).
- PBR metallic-roughness pipeline: full material maps, lights + PCF shadow mapping
  (dir/point/spot), transparency with correct ordering, frustum culling.
- Sky/environment + IBL, reflection probes with box-projection parallax.
- Post stack: MSAA + FXAA + TAA (velocity buffer), bloom, SSR, color grade, vignette,
  custom `.post.hlsl` effects; HDR10 output (player, Windows).
- **Ray-traced reflections** (DXR): traces the real scene and shades hits with the same
  material model as raster. A shader opts into a faithful RT hit via a
  `<name>.surf.hlsl` sidecar — see the [core README](https://github.com/Luastris/NukeEngine)
  for the convention.
- Debug-line pass, per-object id pass (for shaders/picking), UI seam for the editor and
  the runtime GUI, native multi-viewport windows.

## Window modes & transparency

Driven by the engine's `WindowDesc` / `Game.Set*` API (persisted in `config/main.json`):
windowed, borderless-fullscreen and exclusive-fullscreen (video-mode switch) via GLFW;
per-pixel **transparency** via a DirectComposition swap chain. The composition path patches
the vendored `SwapChainD3DBase.hpp` (`CreateSwapChainForComposition` + premultiplied alpha,
guarded by `g_NukeCompositionSwapChain` so the opaque path is byte-for-byte unchanged) and
binds the swap chain into a DComp visual on the HWND. Transparency is a creation-time swap
chain property — it applies on the next launch, not live.

## Gotchas

- New renderer-internal shader pairs must be added to `RendererInternalShader()` in the
  engine's `resdb.cpp`, or a broken material pipeline gets built from them.
- `dxcompiler.dll` + `dxil.dll` are NOT system DLLs — the post-build deploys the vendored
  pair; without them shader compilation falls back to FXC (SM 5.1, no RT).

## Building

Part of the [NukeEngine-Eco](https://github.com/Luastris/NukeEngine-Eco) superbuild, or
standalone: `cmake -S . -B build -G "Visual Studio 17 2022" -A x64` +
`cmake --build build --config Debug` (needs `VCPKG_ROOT`; the engine must be built
first). Building outside a VS dev prompt is supported (the ATL include/lib dirs are
wired in explicitly).
