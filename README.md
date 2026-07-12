# NukeRenderDiligent

The main renderer of [NukeEngine](https://github.com/Luastris/NukeEngine-Eco), built on
[Diligent Engine](https://github.com/DiligentGraphics/DiligentEngine) (vendored in
`deps/`). A render **module**: implements the engine's `iRender` seam and is chosen per
project — the engine core contains zero graphics code.

## Features

- D3D12 (D3D11 fallback), HLSL compiled at runtime through DXC (SM 6.5 — the vendored
  `dxcompiler.dll`/`dxil.dll` deploy next to the exe).
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
