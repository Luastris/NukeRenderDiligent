# DiligentCore local patch set (NUKE PATCH)

The DiligentCore submodule stays PRISTINE in git; every local fix lives in
`DiligentCore.patch` and is applied at CONFIGURE time by this module's CMakeLists
(idempotent: an already-patched tree is detected and left alone). A fresh clone
therefore builds with the patches without any manual step.

**Base commit:** `42153fe22787a66c7152f25ade69acc5625d44b9` (API256019-2-g42153fe22)

What the set contains (all marked `NUKE PATCH` in the code):
- `CMakeLists.txt` — the HAS_D3D11/D3D12/ATL `try_compile` probes honor pre-seeded cache
  values (outside a VS dev prompt the ATL probe false-fails and silently drops the D3D
  backends).
- `SwapChainD3D12Impl.cpp` + `SwapChainD3DBase.{hpp,cpp}` — composition-swapchain support
  (per-pixel window transparency through DirectComposition).
- `CommandContext.cpp` (D3D12) — device-removal diagnostics: the actual HRESULT + DRED
  reason logged before the assert.
- `CommandQueueVkImpl.cpp` (Vulkan) — the actual VkResult named before the submit assert.
- `DeviceContextVkImpl.cpp` (Vulkan) — empty TLAS (0 instances, spec-legal) skips the
  instance upload: the zero-size copy dereferenced a null block inside the NVIDIA driver.

## Upgrading DiligentCore
1. Move the submodule to the new commit with a CLEAN tree (the configure step will warn
   "neither applies nor is applied" if the patch no longer fits).
2. Re-apply what still applies: `git -C deps/DiligentEngine/DiligentCore apply ../../..' \
   '/patches/DiligentCore.patch`, hand-fix rejects.
3. Regenerate: `git -C deps/DiligentEngine/DiligentCore diff > patches/DiligentCore.patch`
   and update the base commit above.
