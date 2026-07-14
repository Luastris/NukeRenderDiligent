cmake_minimum_required(VERSION 3.20)
project(NukeRenderDiligent CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Match the rest of the toolchain: dynamic debug CRT (/MDd in Debug).
cmake_policy(SET CMP0091 NEW)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

# --- Diligent Engine (built as a subproject; only what we need) ---
# Pre-seed Diligent's backend try_compile results: the atlbase.h probe only passes from a
# VS dev prompt (a stray older toolset without ATL wins the probe otherwise), while the
# ACTUAL build resolves ATL fine through the vcxproj default include path — this project
# is Windows-only with the C++ workload, D3D11/12 + ATL are a given.
set(HAS_D3D11 TRUE CACHE INTERNAL "")
set(HAS_D3D12 TRUE CACHE INTERNAL "")
set(HAS_ATL   TRUE CACHE INTERNAL "")
# ...and hand every Diligent target the ATL include dir explicitly: outside a VS dev
# prompt the generated vcxprojs carry no ATLMFC path, and Diligent's Win32 platform code
# includes atlbase.h (the newest installed toolset wins).
file(GLOB _nuke_atl_dirs "$ENV{ProgramFiles}/Microsoft Visual Studio/2022/*/VC/Tools/MSVC/*/atlmfc/include")
if(_nuke_atl_dirs)
    list(SORT _nuke_atl_dirs)
    list(REVERSE _nuke_atl_dirs)
    list(GET _nuke_atl_dirs 0 _nuke_atl)
    include_directories("${_nuke_atl}")   # directory-scoped: inherited by the subproject
    string(REPLACE "/include" "/lib/x64" _nuke_atl_lib "${_nuke_atl}")
    link_directories("${_nuke_atl_lib}")  # atls.lib for the D3D backends
    message(STATUS "ATL include: ${_nuke_atl}")
endif()
set(DILIGENT_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
set(DILIGENT_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
set(DILIGENT_NO_VULKAN     ON  CACHE BOOL "" FORCE)
set(DILIGENT_NO_OPENGL     ON  CACHE BOOL "" FORCE)
set(DILIGENT_NO_WEBGPU     ON  CACHE BOOL "" FORCE)
set(DILIGENT_NO_ARCHIVER   ON  CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/DiligentEngine DiligentEngine)

set(VCPKG_INSTALLED "$ENV{VCPKG_ROOT}/installed/x64-windows")   # VCPKG_ROOT env var, not a hardcoded path

add_library(NukeRenderDiligent SHARED
    src/NukeDiligent.cpp
    src/NukeDiligent_Pipelines.cpp
    src/NukeDiligent_Resources.cpp
    src/NukeDiligent_Scene.cpp
    src/NukeDiligent_Sky.cpp
    src/NukeDiligent_Shadow.cpp
    src/NukeDiligent_Probe.cpp
    src/NukeDiligent_SSR.cpp
    src/NukeDiligent_Post.cpp
    src/NukeDiligent_UI.cpp
    src/NukeDiligent_RT.cpp
    src/NukeDiligent_Sprite.cpp
)

target_compile_definitions(NukeRenderDiligent PRIVATE
    PLATFORM_WIN32=1
    GLM_ENABLE_EXPERIMENTAL
    NOMINMAX
    GLFW_EXPOSE_NATIVE_WIN32
    _CRT_SECURE_NO_WARNINGS
)

target_include_directories(NukeRenderDiligent PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../NukeEngine/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../NukeEngine
    ${VCPKG_INSTALLED}/include
)

# glfw import lib (vcpkg). Debug build links the debug import lib.
target_link_directories(NukeRenderDiligent PRIVATE
    "${VCPKG_INSTALLED}/$<$<CONFIG:Debug>:debug/>lib"
)

target_link_libraries(NukeRenderDiligent PRIVATE
    Diligent-GraphicsEngineD3D11-static
    Diligent-GraphicsEngineD3D12-static
    Diligent-GraphicsTools
    Diligent-Common
    Diligent-GraphicsAccessories
    Diligent-BasicPlatform
    Diligent-Win32Platform
    Diligent-Primitives
    glfw3dll
    d3d11 d3d12 dxgi d3dcompiler   # d3d12: D3D12GetDebugInterface (DRED enable)
    shell32 user32   # ExtractIconEx + SendMessage(WM_SETICON) for the window icon
    dwmapi           # DwmSetWindowAttribute — dark title bar
    dcomp            # DirectComposition — per-pixel window transparency (Game.SetTransparent)
)

# Deploy the module DLL next to the editor, into modules/.
set(MODULES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../NukeEngine/x64/$<CONFIG>/modules")
set(EDITOR_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/../NukeEngine/x64/$<CONFIG>")
add_custom_command(TARGET NukeRenderDiligent POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${MODULES_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:NukeRenderDiligent>" "${MODULES_DIR}/"
    # glfw3.dll is this module's runtime dependency — deploy it next to the exe.
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${VCPKG_INSTALLED}/$<$<CONFIG:Debug>:debug/>bin/glfw3.dll" "${EDITOR_DIR}/"
    # DXC (dxcompiler.dll + dxil.dll): Diligent LoadLibrary's dxcompiler.dll by bare name at
    # runtime to compile SM6.5 shaders (RayTracingAccelerationStructure etc). It is NOT a system
    # DLL and NOT on the default search path, so it must sit next to the exe. Vendored (deps/dxc)
    # so shipped builds are self-sufficient — same DLL for Debug/Release (config-agnostic tool).
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/deps/dxc/dxcompiler.dll" "${EDITOR_DIR}/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/deps/dxc/dxil.dll" "${EDITOR_DIR}/"
    COMMENT "Deploying NukeRenderDiligent.dll + glfw3.dll + DXC (dxcompiler/dxil)"
)
