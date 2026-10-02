cmake_minimum_required(VERSION 3.20)
project(NukeRenderDiligent CXX)

include("${CMAKE_CURRENT_SOURCE_DIR}/../NukeEngine/cmake/NukeRunDir.cmake")   # NUKE_RUN_SUBDIR: x64 / macos / linux

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Match the rest of the toolchain: dynamic debug CRT (/MDd in Debug).
cmake_policy(SET CMP0091 NEW)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

# --- Diligent Engine (built as a subproject; only what we need) ---
if(WIN32)
    # Pre-seed Diligent's backend try_compile results: the atlbase.h probe only passes from a
    # VS dev prompt (a stray older toolset without ATL wins the probe otherwise), while the
    # ACTUAL build resolves ATL fine through the vcxproj default include path — on Windows
    # the C++ workload makes D3D11/12 + ATL a given.
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
endif()
# ---- 4.2 vendor technologies built in: NUKE_TECH ------------------------------------------------
# NUKE_TECH = All (default), None, or a list of DLSS;FSR;XeSS (comma or semicolon, any case): which
# vendor upscaling / frame-generation SDKs are fetched, compiled against and deployed. The same
# switch as Package Project's "Vendor Tech" and stage_release's -Tech; a technology left out is
# not offered by the renderer (FSR 1 always is). Superbuild: -DNUKE_TECH=DLSS,FSR
set(NUKE_TECH "All" CACHE STRING "Vendor upscaling / frame-generation SDKs to build in: All, None, or a list of DLSS;FSR;XeSS")
set(NUKE_TECH_DLSS OFF)
set(NUKE_TECH_FSR  OFF)
set(NUKE_TECH_XESS OFF)
string(REPLACE "," ";" _nuke_tech_list "${NUKE_TECH}")
foreach(_t IN LISTS _nuke_tech_list)
    string(STRIP "${_t}" _t)
    string(TOLOWER "${_t}" _t)
    if(_t STREQUAL "all")
        set(NUKE_TECH_DLSS ON)
        set(NUKE_TECH_FSR  ON)
        set(NUKE_TECH_XESS ON)
    elseif(_t STREQUAL "none")
        set(NUKE_TECH_DLSS OFF)
        set(NUKE_TECH_FSR  OFF)
        set(NUKE_TECH_XESS OFF)
    elseif(_t STREQUAL "dlss")
        set(NUKE_TECH_DLSS ON)
    elseif(_t STREQUAL "fsr")
        set(NUKE_TECH_FSR ON)
    elseif(_t STREQUAL "xess")
        set(NUKE_TECH_XESS ON)
    elseif(NOT _t STREQUAL "")
        message(FATAL_ERROR "NUKE_TECH: unknown technology '${_t}' (All, None, DLSS, FSR, XeSS)")
    endif()
endforeach()
message(STATUS "NukeRenderDiligent: vendor tech DLSS=${NUKE_TECH_DLSS} FSR=${NUKE_TECH_FSR} XeSS=${NUKE_TECH_XESS} (NUKE_TECH=${NUKE_TECH})")
# DLSS super resolution rides Diligent's SuperResolution (NGX): off = no DLSS-Headers fetch, no NGX.
if(NUKE_TECH_DLSS)
    set(DILIGENT_NO_DLSS OFF CACHE BOOL "" FORCE)
else()
    set(DILIGENT_NO_DLSS ON  CACHE BOOL "" FORCE)
endif()

set(DILIGENT_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
set(DILIGENT_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
set(DILIGENT_NO_VULKAN     OFF CACHE BOOL "" FORCE)   # Vulkan backend (task #138): editor default; DX stays for the runtime
set(DILIGENT_NO_OPENGL     ON  CACHE BOOL "" FORCE)
set(DILIGENT_NO_WEBGPU     ON  CACHE BOOL "" FORCE)
set(DILIGENT_NO_ARCHIVER   ON  CACHE BOOL "" FORCE)
# 4.2 super resolution: Diligent's SuperResolution (DLSS via NGX + FSR 1) links statically; the
# vendor runtime (nvngx_dlss.dll) is a separate DLL next to the host. DirectSR stays off: it lives
# only in PREVIEW Agility SDKs (developer mode on the player's PC) - not shippable.
set(DILIGENT_NO_DSR        ON  CACHE BOOL "" FORCE)
if(APPLE)
    # Vulkan-on-Metal via MoltenVK; the native Metal backend sources are not in the public
    # DiligentCore checkout, so keep that backend off explicitly.
    set(DILIGENT_NO_METAL  ON  CACHE BOOL "" FORCE)
endif()

# ---- DiligentCore local patch set (see patches/README.md) --------------------------------
# The submodule stays pristine in git; the NUKE fixes (DComp swapchain, empty-TLAS crash,
# device-removal diagnostics, try_compile pre-seed) apply here, at configure time, before
# the subdirectory is entered. Idempotent: an already-patched tree is detected via the
# reverse-check and left alone — the same configure works on a dirty dev tree and on a
# fresh clone. Same pattern as the root superbuild's GLFW vcpkg overlay.
set(_nuke_dc_dir   "${CMAKE_CURRENT_SOURCE_DIR}/deps/DiligentEngine/DiligentCore")
set(_nuke_dc_patch "${CMAKE_CURRENT_SOURCE_DIR}/patches/DiligentCore.patch")
if(EXISTS "${_nuke_dc_patch}")
    find_package(Git QUIET)
    if(Git_FOUND)
        execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${_nuke_dc_dir}" apply --check "${_nuke_dc_patch}"
                        RESULT_VARIABLE _nuke_dc_applies OUTPUT_QUIET ERROR_QUIET)
        if(_nuke_dc_applies EQUAL 0)
            execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${_nuke_dc_dir}" apply "${_nuke_dc_patch}"
                            RESULT_VARIABLE _nuke_dc_rc)
            if(_nuke_dc_rc EQUAL 0)
                message(STATUS "DiligentCore: NUKE patch set applied")
            else()
                message(FATAL_ERROR "DiligentCore: patch passed --check but failed to apply — inspect patches/DiligentCore.patch")
            endif()
        else()
            execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${_nuke_dc_dir}" apply --check --reverse "${_nuke_dc_patch}"
                            RESULT_VARIABLE _nuke_dc_applied OUTPUT_QUIET ERROR_QUIET)
            if(_nuke_dc_applied EQUAL 0)
                message(STATUS "DiligentCore: NUKE patch set already applied")
            else()
                # Neither cleanly applies nor is fully applied: the submodule moved or the
                # tree was hand-edited. Building an UNPATCHED renderer silently would ship
                # the empty-TLAS crash back in — refuse loudly instead.
                message(FATAL_ERROR "DiligentCore: patches/DiligentCore.patch neither applies nor is applied — rebase it (patches/README.md, 'Upgrading DiligentCore')")
            endif()
        endif()
    else()
        message(WARNING "git not found: cannot apply DiligentCore patches — a pristine checkout builds WITHOUT the NUKE fixes")
    endif()
endif()

# ---- addendum patch: Vulkan WSI alpha compositing (transparent windows off Windows) -----
# Kept SEPARATE from DiligentCore.patch so an already-patched Windows tree takes it without
# a rebase (it touches only SwapChainVkImpl.cpp, which the base set never modifies).
set(_nuke_dc_patch2 "${CMAKE_CURRENT_SOURCE_DIR}/patches/DiligentCore-vk-alpha.patch")
if(EXISTS "${_nuke_dc_patch2}" AND Git_FOUND)
    execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${_nuke_dc_dir}" apply --check "${_nuke_dc_patch2}"
                    RESULT_VARIABLE _nuke_dc2_applies OUTPUT_QUIET ERROR_QUIET)
    if(_nuke_dc2_applies EQUAL 0)
        execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${_nuke_dc_dir}" apply "${_nuke_dc_patch2}")
        message(STATUS "DiligentCore: NUKE vk-alpha patch applied")
    else()
        execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${_nuke_dc_dir}" apply --check --reverse "${_nuke_dc_patch2}"
                        RESULT_VARIABLE _nuke_dc2_applied OUTPUT_QUIET ERROR_QUIET)
        if(_nuke_dc2_applied EQUAL 0)
            message(STATUS "DiligentCore: NUKE vk-alpha patch already applied")
        else()
            message(FATAL_ERROR "DiligentCore: patches/DiligentCore-vk-alpha.patch neither applies nor is applied — rebase it")
        endif()
    endif()
endif()

# ---- addendum patch: no per-pipeline SPIR-V legalization (Vulkan PSO creation cost) --------
# Diligent re-ran spirv-opt's legalization passes over every stage's SPIR-V at EVERY pipeline
# creation (PipelineStateVkImpl::RemapOrVerifyShaderResources) — 0.3-0.5 s per pipeline on the
# big world shaders, immune to every cache. Shaders are legalized once at creation (or by DXC);
# the per-pipeline pass keeps only the reflection strip.
set(_nuke_dc_patch3 "${CMAKE_CURRENT_SOURCE_DIR}/patches/DiligentCore-pso-legalize.patch")
if(EXISTS "${_nuke_dc_patch3}" AND Git_FOUND)
    execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${_nuke_dc_dir}" apply --check "${_nuke_dc_patch3}"
                    RESULT_VARIABLE _nuke_dc3_applies OUTPUT_QUIET ERROR_QUIET)
    if(_nuke_dc3_applies EQUAL 0)
        execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${_nuke_dc_dir}" apply "${_nuke_dc_patch3}")
        message(STATUS "DiligentCore: NUKE pso-legalize patch applied")
    else()
        execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${_nuke_dc_dir}" apply --check --reverse "${_nuke_dc_patch3}"
                        RESULT_VARIABLE _nuke_dc3_applied OUTPUT_QUIET ERROR_QUIET)
        if(_nuke_dc3_applied EQUAL 0)
            message(STATUS "DiligentCore: NUKE pso-legalize patch already applied")
        else()
            message(FATAL_ERROR "DiligentCore: patches/DiligentCore-pso-legalize.patch neither applies nor is applied — rebase it")
        endif()
    endif()
endif()

# ---- addendum patch: D3D12 swap chain attach / detach (4.2 frame generation) ------------------
# ISwapChainD3D12::AttachDXGISwapChain / DetachDXGISwapChain: a vendor frame-generation proxy
# (FSR FG, DLSS-G, XeSS-FG) replaces the DXGI swap chain under the live Diligent one.
set(_nuke_dc_patch4 "${CMAKE_CURRENT_SOURCE_DIR}/patches/DiligentCore-fg-swapchain.patch")
if(EXISTS "${_nuke_dc_patch4}" AND Git_FOUND)
    execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${_nuke_dc_dir}" apply --check "${_nuke_dc_patch4}"
                    RESULT_VARIABLE _nuke_dc4_applies OUTPUT_QUIET ERROR_QUIET)
    if(_nuke_dc4_applies EQUAL 0)
        execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${_nuke_dc_dir}" apply "${_nuke_dc_patch4}")
        message(STATUS "DiligentCore: NUKE fg-swapchain patch applied")
    else()
        execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${_nuke_dc_dir}" apply --check --reverse "${_nuke_dc_patch4}"
                        RESULT_VARIABLE _nuke_dc4_applied OUTPUT_QUIET ERROR_QUIET)
        if(_nuke_dc4_applied EQUAL 0)
            message(STATUS "DiligentCore: NUKE fg-swapchain patch already applied")
        else()
            message(FATAL_ERROR "DiligentCore: patches/DiligentCore-fg-swapchain.patch neither applies nor is applied — rebase it")
        endif()
    endif()
endif()

# ---- addendum patch: Vulkan creation hooks + swap chain attach / detach (4.2 frame generation) --
# g_NukeVkCreateInstance / g_NukeVkCreateDevice route instance and device creation through the
# host (Streamline's proxies, the extra queues AMD's frame-interpolation swap chain wants);
# ISwapChainVk::DetachVkSwapChain / AttachVkSwapChain / GetVkSwapChainCreateInfo mirror the D3D12 ones.
set(_nuke_dc_patch5 "${CMAKE_CURRENT_SOURCE_DIR}/patches/DiligentCore-vk-hooks.patch")
if(EXISTS "${_nuke_dc_patch5}" AND Git_FOUND)
    execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${_nuke_dc_dir}" apply --check "${_nuke_dc_patch5}"
                    RESULT_VARIABLE _nuke_dc5_applies OUTPUT_QUIET ERROR_QUIET)
    if(_nuke_dc5_applies EQUAL 0)
        execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${_nuke_dc_dir}" apply "${_nuke_dc_patch5}")
        message(STATUS "DiligentCore: NUKE vk-hooks patch applied")
    else()
        execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${_nuke_dc_dir}" apply --check --reverse "${_nuke_dc_patch5}"
                        RESULT_VARIABLE _nuke_dc5_applied OUTPUT_QUIET ERROR_QUIET)
        if(_nuke_dc5_applied EQUAL 0)
            message(STATUS "DiligentCore: NUKE vk-hooks patch already applied")
        else()
            message(FATAL_ERROR "DiligentCore: patches/DiligentCore-vk-hooks.patch neither applies nor is applied — rebase it")
        endif()
    endif()
endif()

# Third-party code is OPTIMIZED even in the Debug configuration. Diligent's Vulkan backend
# re-patches every shader's SPIR-V (SPIRV-Tools) and validates resources at EACH pipeline
# creation; unoptimized that was 0.4-1 s per pipeline = ~20 s of frozen startup. The debug
# CRT (/MDd) and the asserts stay — only /Od and /RTC1 go — so nothing mixes CRTs with the
# renderer's own Debug objects.
if(MSVC)
    set(_nuke_saved_cxx_debug "${CMAKE_CXX_FLAGS_DEBUG}")
    set(_nuke_saved_c_debug   "${CMAKE_C_FLAGS_DEBUG}")
    set(CMAKE_CXX_FLAGS_DEBUG "/MDd /Zi /O2 /Ob2")
    set(CMAKE_C_FLAGS_DEBUG   "/MDd /Zi /O2 /Ob2")
endif()
# The Windows dev tree vendors the full DiligentEngine meta-repo; a fresh (macOS/Linux)
# checkout needs only DiligentCore — include whichever is present.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/deps/DiligentEngine/CMakeLists.txt")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/DiligentEngine DiligentEngine)
else()
    add_subdirectory(${_nuke_dc_dir} DiligentCore)
endif()
if(MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${_nuke_saved_cxx_debug}")
    set(CMAKE_C_FLAGS_DEBUG   "${_nuke_saved_c_debug}")
endif()
# DiligentCore's combined mega-archive (its "ar -M" script) chokes on universal fat .a
# files, and this project links the individual Diligent-* targets anyway — drop it from ALL.
if(TARGET DiligentCore-static)
    set_target_properties(DiligentCore-static PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()

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
    src/NukeDiligent_AO.cpp
    src/NukeDiligent_GI.cpp
    src/NukeDiligent_SSGI.cpp
    src/NukeDiligent_Vol.cpp
    src/NukeDiligent_Clouds.cpp
    src/NukeDiligent_Lens.cpp
    src/NukeDiligent_Panini.cpp
    src/NukeDiligent_Reload.cpp
    src/NukeDiligent_Gpu.cpp
    src/NukeDiligent_SkyOcc.cpp
    src/NukeDiligent_Trails.cpp
    src/NukeDiligent_SkyMap.cpp
    src/NukeDiligent_Atmo.cpp
    src/NukeDiligent_Post.cpp
    src/NukeDiligent_PostFX.cpp
    src/NukeDiligent_Upscale.cpp
    src/NukeDiligent_FrameGen.cpp
    src/NukeDiligent_FrameGenVk.cpp
    src/NukeDiligent_UI.cpp
    src/NukeDiligent_RT.cpp
    src/NukeDiligent_Sprite.cpp
    src/NukeDiligent_Decal.cpp
    src/NukeDiligent_Profile.cpp
    src/NukeDiligent_Native.cpp
    src/NukeDiligent_Occlusion.cpp
    src/NukeDiligent_Storage.cpp
)
if(APPLE)
    target_sources(NukeRenderDiligent PRIVATE src/NukeDiligent_Cocoa.mm)   # CAMetalLayer shim
endif()

target_include_directories(NukeRenderDiligent PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../NukeEngine/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../NukeEngine
)

if(WIN32)
    target_compile_definitions(NukeRenderDiligent PRIVATE
        PLATFORM_WIN32=1
        GLM_ENABLE_EXPERIMENTAL
        NOMINMAX
        GLFW_EXPOSE_NATIVE_WIN32
        _CRT_SECURE_NO_WARNINGS
        NUKEDILIGENT_BUILD          # dllexport side of NukeDiligentNative.h
        $<$<TARGET_EXISTS:DLSS-Headers>:NUKE_DLSS_NGX=1>   # NGX SDK present in this tree
    )

    target_include_directories(NukeRenderDiligent PRIVATE ${VCPKG_INSTALLED}/include)

    # glfw import lib (vcpkg). Debug build links the debug import lib.
    # + NukeEngine.lib: Mesh v4 helpers (TriCount/Section/Lod/EnsureBounds) are exported class
    # members — the renderer now links the engine import lib like every other module.
    target_link_directories(NukeRenderDiligent PRIVATE
        "${VCPKG_INSTALLED}/$<$<CONFIG:Debug>:debug/>lib"
        "${CMAKE_CURRENT_SOURCE_DIR}/../NukeEngine/${NUKE_RUN_SUBDIR}/$<CONFIG>"
    )

    target_link_libraries(NukeRenderDiligent PRIVATE
        NukeEngine
        Diligent-GraphicsEngineD3D11-static
        Diligent-GraphicsEngineD3D12-static
        Diligent-GraphicsEngineVk-static
        Diligent-SuperResolution-static   # 4.2: DLSS (NGX loader lib inside) + FSR 1
        $<$<TARGET_EXISTS:DLSS-Headers>:DLSS-Headers>   # NGX headers: the Vulkan device asks NGX for its extensions
        $<$<TARGET_EXISTS:Vulkan::Headers>:Vulkan::Headers>   # (their Vulkan types)
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
        version          # GetFileVersionInfo — DirectStorage runtime build in the self-test verdict key
    )
else()
    # Deuterium (macOS/Linux): Vulkan is the only backend; deps come from the vcpkg toolchain,
    # the engine is the NukeEngine CMake target from the same superbuild.
    target_compile_definitions(NukeRenderDiligent PRIVATE
        $<$<PLATFORM_ID:Darwin>:PLATFORM_MACOS=1>
        $<$<PLATFORM_ID:Linux>:PLATFORM_LINUX=1>
        GLM_ENABLE_EXPERIMENTAL
        NUKEDILIGENT_BUILD
    )

    find_package(glm CONFIG REQUIRED)
    find_package(Boost REQUIRED COMPONENTS filesystem system)
    find_package(nlohmann_json CONFIG REQUIRED)
    if(NOT APPLE)
        # Window-icon PNG decode (stb_image, header-only): X11 icons are files, not resources.
        find_path(STB_INCLUDE_DIRS "stb_image.h" REQUIRED)
        target_include_directories(NukeRenderDiligent PRIVATE ${STB_INCLUDE_DIRS})
    endif()

    target_link_libraries(NukeRenderDiligent PRIVATE
        NukeEngine
        Diligent-GraphicsEngineVk-static
        Diligent-GraphicsTools
        Diligent-Common
        Diligent-GraphicsAccessories
        Diligent-BasicPlatform
        Diligent-Primitives
        glm::glm
        Boost::filesystem Boost::system
        nlohmann_json::nlohmann_json
    )
    # ONE shared GLFW across renderer/ImGui/gamepad (superbuild provides it); static fallback
    # for standalone configures — beware objc GLFW* class duplication with other modules.
    if(NUKE_GLFW_DYLIB)
        target_link_libraries(NukeRenderDiligent PRIVATE "${NUKE_GLFW_DYLIB}")
        target_include_directories(NukeRenderDiligent BEFORE PRIVATE "${NUKE_GLFW_INCLUDE}")
    else()
        find_package(glfw3 CONFIG REQUIRED)
        target_link_libraries(NukeRenderDiligent PRIVATE glfw)
    endif()
    if(APPLE)
        target_link_libraries(NukeRenderDiligent PRIVATE
            Diligent-ApplePlatform
            "-framework AppKit" "-framework QuartzCore")
    else()
        target_link_libraries(NukeRenderDiligent PRIVATE Diligent-LinuxPlatform)
    endif()

    target_compile_options(NukeRenderDiligent PRIVATE
        -Wno-c++11-narrowing -Wno-unknown-attributes)

    set(_nuke_rd_rpath "@loader_path/..;$ORIGIN/..")   # libNukeEngine lives one level up (run dir)
    if(APPLE AND EXISTS "/opt/homebrew/lib/libvulkan.dylib")
        # volk dlopen's libvulkan.dylib at runtime; Homebrew's Vulkan loader (+ MoltenVK ICD)
        # is the dev-machine source — reachable through rpath, no copy into the run dir.
        list(APPEND _nuke_rd_rpath "/opt/homebrew/lib")
    endif()
    if(NUKE_GLFW_LIBDIR)
        list(APPEND _nuke_rd_rpath "${NUKE_GLFW_LIBDIR}")   # @rpath/libglfw.3.5.dylib
    endif()
    set_target_properties(NukeRenderDiligent PROPERTIES
        PREFIX ""                       # module loader matches by stem: NukeRenderDiligent.dylib
        BUILD_RPATH "${_nuke_rd_rpath}")
endif()

# ---- vendored Linux DXC (SM6.x shaders + ray tracing off Windows) ------------------------
# The exact same arrangement as the Windows deps/dxc DLLs and the macOS MoltenVK download:
# official Microsoft release, pinned by hash, fetched once at configure time. Deployed next
# to the hosts below; NukeDiligent.cpp points EngineVkCreateInfo::pDxCompilerPath at it and
# ray tracing gates on its presence in the run root.
if(UNIX AND NOT APPLE)
    set(_nuke_dxc_dir "${CMAKE_CURRENT_SOURCE_DIR}/deps/dxc-linux")
    set(_nuke_dxc     "${_nuke_dxc_dir}/libdxcompiler.so")
    if(NOT EXISTS "${_nuke_dxc}")
        set(_nuke_dxc_url "https://github.com/microsoft/DirectXShaderCompiler/releases/download/v1.9.2607/linux_dxc_2026_07_29.x86_x64.tar.gz")
        set(_nuke_dxc_sha "55665c87824051ed4774ff3280a79ccbbb7d39243b9736ca5e98222134112d54")
        message(STATUS "DXC: downloading Linux libdxcompiler (v1.9.2607, official release)...")
        file(DOWNLOAD "${_nuke_dxc_url}" "${CMAKE_BINARY_DIR}/dxc-linux.tar.gz"
             EXPECTED_HASH SHA256=${_nuke_dxc_sha} STATUS _nuke_dxc_st)
        list(GET _nuke_dxc_st 0 _nuke_dxc_rc)
        if(_nuke_dxc_rc EQUAL 0)
            file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/dxc-x")
            execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf "${CMAKE_BINARY_DIR}/dxc-linux.tar.gz"
                            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/dxc-x")
            file(GLOB _nuke_dxc_libs "${CMAKE_BINARY_DIR}/dxc-x/*/lib/libdxcompiler.so"
                                     "${CMAKE_BINARY_DIR}/dxc-x/*/lib/libdxil.so")
            file(MAKE_DIRECTORY "${_nuke_dxc_dir}")
            file(COPY ${_nuke_dxc_libs} DESTINATION "${_nuke_dxc_dir}")
            # Microsoft ships the .so UNSTRIPPED (301 MB of DWARF nobody will ever debug) —
            # strip to the ~24 MB it really is before it rides into every AppImage.
            find_program(_nuke_strip strip)
            if(_nuke_strip)
                execute_process(COMMAND "${_nuke_strip}" -s "${_nuke_dxc_dir}/libdxcompiler.so"
                                                            "${_nuke_dxc_dir}/libdxil.so")
            endif()
            message(STATUS "DXC: vendored at deps/dxc-linux/ (stripped)")
        else()
            message(WARNING "DXC download failed — SM6.x/ray tracing stays off (glslang-only shaders)")
        endif()
    endif()
endif()

# ---- 4.2 vendor upscalers: AMD FidelityFX (FSR 3.1 / FSR 4) and Intel XeSS -----------------------
# Fetched once at configure time from the vendors' GitHub releases, pinned by hash, like the Linux
# DXC. Only the API headers, the licence and the signed runtime DLLs come down; the runtimes deploy
# next to the renderer DLL (modules/) and are loaded by name at run time - absent = the variant is
# not offered. Package Project decides which of them ship.
if(WIN32)
    set(_nuke_ffx_dir  "${CMAKE_CURRENT_SOURCE_DIR}/deps/ffx")
    set(_nuke_ffx_base "https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/v2.3.0")
    set(_nuke_ffx_files
        "Kits/FidelityFX/api/include/ffx_api.h|api/include/ffx_api.h|e1a9d1b559eaf75f4cb401f1c43a66e2f3dc9621abfeb58b9c6fac56443177fd"
        "Kits/FidelityFX/api/include/ffx_api_loader.h|api/include/ffx_api_loader.h|18d1901dbbc881e8131b91c65ad44bb3b4a1de6c31435d5b75236f16727ba80e"
        "Kits/FidelityFX/api/include/ffx_api_types.h|api/include/ffx_api_types.h|cc0adc0ffc804dd7cd03c39c15581b575e416383177ffe95060201d52cc0fc49"
        "Kits/FidelityFX/api/include/dx12/ffx_api_dx12.h|api/include/dx12/ffx_api_dx12.h|c9afbc3c4c673723e9e5369a666f0ff54560543b5a93537f8ff453bf1c3bf530"
        "Kits/FidelityFX/upscalers/include/ffx_upscale.h|upscalers/include/ffx_upscale.h|65be36d5653eb71c10379e0204f7bb7cd6a8aeec96e6f8cffc5f59ffeb1f89bf"
        "Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_dx12.dll|bin/amd_fidelityfx_upscaler_dx12.dll|d0dcccc74a43c44ba435b7a369b456e0970d8a4464e4bd683119b374f2c9fb46"
        # frame generation (FSR 3.1 FG): the effect + its frame-interpolation swap chain, one runtime
        "Kits/FidelityFX/framegeneration/include/ffx_framegeneration.h|framegeneration/include/ffx_framegeneration.h|e6c42f8a38aab2c7c4fc6931545a8e5463fdf87349902ec9687d9fe4949a0566"
        "Kits/FidelityFX/framegeneration/include/ffx_framegeneration_api_types.h|framegeneration/include/ffx_framegeneration_api_types.h|396596af3df63310674e8036abd2ad0fcd061e61334e7a2c76653a2be3c686b4"
        "Kits/FidelityFX/framegeneration/include/dx12/ffx_api_framegeneration_dx12.h|framegeneration/include/dx12/ffx_api_framegeneration_dx12.h|d533810bbd6fd4f87dfa2c1657c1457c7b1597b66345827d375b081d67127f66"
        "Kits/FidelityFX/signedbin/amd_fidelityfx_framegeneration_dx12.dll|bin/amd_fidelityfx_framegeneration_dx12.dll|02297beedd285e822d3a64f314cf00faf378dcec0edc47ff0c4dd71b3a8c2f18"
        "LICENSE.txt|LICENSE.txt|d5558cd419c8d46bdc958064cb97f963d1ea793866414c025906ec15033512ed")
    set(NUKE_FFX_OK ${NUKE_TECH_FSR})
    if(NUKE_TECH_FSR)
    foreach(_e IN LISTS _nuke_ffx_files)
        string(REPLACE "|" ";" _e "${_e}")
        list(GET _e 0 _src)
        list(GET _e 1 _dst)
        list(GET _e 2 _sha)
        if(NOT EXISTS "${_nuke_ffx_dir}/${_dst}")
            message(STATUS "FidelityFX: downloading ${_dst}...")
            file(DOWNLOAD "${_nuke_ffx_base}/${_src}" "${_nuke_ffx_dir}/${_dst}" EXPECTED_HASH SHA256=${_sha} STATUS _st)
            list(GET _st 0 _rc)
            if(NOT _rc EQUAL 0)
                message(WARNING "FidelityFX: download of ${_dst} failed - FSR 3.1 / FSR 4 stay off")
                set(NUKE_FFX_OK FALSE)
                file(REMOVE "${_nuke_ffx_dir}/${_dst}")
            endif()
        endif()
    endforeach()
    endif()

    set(_nuke_ffxvk_dir  "${CMAKE_CURRENT_SOURCE_DIR}/deps/ffx-vk")
    set(_nuke_ffxvk_base "https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/v1.1.4")
    set(_nuke_ffxvk_files
        "ffx-api/include/ffx_api/ffx_api.h|include/ffx_api.h|408b4203545e8299ed188fbb1d9ddfe0932b73ea6e8b14fe10edcf1f45771561"
        "ffx-api/include/ffx_api/ffx_api_loader.h|include/ffx_api_loader.h|7be51b7bf1e640b84c0c7f22a9db50a6816debce3755b51224e56c89501659c3"
        "ffx-api/include/ffx_api/ffx_api_types.h|include/ffx_api_types.h|dee2782813d22047c068e0fb27d1347324f8ce79d280a4672848b3b294cb12fe"
        "ffx-api/include/ffx_api/ffx_framegeneration.h|include/ffx_framegeneration.h|344936dccf49267d4421d1d74dc6db617d0a83c5a9f5991841325e812a38f127"
        "ffx-api/include/ffx_api/ffx_upscale.h|include/ffx_upscale.h|1155bcdcea9c10e6a95507bc9422ca062a853ac84250f0b09a637a66836b3ac1"
        "ffx-api/include/ffx_api/vk/ffx_api_vk.h|include/vk/ffx_api_vk.h|a50a8d5ff5689a10a6ce378f2df5569fd880c89ec46e8356aaf35eb40512edd9"
        "PrebuiltSignedDLL/amd_fidelityfx_vk.dll|bin/amd_fidelityfx_vk.dll|a1624cc4238fef046f30c4d80ce3f47be63fc5f5373f49e3ee9edb9960f54c78"
        "LICENSE.txt|LICENSE.txt|82cf74fc23885107c7f69249ac3f1dac55898ff68320eb2c753984cae6b2d78a")
    set(NUKE_FFXVK_OK ${NUKE_TECH_FSR})
    if(NUKE_TECH_FSR)
    foreach(_e IN LISTS _nuke_ffxvk_files)
        string(REPLACE "|" ";" _e "${_e}")
        list(GET _e 0 _src)
        list(GET _e 1 _dst)
        list(GET _e 2 _sha)
        if(NOT EXISTS "${_nuke_ffxvk_dir}/${_dst}")
            message(STATUS "FidelityFX (Vulkan, SDK 1.1.4): downloading ${_dst}...")
            file(DOWNLOAD "${_nuke_ffxvk_base}/${_src}" "${_nuke_ffxvk_dir}/${_dst}" EXPECTED_HASH SHA256=${_sha} STATUS _st)
            list(GET _st 0 _rc)
            if(NOT _rc EQUAL 0)
                message(WARNING "FidelityFX (Vulkan): download of ${_dst} failed - FSR on Vulkan stays off")
                set(NUKE_FFXVK_OK FALSE)
                file(REMOVE "${_nuke_ffxvk_dir}/${_dst}")
            endif()
        endif()
    endforeach()
    endif()

    set(_nuke_xess_dir "${CMAKE_CURRENT_SOURCE_DIR}/deps/xess")
    set(NUKE_XESS_OK ${NUKE_TECH_XESS})
    if(NUKE_TECH_XESS AND (NOT EXISTS "${_nuke_xess_dir}/bin/libxess.dll" OR NOT EXISTS "${_nuke_xess_dir}/bin/libxess_fg.dll" OR NOT EXISTS "${_nuke_xess_dir}/bin/libxell.dll"))
        message(STATUS "XeSS: downloading the 3.0.2 SDK...")
        file(DOWNLOAD "https://github.com/intel/xess/releases/download/v3.0.2/XeSS_SDK_3.0.2.zip" "${CMAKE_BINARY_DIR}/xess_sdk.zip"
             EXPECTED_HASH SHA256=88b8a373f30e33f3558a77a93e634f11b8132fc3047ea1a8edeead32b8471990 STATUS _st)
        list(GET _st 0 _rc)
        if(_rc EQUAL 0)
            file(ARCHIVE_EXTRACT INPUT "${CMAKE_BINARY_DIR}/xess_sdk.zip" DESTINATION "${CMAKE_BINARY_DIR}/xess-x"
                 PATTERNS "inc/xess/*" "inc/xess_fg/*" "inc/xell/*" "bin/libxess.dll" "bin/libxess_fg.dll" "bin/libxell.dll" "LICENSE.txt" "third-party-programs.txt")
            file(MAKE_DIRECTORY "${_nuke_xess_dir}/inc/xess" "${_nuke_xess_dir}/inc/xess_fg" "${_nuke_xess_dir}/inc/xell" "${_nuke_xess_dir}/bin")
            file(GLOB _nuke_xess_inc "${CMAKE_BINARY_DIR}/xess-x/inc/xess/*.h")
            file(COPY ${_nuke_xess_inc} DESTINATION "${_nuke_xess_dir}/inc/xess")
            file(GLOB _nuke_xess_inc "${CMAKE_BINARY_DIR}/xess-x/inc/xess_fg/*.h")
            file(COPY ${_nuke_xess_inc} DESTINATION "${_nuke_xess_dir}/inc/xess_fg")
            file(GLOB _nuke_xess_inc "${CMAKE_BINARY_DIR}/xess-x/inc/xell/*.h")
            file(COPY ${_nuke_xess_inc} DESTINATION "${_nuke_xess_dir}/inc/xell")
            file(COPY "${CMAKE_BINARY_DIR}/xess-x/bin/libxess.dll" "${CMAKE_BINARY_DIR}/xess-x/bin/libxess_fg.dll" "${CMAKE_BINARY_DIR}/xess-x/bin/libxell.dll" DESTINATION "${_nuke_xess_dir}/bin")
            file(COPY "${CMAKE_BINARY_DIR}/xess-x/LICENSE.txt" "${CMAKE_BINARY_DIR}/xess-x/third-party-programs.txt" DESTINATION "${_nuke_xess_dir}")
            message(STATUS "XeSS: vendored at deps/xess/")
        else()
            message(WARNING "XeSS: download failed - XeSS stays off")
            set(NUKE_XESS_OK FALSE)
        endif()
    endif()
    # NVIDIA Streamline: DLSS Frame Generation runs only through it (sl.interposer.dll loads the
    # sl.dlss_g / sl.reflex / sl.pcl plugins + nvngx_dlssg.dll). Headers and the production
    # (signed, non-watermarked) runtimes only; the SDK zip is ~276 MB, fetched once.
    set(_nuke_sl_dir "${CMAKE_CURRENT_SOURCE_DIR}/deps/streamline")
    set(NUKE_SL_OK ${NUKE_TECH_DLSS})
    if(NUKE_TECH_DLSS AND (NOT EXISTS "${_nuke_sl_dir}/bin/sl.dlss_g.dll" OR NOT EXISTS "${_nuke_sl_dir}/bin/NvLowLatencyVk.dll"))
        message(STATUS "Streamline: downloading the 2.14.1 SDK...")
        file(DOWNLOAD "https://github.com/NVIDIA-RTX/Streamline/releases/download/v2.14.1/streamline-sdk-v2.14.1.zip" "${CMAKE_BINARY_DIR}/streamline_sdk.zip"
             EXPECTED_HASH SHA256=92c4d954631a1710da86ca3fa8d5034f2b9503838c95fc4ae977ae149319781b STATUS _st)
        list(GET _st 0 _rc)
        if(_rc EQUAL 0)
            file(ARCHIVE_EXTRACT INPUT "${CMAKE_BINARY_DIR}/streamline_sdk.zip" DESTINATION "${CMAKE_BINARY_DIR}/sl-x"
                 PATTERNS "include/*" "bin/x64/sl.interposer.dll" "bin/x64/sl.common.dll" "bin/x64/sl.dlss_g.dll"
                          "bin/x64/sl.reflex.dll" "bin/x64/sl.pcl.dll" "bin/x64/nvngx_dlssg.dll" "bin/x64/NvLowLatencyVk.dll"
                          "license.txt" "3rd-party-licenses.md")
            file(MAKE_DIRECTORY "${_nuke_sl_dir}/include" "${_nuke_sl_dir}/bin")
            file(GLOB _nuke_sl_inc "${CMAKE_BINARY_DIR}/sl-x/include/*.h")
            file(COPY ${_nuke_sl_inc} DESTINATION "${_nuke_sl_dir}/include")
            file(GLOB _nuke_sl_bin "${CMAKE_BINARY_DIR}/sl-x/bin/x64/*.dll")
            file(COPY ${_nuke_sl_bin} DESTINATION "${_nuke_sl_dir}/bin")
            file(COPY "${CMAKE_BINARY_DIR}/sl-x/license.txt" "${CMAKE_BINARY_DIR}/sl-x/3rd-party-licenses.md" DESTINATION "${_nuke_sl_dir}")
            message(STATUS "Streamline: vendored at deps/streamline/")
        else()
            message(WARNING "Streamline: download failed - DLSS Frame Generation stays off")
            set(NUKE_SL_OK FALSE)
        endif()
    endif()

    if(NUKE_FFX_OK AND EXISTS "${_nuke_ffx_dir}/upscalers/include/ffx_upscale.h")
        target_include_directories(NukeRenderDiligent PRIVATE "${_nuke_ffx_dir}" "${_nuke_ffx_dir}/upscalers/include")
        target_compile_definitions(NukeRenderDiligent PRIVATE NUKE_FFX=1)
        if(EXISTS "${_nuke_ffx_dir}/framegeneration/include/ffx_framegeneration.h")
            target_include_directories(NukeRenderDiligent PRIVATE "${_nuke_ffx_dir}/framegeneration/include")
            target_compile_definitions(NukeRenderDiligent PRIVATE NUKE_FFX_FG=1)
        endif()
    endif()
    if(NUKE_XESS_OK AND EXISTS "${_nuke_xess_dir}/inc/xess/xess.h")
        target_include_directories(NukeRenderDiligent PRIVATE "${_nuke_xess_dir}/inc")
        target_compile_definitions(NukeRenderDiligent PRIVATE NUKE_XESS=1)
        if(EXISTS "${_nuke_xess_dir}/inc/xess_fg/xefg_swapchain_d3d12.h" AND EXISTS "${_nuke_xess_dir}/inc/xell/xell_d3d12.h")
            target_compile_definitions(NukeRenderDiligent PRIVATE NUKE_XESS_FG=1)
        endif()
    endif()
    if(NUKE_SL_OK AND EXISTS "${_nuke_sl_dir}/include/sl.h")
        target_include_directories(NukeRenderDiligent PRIVATE "${_nuke_sl_dir}/include")
        target_compile_definitions(NukeRenderDiligent PRIVATE NUKE_STREAMLINE=1)
    endif()
    if(NUKE_FFXVK_OK AND EXISTS "${_nuke_ffxvk_dir}/include/vk/ffx_api_vk.h")
        target_include_directories(NukeRenderDiligent PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/deps")   # <ffx-vk/include/...>
        target_compile_definitions(NukeRenderDiligent PRIVATE NUKE_FFX_VK=1)
    endif()
    # volk's global function pointers (the Vulkan swap-chain entry points the vendors replace).
    target_include_directories(NukeRenderDiligent PRIVATE "${_nuke_dc_dir}/ThirdParty/volk")
endif()

# Deploy the module DLL next to the editor, into modules/.
set(MODULES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../NukeEngine/${NUKE_RUN_SUBDIR}/$<CONFIG>/modules")
set(EDITOR_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/../NukeEngine/${NUKE_RUN_SUBDIR}/$<CONFIG>")

# The vendor runtimes of the technologies built in (NUKE_TECH), when the configure-time fetch got
# them: FSR = FidelityFX D3D12 upscaler + FG + Vulkan, XeSS = XeSS + XeSS-FG + XeLL, DLSS = NGX's
# nvngx_dlss.dll + the Streamline DLSS-G set. A technology left out deploys nothing (a runtime
# already in modules/ from an earlier build is inert: the renderer does not look for it).
set(_nuke_vendor_deploy "")
if(WIN32)
    set(_nuke_vendor_dlls "")
    if(NUKE_TECH_FSR)
        list(APPEND _nuke_vendor_dlls "${_nuke_ffx_dir}/bin/amd_fidelityfx_upscaler_dx12.dll" "${_nuke_ffx_dir}/bin/amd_fidelityfx_framegeneration_dx12.dll"
                                      "${_nuke_ffxvk_dir}/bin/amd_fidelityfx_vk.dll")
    endif()
    if(NUKE_TECH_XESS)
        list(APPEND _nuke_vendor_dlls "${_nuke_xess_dir}/bin/libxess.dll" "${_nuke_xess_dir}/bin/libxess_fg.dll" "${_nuke_xess_dir}/bin/libxell.dll")
    endif()
    if(NUKE_TECH_DLSS)
        list(APPEND _nuke_vendor_dlls "${_nuke_sl_dir}/bin/sl.interposer.dll" "${_nuke_sl_dir}/bin/sl.common.dll" "${_nuke_sl_dir}/bin/sl.dlss_g.dll"
                                      "${_nuke_sl_dir}/bin/sl.reflex.dll" "${_nuke_sl_dir}/bin/sl.pcl.dll" "${_nuke_sl_dir}/bin/nvngx_dlssg.dll"
                                      "${_nuke_sl_dir}/bin/NvLowLatencyVk.dll")
        # DLSS runtime (the fetched DLSS-Headers package): NGX searches the directory of the module
        # that initialised it - this renderer DLL's, modules/ - not the host's.
        if(DEFINED DLSS_BIN_DIR)
            list(APPEND _nuke_vendor_dlls "${DLSS_BIN_DIR}/nvngx_dlss.dll")
        endif()
    endif()
    foreach(_dll IN LISTS _nuke_vendor_dlls)
        if(EXISTS "${_dll}")
            list(APPEND _nuke_vendor_deploy COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_dll}" "${MODULES_DIR}/")
        endif()
    endforeach()
endif()

# Vendored-DXC deploy commands for the Linux branches below (empty without the download).
set(_nuke_dxc_deploy "")
if(UNIX AND NOT APPLE AND EXISTS "${_nuke_dxc}")
    set(_nuke_dxc_deploy
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_nuke_dxc}" "${EDITOR_DIR}/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_nuke_dxc_dir}/libdxil.so" "${EDITOR_DIR}/")
endif()
if(WIN32)
    add_custom_command(TARGET NukeRenderDiligent POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${MODULES_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:NukeRenderDiligent>" "${MODULES_DIR}/"
        # glfw3.dll is this module's runtime dependency — deploy it next to the exe.
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${VCPKG_INSTALLED}/$<$<CONFIG:Debug>:debug/>bin/glfw3.dll" "${EDITOR_DIR}/"
        # DXC (dxcompiler.dll + dxil.dll): Diligent LoadLibrary's dxcompiler.dll by bare name at
        # runtime to compile SM6.x shaders (RT etc). It is NOT a system DLL and NOT on the default
        # search path, so it must sit next to the exe. Vendored (deps/dxc) from the OFFICIAL DXC
        # release: ONE compiler for both backends — DXIL for D3D12 (dxil.dll = matching signer)
        # and SPIR-V for Vulkan (EngineVkCreateInfo::pDxCompilerPath points at this same DLL).
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/deps/dxc/dxcompiler.dll" "${EDITOR_DIR}/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/deps/dxc/dxil.dll" "${EDITOR_DIR}/"
        # DirectStorage runtime (vcpkg port "dstorage": the Microsoft redistributable). Loaded by
        # name at runtime by the D3D12 provider — absent DLLs mean CPU pak reads, never a failed
        # module load. dstoragecore.dll is dstorage.dll's own dependency.
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${VCPKG_INSTALLED}/bin/dstorage.dll" "${EDITOR_DIR}/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${VCPKG_INSTALLED}/bin/dstoragecore.dll" "${EDITOR_DIR}/"
        # The vendor runtimes of the technologies built in (NUKE_TECH; absent = not offered, never
        # a failed load). Package Project decides which of them ship.
        ${_nuke_vendor_deploy}
        COMMENT "Deploying NukeRenderDiligent.dll + glfw3.dll + DXC (dxcompiler/dxil) + DirectStorage + vendor tech (${NUKE_TECH})"
    )
elseif(APPLE)
    # ---- vendored UNIVERSAL MoltenVK (download-and-run games) --------------------------------
    # volk resolves "libvulkan.dylib"/"libMoltenVK.dylib" leaf names through the CALLER'S
    # rpaths — this module carries @loader_path/.., so a copy in the run dir (and therefore
    # in a packaged bundle's Contents/MacOS) is found on machines with NO Vulkan SDK at all.
    # Dev machines still prefer the brew loader (validation layers) — MoltenVK is the
    # last-resort fallback in volk's list. Official Khronos release, pinned by hash.
    set(_nuke_mvk_dir "${CMAKE_CURRENT_SOURCE_DIR}/deps/moltenvk")
    set(_nuke_mvk     "${_nuke_mvk_dir}/libMoltenVK.dylib")
    if(NOT EXISTS "${_nuke_mvk}")
        set(_nuke_mvk_url  "https://github.com/KhronosGroup/MoltenVK/releases/download/v1.4.2/MoltenVK-macos.tar")
        set(_nuke_mvk_sha  "f95765a6229cb7b915990a2890ce12ebe36a730b021545d3d52ae69ce4c4024e")
        message(STATUS "MoltenVK: downloading universal dylib (v1.4.2, Khronos release)...")
        file(DOWNLOAD "${_nuke_mvk_url}" "${CMAKE_BINARY_DIR}/moltenvk.tar"
             EXPECTED_HASH SHA256=${_nuke_mvk_sha} STATUS _nuke_mvk_st)
        list(GET _nuke_mvk_st 0 _nuke_mvk_rc)
        if(_nuke_mvk_rc EQUAL 0)
            file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/moltenvk-x")
            execute_process(COMMAND ${CMAKE_COMMAND} -E tar xf "${CMAKE_BINARY_DIR}/moltenvk.tar"
                                    MoltenVK/MoltenVK/dynamic/dylib/macOS/libMoltenVK.dylib
                            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/moltenvk-x")
            file(MAKE_DIRECTORY "${_nuke_mvk_dir}")
            file(COPY "${CMAKE_BINARY_DIR}/moltenvk-x/MoltenVK/MoltenVK/dynamic/dylib/macOS/libMoltenVK.dylib"
                 DESTINATION "${_nuke_mvk_dir}")
            message(STATUS "MoltenVK: vendored at deps/moltenvk/libMoltenVK.dylib")
        else()
            message(WARNING "MoltenVK download failed — packaged games will need a machine-installed Vulkan runtime")
        endif()
    endif()

    # The run dir mirrors what a packaged game ships: module + the SHARED glfw (real file,
    # matching the @rpath/libglfw.3.5.dylib install name) + the MoltenVK fallback. The
    # packager's dylib sweep then ships them automatically.
    get_filename_component(_nuke_glfw_real "${NUKE_GLFW_DYLIB}" REALPATH)
    add_custom_command(TARGET NukeRenderDiligent POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${MODULES_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:NukeRenderDiligent>" "${MODULES_DIR}/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_nuke_glfw_real}" "${EDITOR_DIR}/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_nuke_mvk}" "${EDITOR_DIR}/"
        COMMENT "Deploying NukeRenderDiligent module + shared GLFW + MoltenVK fallback"
    )
elseif(NUKE_GLFW_DYLIB)
    # Linux, shared GLFW: mirror the mac arrangement — the run dir carries what a packaged
    # game ships. Copying the soname path (libglfw.so.3, not its realpath) dereferences the
    # symlink while keeping the exact name DT_NEEDED looks up.
    add_custom_command(TARGET NukeRenderDiligent POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${MODULES_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:NukeRenderDiligent>" "${MODULES_DIR}/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${NUKE_GLFW_DYLIB}" "${EDITOR_DIR}/"
        ${_nuke_dxc_deploy}
        COMMENT "Deploying NukeRenderDiligent module + shared GLFW + DXC"
    )
else()
    add_custom_command(TARGET NukeRenderDiligent POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${MODULES_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:NukeRenderDiligent>" "${MODULES_DIR}/"
        ${_nuke_dxc_deploy}
        COMMENT "Deploying NukeRenderDiligent module"
    )
endif()
