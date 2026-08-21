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
set(DILIGENT_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
set(DILIGENT_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
set(DILIGENT_NO_VULKAN     OFF CACHE BOOL "" FORCE)   # Vulkan backend (task #138): editor default; DX stays for the runtime
set(DILIGENT_NO_OPENGL     ON  CACHE BOOL "" FORCE)
set(DILIGENT_NO_WEBGPU     ON  CACHE BOOL "" FORCE)
set(DILIGENT_NO_ARCHIVER   ON  CACHE BOOL "" FORCE)
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
    src/NukeDiligent_Post.cpp
    src/NukeDiligent_UI.cpp
    src/NukeDiligent_RT.cpp
    src/NukeDiligent_Sprite.cpp
    src/NukeDiligent_Decal.cpp
    src/NukeDiligent_Profile.cpp
    src/NukeDiligent_Native.cpp
    src/NukeDiligent_Occlusion.cpp
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

# Deploy the module DLL next to the editor, into modules/.
set(MODULES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../NukeEngine/${NUKE_RUN_SUBDIR}/$<CONFIG>/modules")
set(EDITOR_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/../NukeEngine/${NUKE_RUN_SUBDIR}/$<CONFIG>")

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
        COMMENT "Deploying NukeRenderDiligent.dll + glfw3.dll + DXC (dxcompiler/dxil)"
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
