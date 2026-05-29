# Require a CMake version new enough for the commands and CUDA handling used below.
cmake_minimum_required(VERSION 3.21)

# Name the project and say that configuration starts in plain C/C++ mode.
# CUDA is enabled later only if the selected app is a `.cu` file.
project(PortableMultiConv LANGUAGES C CXX)

# Refuse to configure anywhere except Linux, because the repo is currently maintained only there.
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "PortableMultiConv is now a Linux-only repo.")
endif()

# Force all C sources to compile as C11.
set(CMAKE_C_STANDARD 11)
# Make the C11 requirement mandatory instead of best-effort.
set(CMAKE_C_STANDARD_REQUIRED ON)
# Force all C++ sources to compile as C++17.
set(CMAKE_CXX_STANDARD 17)
# Make the C++17 requirement mandatory instead of best-effort.
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Absolute path to this repo's public headers, used by most app targets.
set(PORTABLE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
# Absolute path to the repo output folder used by some apps and helper code.
set(PORTABLE_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")
# Absolute path to the generic plotting helper script.
set(PORTABLE_PLOT_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/scripts/plot.py")
# Absolute path to the callback-timing visualization helper script.
set(PORTABLE_CALLBACK_TIMING_VIEW_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/scripts/show_callback_timing.py")
# Absolute path to the topology-matrix visualization helper script.
set(PORTABLE_TOPOLOGY_VIEW_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/scripts/show_topology_matrix.py")
# Absolute path to the exact-convolution validation helper script.
set(PORTABLE_EXACT_CONV_VIEW_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/scripts/show_exact_convolution_validation.py")
# Absolute path to the detect-timing plot helper script.
set(PORTABLE_DETECT_TIMMING_PLOT_SCRIPT
    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/plot_detect_timming_results.py")
# Absolute path to the vendored third-party dependency folder.
set(PORTABLE_THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
# Absolute path to the vendored CUDA helper headers copied from NVIDIA samples.
set(PORTABLE_CUDA_SAMPLES_COMMON_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cuda-samples/Common")

# User-visible configure switch: the basename of the app to build, without extension.
# Example: `-DPORTABLE_APP=sound_device_test`.
set(PORTABLE_APP "" CACHE STRING
    "Basename of the app source file under Portable/apps, without extension.")
# User-visible configure switch: whether the selected app should use mock audio or real PortAudio.
# Example: `-DPORTABLE_USE_MOCK=ON` or `-DPORTABLE_USE_MOCK=OFF`.
set(PORTABLE_USE_MOCK "" CACHE STRING
    "Whether the selected app should use the mock audio backend (ON or OFF).")
# Help GUIs present the accepted values for the mock-audio switch.
set_property(CACHE PORTABLE_USE_MOCK PROPERTY STRINGS ON OFF)

# Fail early if the user did not specify which app to build.
if(PORTABLE_APP STREQUAL "")
    message(FATAL_ERROR
        "PORTABLE_APP was not set.\n"
        "Example:\n"
        "  cmake -S Portable -B Portable/build -G Ninja "
        "-DPORTABLE_APP=sound_device_test -DPORTABLE_USE_MOCK=OFF")
endif()

# Fail early if the user did not specify whether to use the mock backend.
if(PORTABLE_USE_MOCK STREQUAL "")
    message(FATAL_ERROR
        "PORTABLE_USE_MOCK was not set.\n"
        "Example:\n"
        "  cmake -S Portable -B Portable/build -G Ninja "
        "-DPORTABLE_APP=sound_device_test -DPORTABLE_USE_MOCK=OFF")
endif()

# Accept both `sound_device_test` and `portable_sound_device_test` as the app name.
set(PORTABLE_APP_NAME "${PORTABLE_APP}")
# If the user passed the executable name, strip the `portable_` prefix to get the source basename.
string(REGEX REPLACE "^portable_" "" PORTABLE_APP_NAME "${PORTABLE_APP_NAME}")

# Normalize the mock-audio choice into a strict boolean variable.
string(TOUPPER "${PORTABLE_USE_MOCK}" PORTABLE_USE_MOCK_UPPER)
# Interpret the usual truthy spellings as ON.
if(PORTABLE_USE_MOCK_UPPER STREQUAL "ON"
   OR PORTABLE_USE_MOCK_UPPER STREQUAL "TRUE"
   OR PORTABLE_USE_MOCK_UPPER STREQUAL "1")
    set(PORTABLE_USE_MOCK_BOOL TRUE)
# Interpret the usual falsy spellings as OFF.
elseif(PORTABLE_USE_MOCK_UPPER STREQUAL "OFF"
       OR PORTABLE_USE_MOCK_UPPER STREQUAL "FALSE"
       OR PORTABLE_USE_MOCK_UPPER STREQUAL "0")
    set(PORTABLE_USE_MOCK_BOOL FALSE)
# Reject anything else so the command line stays explicit and predictable.
else()
    message(FATAL_ERROR
        "PORTABLE_USE_MOCK must be ON or OFF, but got `${PORTABLE_USE_MOCK}`.")
endif()

# Discover all normal C++ apps automatically so adding a new `.cpp` file does not require editing this file.
file(GLOB PORTABLE_CPP_APP_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/apps/*.cpp")
# Discover all CUDA apps automatically so adding a new `.cu` file does not require editing this file.
file(GLOB PORTABLE_CUDA_APP_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/apps/*.cu")

# Combine both app lists so we can search them with one loop.
set(PORTABLE_ALL_APP_SOURCES
    ${PORTABLE_CPP_APP_SOURCES}
    ${PORTABLE_CUDA_APP_SOURCES})
# Start with no selected source until we find the requested app.
set(PORTABLE_SELECTED_SOURCE "")
# Start by assuming the selected app is not a CUDA app.
set(PORTABLE_SELECTED_IS_CUDA FALSE)
# Collect app basenames for the error message shown when the requested app does not exist.
set(PORTABLE_AVAILABLE_APPS "")

# Walk over every discovered app source file.
foreach(source IN LISTS PORTABLE_ALL_APP_SOURCES)
    # Extract the basename without extension, such as `sound_device_test`.
    get_filename_component(source_name "${source}" NAME_WE)
    # Extract the source-file extension, such as `.cpp` or `.cu`.
    get_filename_component(source_extension "${source}" EXT)
    # Remember this app name for diagnostics.
    list(APPEND PORTABLE_AVAILABLE_APPS "${source_name}")

    # If this file matches the app requested on the command line, select it.
    if(source_name STREQUAL PORTABLE_APP_NAME)
        # Guard against duplicate basenames with different extensions.
        if(NOT PORTABLE_SELECTED_SOURCE STREQUAL "")
            message(FATAL_ERROR
                "More than one app source matches `PORTABLE_APP=${PORTABLE_APP_NAME}`.")
        endif()

        # Store the exact source file path for the selected app.
        set(PORTABLE_SELECTED_SOURCE "${source}")

        # Mark whether the selected app is CUDA based on its file extension.
        if(source_extension STREQUAL ".cu")
            set(PORTABLE_SELECTED_IS_CUDA TRUE)
        else()
            set(PORTABLE_SELECTED_IS_CUDA FALSE)
        endif()
    endif()
endforeach()

# Sort the discovered app names so any error message is stable and easy to scan.
list(SORT PORTABLE_AVAILABLE_APPS)
# Turn the app-name list into comma-separated text for the error message below.
string(REPLACE ";" ", " PORTABLE_AVAILABLE_APPS_TEXT "${PORTABLE_AVAILABLE_APPS}")

# Fail early if the requested app name does not match any file under `Portable/apps`.
if(PORTABLE_SELECTED_SOURCE STREQUAL "")
    message(FATAL_ERROR
        "No app source matched `PORTABLE_APP=${PORTABLE_APP_NAME}`.\n"
        "Available apps: ${PORTABLE_AVAILABLE_APPS_TEXT}")
endif()

# Derive the executable target name from the source basename.
set(PORTABLE_TARGET_NAME "portable_${PORTABLE_APP_NAME}")

# Convert the CUDA/non-CUDA boolean into readable text for the configure log.
if(PORTABLE_SELECTED_IS_CUDA)
    set(PORTABLE_SELECTED_LANGUAGE "CUDA")
else()
    set(PORTABLE_SELECTED_LANGUAGE "C++")
endif()

# Normalize path strings before embedding them into compile-time macros.
file(TO_CMAKE_PATH "${PORTABLE_OUTPUT_DIR}" PORTABLE_OUTPUT_DIR_PATH)
# Normalize the generic plot script path before embedding it into compile-time macros.
file(TO_CMAKE_PATH "${PORTABLE_PLOT_SCRIPT}" PORTABLE_PLOT_SCRIPT_PATH)
# Normalize the callback timing viewer path before embedding it into compile-time macros.
file(TO_CMAKE_PATH "${PORTABLE_CALLBACK_TIMING_VIEW_SCRIPT}" PORTABLE_CALLBACK_TIMING_VIEW_SCRIPT_PATH)
# Normalize the topology viewer path before embedding it into compile-time macros.
file(TO_CMAKE_PATH "${PORTABLE_TOPOLOGY_VIEW_SCRIPT}" PORTABLE_TOPOLOGY_VIEW_SCRIPT_PATH)
# Normalize the exact-convolution viewer path before embedding it into compile-time macros.
file(TO_CMAKE_PATH "${PORTABLE_EXACT_CONV_VIEW_SCRIPT}" PORTABLE_EXACT_CONV_VIEW_SCRIPT_PATH)
# Normalize the detect-timing plot helper path before embedding it into compile-time macros.
file(TO_CMAKE_PATH "${PORTABLE_DETECT_TIMMING_PLOT_SCRIPT}" PORTABLE_DETECT_TIMMING_PLOT_SCRIPT_PATH)

# Print the chosen app name during configure.
message(STATUS "portable app: ${PORTABLE_APP_NAME}")
# Print the chosen source path during configure.
message(STATUS "selected source: ${PORTABLE_SELECTED_SOURCE}")
# Print the chosen executable target name during configure.
message(STATUS "selected target: ${PORTABLE_TARGET_NAME}")
# Print whether the selected app is handled as CUDA or plain C++.
message(STATUS "selected app language: ${PORTABLE_SELECTED_LANGUAGE}")
# Print the chosen mock/real backend setting during configure.
message(STATUS "MOCK=${PORTABLE_USE_MOCK_BOOL}")

# Helper: embed the repo's script and output paths into a target as compile-time macros.
function(portable_apply_common_path_definitions target)
    # Embed the output directory path.
    target_compile_definitions("${target}" PRIVATE
        PORTABLE_OUTPUT_DIR=\"${PORTABLE_OUTPUT_DIR_PATH}\"
        PORTABLE_PLOT_SCRIPT=\"${PORTABLE_PLOT_SCRIPT_PATH}\"
        PORTABLE_CALLBACK_TIMING_VIEW_SCRIPT=\"${PORTABLE_CALLBACK_TIMING_VIEW_SCRIPT_PATH}\"
        PORTABLE_TOPOLOGY_VIEW_SCRIPT=\"${PORTABLE_TOPOLOGY_VIEW_SCRIPT_PATH}\"
        PORTABLE_EXACT_CONV_VIEW_SCRIPT=\"${PORTABLE_EXACT_CONV_VIEW_SCRIPT_PATH}\"
        PORTABLE_DETECT_TIMMING_PLOT_SCRIPT=\"${PORTABLE_DETECT_TIMMING_PLOT_SCRIPT_PATH}\")
endfunction()

# Helper: attach either the mock-audio backend or the real PortAudio backend to a target.
function(portable_link_audio_backend target)
    # When mock mode is requested, tell the source code via `MOCK=TRUE` and link the mock backend.
    if(PORTABLE_USE_MOCK_BOOL)
        target_compile_definitions("${target}" PRIVATE MOCK=TRUE)
        target_link_libraries("${target}" PRIVATE
            portable_mockaudio
            portable_fftw3f)
    # Otherwise tell the source code `MOCK=FALSE` and link the real PortAudio backend.
    else()
        target_compile_definitions("${target}" PRIVATE MOCK=FALSE)
        target_link_libraries("${target}" PRIVATE
            portaudio
            portable_fftw3f)
    endif()
endfunction()

# Helper: apply the common CUDA-related compile and link settings shared by all CUDA apps.
function(portable_configure_cuda_target target)
    # Make sure the host-side C++ parts of a CUDA target still get C++17 features.
    target_compile_features("${target}" PRIVATE cxx_std_17)
    # Require CUDA source files to compile as CUDA 17.
    set_target_properties("${target}" PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON)

    # Prefer the static CUDA runtime if CMake found it.
    if(TARGET CUDA::cudart_static)
        target_link_libraries("${target}" PRIVATE CUDA::cudart_static)
    # Fall back to the shared CUDA runtime if the static one is not available.
    else()
        target_link_libraries("${target}" PRIVATE CUDA::cudart)
    endif()

    # Link cuFFT for every CUDA app so all `.cu` targets are treated symmetrically.
    target_link_libraries("${target}" PRIVATE CUDA::cufft)
endfunction()

# Find `pkg-config`, which is how this repo discovers FFTW on Linux.
find_package(PkgConfig REQUIRED)
# Ask `pkg-config` for the compile/link information for the single-precision FFTW package.
pkg_check_modules(FFTW3F IMPORTED_TARGET REQUIRED fftw3f)
# Wrap the pkg-config FFTW result in a repo-local interface target with a stable name.
add_library(portable_fftw3f INTERFACE)
# Make anything linking `portable_fftw3f` inherit the FFTW flags from pkg-config.
target_link_libraries(portable_fftw3f INTERFACE PkgConfig::FFTW3F)

# Build the repo's mock-audio implementation as a normal static library.
add_library(portable_mockaudio STATIC
    src/mockportaudio.cpp
    src/mock_device_registry.cpp
    src/signal.cpp)
# Let consumers of `portable_mockaudio` include headers from `Portable/include`.
target_include_directories(portable_mockaudio PUBLIC "${PORTABLE_INCLUDE_DIR}")
# Embed the repo paths used internally by the mock-audio library.
target_compile_definitions(portable_mockaudio PRIVATE
    PORTABLE_OUTPUT_DIR=\"${PORTABLE_OUTPUT_DIR_PATH}\"
    PORTABLE_EXACT_CONV_VIEW_SCRIPT=\"${PORTABLE_EXACT_CONV_VIEW_SCRIPT_PATH}\")
# Ensure the mock-audio library also brings in FFTW.
target_link_libraries(portable_mockaudio PUBLIC portable_fftw3f)

# Only configure the vendored PortAudio dependency when the user asked for real audio.
if(NOT PORTABLE_USE_MOCK_BOOL)
    # Build vendored PortAudio as a static library.
    set(PA_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    # Skip PortAudio's own tests to keep repo builds smaller and simpler.
    set(PA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    # Skip PortAudio's sample programs for the same reason.
    set(PA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    # Enable ALSA support because this repo currently targets Linux audio through ALSA.
    set(PA_USE_ALSA ON CACHE BOOL "" FORCE)
    # Link ALSA directly instead of trying to load it dynamically at runtime.
    set(PA_ALSA_DYNAMIC OFF CACHE BOOL "" FORCE)
    # Disable JACK support.
    set(PA_USE_JACK OFF CACHE BOOL "" FORCE)
    # Disable PulseAudio support.
    set(PA_USE_PULSEAUDIO OFF CACHE BOOL "" FORCE)
    # Disable sndio support.
    set(PA_USE_SNDIO OFF CACHE BOOL "" FORCE)
    # Disable OSS support.
    set(PA_USE_OSS OFF CACHE BOOL "" FORCE)
    # Disable PortAudio's optional AudioIO bits.
    set(AUDIOIO OFF CACHE BOOL "" FORCE)

    # Bring the vendored PortAudio CMake project into this build.
    add_subdirectory("${PORTABLE_THIRD_PARTY_DIR}/portaudio" EXCLUDE_FROM_ALL)

    # Fail early if the vendored PortAudio subproject did not produce the expected target.
    if(NOT TARGET portaudio)
        message(FATAL_ERROR
            "Real audio was requested, but the vendored PortAudio target was not created.")
    endif()
endif()

# If the selected app is CUDA, turn on the CUDA language now.
if(PORTABLE_SELECTED_IS_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    # Make sure the vendored NVIDIA helper headers exist before defining CUDA sample targets that include them.
    if(NOT EXISTS "${PORTABLE_CUDA_SAMPLES_COMMON_DIR}/helper_cuda.h" OR
       NOT EXISTS "${PORTABLE_CUDA_SAMPLES_COMMON_DIR}/helper_string.h")
        message(FATAL_ERROR
            "Portable CUDA apps expected vendored helper headers under "
            "${PORTABLE_CUDA_SAMPLES_COMMON_DIR} but they were not found.")
    endif()
endif()

# Register the one selected executable target from the chosen source file.
add_executable("${PORTABLE_TARGET_NAME}" "${PORTABLE_SELECTED_SOURCE}")
# Embed the common repo paths into the selected app.
portable_apply_common_path_definitions("${PORTABLE_TARGET_NAME}")
# Attach either the mock or real audio backend and define `MOCK` accordingly.
portable_link_audio_backend("${PORTABLE_TARGET_NAME}")

# If the selected app is CUDA, apply the shared CUDA rule used for all `.cu` files.
if(PORTABLE_SELECTED_IS_CUDA)
    # Let CUDA apps include both repo headers and the vendored NVIDIA helper headers.
    target_include_directories("${PORTABLE_TARGET_NAME}" PRIVATE
        "${PORTABLE_INCLUDE_DIR}"
        "${PORTABLE_CUDA_SAMPLES_COMMON_DIR}")
    # Apply the shared CUDA compile/runtime settings.
    portable_configure_cuda_target("${PORTABLE_TARGET_NAME}")
# Otherwise apply the shared C++ rule used for all `.cpp` files.
else()
    # Let normal C++ apps include repo headers.
    target_include_directories("${PORTABLE_TARGET_NAME}" PRIVATE
        "${PORTABLE_INCLUDE_DIR}")
endif()
