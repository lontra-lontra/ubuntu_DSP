cmake_minimum_required(VERSION 3.21)

project(PortableMultiConv LANGUAGES C CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "PortableMultiConv is now a Linux-only repo.")
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PORTABLE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(PORTABLE_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")
set(PORTABLE_PLOT_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/scripts/plot.py")
set(PORTABLE_CALLBACK_TIMING_VIEW_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/scripts/show_callback_timing.py")
set(PORTABLE_TOPOLOGY_VIEW_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/scripts/show_topology_matrix.py")
set(PORTABLE_EXACT_CONV_VIEW_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/scripts/show_exact_convolution_validation.py")
set(PORTABLE_THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party")

set(PORTABLE_MULTI_CONV_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/apps/multi_conv_benchmarking.cpp")
set(PORTABLE_SOUND_DEVICE_QUERY_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/apps/sound_device_query.cpp")
set(PORTABLE_SOUND_DEVICE_TEST_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/apps/sound_device_test.cpp")
set(PORTABLE_INFER_TOPOLOGY_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/apps/infer_topology_and_save_it.cpp")
set(PORTABLE_INFER_FROM_INFERRED_TOPOLOGY_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/apps/infer_topology_from_infered_topology_and_save_it.cpp")
set(PORTABLE_DETECT_TIMMING_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/apps/detect_timming.cpp")
set(PORTABLE_CUDA_DEVICE_QUERY_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/apps/cuda_device_query.cu")
set(PORTABLE_SIMPLE_CUDA_PORTAUDIO_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/apps/simple_cuda_portaudio.cu")
set(PORTABLE_SIMPLE_CUDA_NAIVE_CONV_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/apps/simple_cuda_naive_convolution.cu")
set(PORTABLE_SIMPLE_CUDA_LESS_NAIVE_CONV_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/apps/simple_cuda_less_naive_convolution.cu")
set(PORTABLE_CUDA_SAMPLES_COMMON_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cuda-samples/Common")

option(PORTABLE_ENABLE_CUDA_APPS
    "Build CUDA apps that require the NVIDIA CUDA Toolkit"
    OFF)

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${PORTABLE_MULTI_CONV_SOURCE}"
    "${PORTABLE_SOUND_DEVICE_QUERY_SOURCE}"
    "${PORTABLE_SOUND_DEVICE_TEST_SOURCE}"
    "${PORTABLE_INFER_TOPOLOGY_SOURCE}"
    "${PORTABLE_INFER_FROM_INFERRED_TOPOLOGY_SOURCE}"
    "${PORTABLE_DETECT_TIMMING_SOURCE}"
    "${PORTABLE_CUDA_DEVICE_QUERY_SOURCE}"
    "${PORTABLE_SIMPLE_CUDA_PORTAUDIO_SOURCE}"
    "${PORTABLE_SIMPLE_CUDA_NAIVE_CONV_SOURCE}"
    "${PORTABLE_SIMPLE_CUDA_LESS_NAIVE_CONV_SOURCE}")

file(TO_CMAKE_PATH "${PORTABLE_OUTPUT_DIR}" PORTABLE_OUTPUT_DIR_PATH)
file(TO_CMAKE_PATH "${PORTABLE_PLOT_SCRIPT}" PORTABLE_PLOT_SCRIPT_PATH)
file(TO_CMAKE_PATH "${PORTABLE_CALLBACK_TIMING_VIEW_SCRIPT}" PORTABLE_CALLBACK_TIMING_VIEW_SCRIPT_PATH)
file(TO_CMAKE_PATH "${PORTABLE_TOPOLOGY_VIEW_SCRIPT}" PORTABLE_TOPOLOGY_VIEW_SCRIPT_PATH)
file(TO_CMAKE_PATH "${PORTABLE_EXACT_CONV_VIEW_SCRIPT}" PORTABLE_EXACT_CONV_VIEW_SCRIPT_PATH)

function(portable_read_source_define source macro out_var)
    file(STRINGS "${source}" define_lines REGEX "^#define[ \t]+${macro}[ \t]+")
    list(LENGTH define_lines define_count)

    if(define_count EQUAL 0)
        message(FATAL_ERROR
            "${source} must contain '#define ${macro} ...' near the top of the file.")
    endif()

    list(GET define_lines 0 define_line)
    string(REGEX REPLACE "^#define[ \t]+${macro}[ \t]+(.+)$" "\\1"
        define_value "${define_line}")
    string(STRIP "${define_value}" define_value)
    set(${out_var} "${define_value}" PARENT_SCOPE)
endfunction()

function(portable_source_is_mock source out_var)
    portable_read_source_define("${source}" "MOCK" mock_value)

    if(mock_value STREQUAL "TRUE")
        set(${out_var} TRUE PARENT_SCOPE)
    elseif(mock_value STREQUAL "FALSE")
        set(${out_var} FALSE PARENT_SCOPE)
    else()
        message(FATAL_ERROR
            "${source} must define MOCK as either TRUE or FALSE.")
    endif()
endfunction()

function(portable_link_audio_backend target is_mock)
    if(${is_mock})
        target_link_libraries("${target}" PRIVATE
            portable_mockaudio
            portable_fftw3f)
    else()
        target_link_libraries("${target}" PRIVATE
            portaudio
            portable_fftw3f)
    endif()
endfunction()

function(portable_configure_cuda_target target)
    target_compile_features("${target}" PRIVATE cxx_std_17)
    set_target_properties("${target}" PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON)

    if(TARGET CUDA::cudart_static)
        target_link_libraries("${target}" PRIVATE CUDA::cudart_static)
    else()
        target_link_libraries("${target}" PRIVATE CUDA::cudart)
    endif()
endfunction()

portable_source_is_mock("${PORTABLE_MULTI_CONV_SOURCE}" PORTABLE_MULTI_CONV_IS_MOCK)
portable_source_is_mock("${PORTABLE_SOUND_DEVICE_QUERY_SOURCE}" PORTABLE_SOUND_DEVICE_QUERY_IS_MOCK)
portable_source_is_mock("${PORTABLE_SOUND_DEVICE_TEST_SOURCE}" PORTABLE_SOUND_DEVICE_TEST_IS_MOCK)
portable_source_is_mock("${PORTABLE_INFER_TOPOLOGY_SOURCE}" PORTABLE_INFER_TOPOLOGY_IS_MOCK)
portable_source_is_mock("${PORTABLE_INFER_FROM_INFERRED_TOPOLOGY_SOURCE}" PORTABLE_INFER_FROM_INFERRED_TOPOLOGY_IS_MOCK)
portable_source_is_mock("${PORTABLE_DETECT_TIMMING_SOURCE}" PORTABLE_DETECT_TIMMING_IS_MOCK)
portable_source_is_mock("${PORTABLE_SIMPLE_CUDA_NAIVE_CONV_SOURCE}" PORTABLE_SIMPLE_CUDA_NAIVE_CONV_IS_MOCK)
portable_source_is_mock("${PORTABLE_SIMPLE_CUDA_LESS_NAIVE_CONV_SOURCE}" PORTABLE_SIMPLE_CUDA_LESS_NAIVE_CONV_IS_MOCK)

message(STATUS "portable_multi_conv_benchmarking: MOCK=${PORTABLE_MULTI_CONV_IS_MOCK}")
message(STATUS "portable_sound_device_query: MOCK=${PORTABLE_SOUND_DEVICE_QUERY_IS_MOCK}")
message(STATUS "portable_sound_device_test: MOCK=${PORTABLE_SOUND_DEVICE_TEST_IS_MOCK}")
message(STATUS "portable_infer_topology_and_save_it: MOCK=${PORTABLE_INFER_TOPOLOGY_IS_MOCK}")
message(STATUS "portable_infer_topology_from_infered_topology_and_save_it: MOCK=${PORTABLE_INFER_FROM_INFERRED_TOPOLOGY_IS_MOCK}")
message(STATUS "portable_detect_timming: MOCK=${PORTABLE_DETECT_TIMMING_IS_MOCK}")
message(STATUS "portable_simple_cuda_naive_convolution: MOCK=${PORTABLE_SIMPLE_CUDA_NAIVE_CONV_IS_MOCK}")
message(STATUS "portable_simple_cuda_less_naive_convolution: MOCK=${PORTABLE_SIMPLE_CUDA_LESS_NAIVE_CONV_IS_MOCK}")

set(PORTABLE_NEEDS_REAL_AUDIO FALSE)
if(NOT PORTABLE_MULTI_CONV_IS_MOCK
   OR NOT PORTABLE_SOUND_DEVICE_QUERY_IS_MOCK
   OR NOT PORTABLE_SOUND_DEVICE_TEST_IS_MOCK
   OR NOT PORTABLE_INFER_TOPOLOGY_IS_MOCK
   OR NOT PORTABLE_INFER_FROM_INFERRED_TOPOLOGY_IS_MOCK
   OR NOT PORTABLE_DETECT_TIMMING_IS_MOCK
   OR NOT PORTABLE_SIMPLE_CUDA_NAIVE_CONV_IS_MOCK
   OR NOT PORTABLE_SIMPLE_CUDA_LESS_NAIVE_CONV_IS_MOCK)
    set(PORTABLE_NEEDS_REAL_AUDIO TRUE)
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3F IMPORTED_TARGET REQUIRED fftw3f)
add_library(portable_fftw3f INTERFACE)
target_link_libraries(portable_fftw3f INTERFACE PkgConfig::FFTW3F)

add_library(portable_mockaudio STATIC
    src/mockportaudio.cpp
    src/mock_device_registry.cpp
    src/signal.cpp)
target_include_directories(portable_mockaudio PUBLIC "${PORTABLE_INCLUDE_DIR}")
target_compile_definitions(portable_mockaudio PRIVATE
    PORTABLE_OUTPUT_DIR=\"${PORTABLE_OUTPUT_DIR_PATH}\"
    PORTABLE_EXACT_CONV_VIEW_SCRIPT=\"${PORTABLE_EXACT_CONV_VIEW_SCRIPT_PATH}\")
target_link_libraries(portable_mockaudio PUBLIC portable_fftw3f)

if(PORTABLE_NEEDS_REAL_AUDIO)
    set(PA_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(PA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(PA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(PA_USE_ALSA ON CACHE BOOL "" FORCE)
    set(PA_ALSA_DYNAMIC OFF CACHE BOOL "" FORCE)
    set(PA_USE_JACK OFF CACHE BOOL "" FORCE)
    set(PA_USE_PULSEAUDIO OFF CACHE BOOL "" FORCE)
    set(PA_USE_SNDIO OFF CACHE BOOL "" FORCE)
    set(PA_USE_OSS OFF CACHE BOOL "" FORCE)
    set(AUDIOIO OFF CACHE BOOL "" FORCE)

    add_subdirectory("${PORTABLE_THIRD_PARTY_DIR}/portaudio" EXCLUDE_FROM_ALL)

    if(NOT TARGET portaudio)
        message(FATAL_ERROR
            "A source file is configured with MOCK=FALSE, but the vendored PortAudio target was not created.")
    endif()
endif()

add_executable(portable_multi_conv_benchmarking
    "${PORTABLE_MULTI_CONV_SOURCE}")
target_include_directories(portable_multi_conv_benchmarking PRIVATE "${PORTABLE_INCLUDE_DIR}")
portable_link_audio_backend(portable_multi_conv_benchmarking ${PORTABLE_MULTI_CONV_IS_MOCK})
target_compile_definitions(portable_multi_conv_benchmarking PRIVATE
    PORTABLE_OUTPUT_DIR=\"${PORTABLE_OUTPUT_DIR_PATH}\"
    PORTABLE_PLOT_SCRIPT=\"${PORTABLE_PLOT_SCRIPT_PATH}\")

add_executable(portable_sound_device_query
    "${PORTABLE_SOUND_DEVICE_QUERY_SOURCE}")
target_include_directories(portable_sound_device_query PRIVATE "${PORTABLE_INCLUDE_DIR}")
portable_link_audio_backend(portable_sound_device_query ${PORTABLE_SOUND_DEVICE_QUERY_IS_MOCK})
target_compile_definitions(portable_sound_device_query PRIVATE
    PORTABLE_OUTPUT_DIR=\"${PORTABLE_OUTPUT_DIR_PATH}\"
    PORTABLE_PLOT_SCRIPT=\"${PORTABLE_PLOT_SCRIPT_PATH}\")

add_executable(portable_sound_device_test
    "${PORTABLE_SOUND_DEVICE_TEST_SOURCE}")
target_include_directories(portable_sound_device_test PRIVATE "${PORTABLE_INCLUDE_DIR}")
portable_link_audio_backend(portable_sound_device_test ${PORTABLE_SOUND_DEVICE_TEST_IS_MOCK})

add_executable(portable_infer_topology_and_save_it
    "${PORTABLE_INFER_TOPOLOGY_SOURCE}")
target_include_directories(portable_infer_topology_and_save_it PRIVATE "${PORTABLE_INCLUDE_DIR}")
portable_link_audio_backend(portable_infer_topology_and_save_it ${PORTABLE_INFER_TOPOLOGY_IS_MOCK})
target_compile_definitions(portable_infer_topology_and_save_it PRIVATE
    PORTABLE_OUTPUT_DIR=\"${PORTABLE_OUTPUT_DIR_PATH}\"
    PORTABLE_PLOT_SCRIPT=\"${PORTABLE_PLOT_SCRIPT_PATH}\"
    PORTABLE_TOPOLOGY_VIEW_SCRIPT=\"${PORTABLE_TOPOLOGY_VIEW_SCRIPT_PATH}\")

add_executable(portable_infer_topology_from_infered_topology_and_save_it
    "${PORTABLE_INFER_FROM_INFERRED_TOPOLOGY_SOURCE}")
target_include_directories(portable_infer_topology_from_infered_topology_and_save_it PRIVATE "${PORTABLE_INCLUDE_DIR}")
portable_link_audio_backend(
    portable_infer_topology_from_infered_topology_and_save_it
    ${PORTABLE_INFER_FROM_INFERRED_TOPOLOGY_IS_MOCK})
target_compile_definitions(portable_infer_topology_from_infered_topology_and_save_it PRIVATE
    PORTABLE_OUTPUT_DIR=\"${PORTABLE_OUTPUT_DIR_PATH}\"
    PORTABLE_TOPOLOGY_VIEW_SCRIPT=\"${PORTABLE_TOPOLOGY_VIEW_SCRIPT_PATH}\")

add_executable(portable_detect_timming
    "${PORTABLE_DETECT_TIMMING_SOURCE}")
target_include_directories(portable_detect_timming PRIVATE "${PORTABLE_INCLUDE_DIR}")
portable_link_audio_backend(portable_detect_timming ${PORTABLE_DETECT_TIMMING_IS_MOCK})
target_compile_definitions(portable_detect_timming PRIVATE
    PORTABLE_OUTPUT_DIR=\"${PORTABLE_OUTPUT_DIR_PATH}\")

if(PORTABLE_ENABLE_CUDA_APPS)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    if(NOT EXISTS "${PORTABLE_CUDA_SAMPLES_COMMON_DIR}/helper_cuda.h" OR
       NOT EXISTS "${PORTABLE_CUDA_SAMPLES_COMMON_DIR}/helper_string.h")
        message(FATAL_ERROR
            "Portable CUDA apps expected vendored helper headers under "
            "${PORTABLE_CUDA_SAMPLES_COMMON_DIR} but they were not found.")
    endif()

    add_executable(portable_simple_cuda_portaudio
        "${PORTABLE_SIMPLE_CUDA_PORTAUDIO_SOURCE}")
    target_include_directories(portable_simple_cuda_portaudio PRIVATE
        "${PORTABLE_INCLUDE_DIR}")
    target_link_libraries(portable_simple_cuda_portaudio PRIVATE portable_mockaudio)
    target_compile_definitions(portable_simple_cuda_portaudio PRIVATE
        PORTABLE_OUTPUT_DIR=\"${PORTABLE_OUTPUT_DIR_PATH}\"
        PORTABLE_PLOT_SCRIPT=\"${PORTABLE_PLOT_SCRIPT_PATH}\"
        PORTABLE_CALLBACK_TIMING_VIEW_SCRIPT=\"${PORTABLE_CALLBACK_TIMING_VIEW_SCRIPT_PATH}\")
    portable_configure_cuda_target(portable_simple_cuda_portaudio)

    add_executable(portable_simple_cuda_naive_convolution
        "${PORTABLE_SIMPLE_CUDA_NAIVE_CONV_SOURCE}")
    target_include_directories(portable_simple_cuda_naive_convolution PRIVATE
        "${PORTABLE_INCLUDE_DIR}")
    portable_link_audio_backend(portable_simple_cuda_naive_convolution ${PORTABLE_SIMPLE_CUDA_NAIVE_CONV_IS_MOCK})
    target_compile_definitions(portable_simple_cuda_naive_convolution PRIVATE
        PORTABLE_OUTPUT_DIR=\"${PORTABLE_OUTPUT_DIR_PATH}\"
        PORTABLE_PLOT_SCRIPT=\"${PORTABLE_PLOT_SCRIPT_PATH}\"
        PORTABLE_CALLBACK_TIMING_VIEW_SCRIPT=\"${PORTABLE_CALLBACK_TIMING_VIEW_SCRIPT_PATH}\")
    portable_configure_cuda_target(portable_simple_cuda_naive_convolution)

    add_executable(portable_simple_cuda_less_naive_convolution
        "${PORTABLE_SIMPLE_CUDA_LESS_NAIVE_CONV_SOURCE}")
    target_include_directories(portable_simple_cuda_less_naive_convolution PRIVATE
        "${PORTABLE_INCLUDE_DIR}")
    portable_link_audio_backend(
        portable_simple_cuda_less_naive_convolution
        ${PORTABLE_SIMPLE_CUDA_LESS_NAIVE_CONV_IS_MOCK})
    target_compile_definitions(portable_simple_cuda_less_naive_convolution PRIVATE
        PORTABLE_OUTPUT_DIR=\"${PORTABLE_OUTPUT_DIR_PATH}\"
        PORTABLE_PLOT_SCRIPT=\"${PORTABLE_PLOT_SCRIPT_PATH}\"
        PORTABLE_CALLBACK_TIMING_VIEW_SCRIPT=\"${PORTABLE_CALLBACK_TIMING_VIEW_SCRIPT_PATH}\")
    portable_configure_cuda_target(portable_simple_cuda_less_naive_convolution)
    target_link_libraries(portable_simple_cuda_less_naive_convolution PRIVATE CUDA::cufft)

    add_executable(portable_cuda_device_query
        "${PORTABLE_CUDA_DEVICE_QUERY_SOURCE}")
    target_include_directories(portable_cuda_device_query PRIVATE
        "${PORTABLE_CUDA_SAMPLES_COMMON_DIR}")
    portable_configure_cuda_target(portable_cuda_device_query)
endif()
