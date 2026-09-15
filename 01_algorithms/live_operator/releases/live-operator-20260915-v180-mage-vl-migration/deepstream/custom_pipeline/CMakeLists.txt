cmake_minimum_required(VERSION 3.18)
project(jiankong_custom_pipeline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/rgba_letterbox.cpp" AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/rgba_letterbox_test.cpp")
    add_library(rgba_letterbox_reference STATIC src/rgba_letterbox.cpp)
    target_include_directories(rgba_letterbox_reference PUBLIC include)
    add_executable(rgba_letterbox_test tests/rgba_letterbox_test.cpp)
    target_link_libraries(rgba_letterbox_test PRIVATE rgba_letterbox_reference)
    add_test(NAME rgba_letterbox_test COMMAND rgba_letterbox_test)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime_contract_test.cpp" AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/live_calibration_policy.hpp" AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/live_pipeline_policy.hpp" AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/measurement_window.hpp" AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/rtsp_live_runtime.hpp")
    add_executable(runtime_contract_test tests/runtime_contract_test.cpp)
    target_include_directories(runtime_contract_test PRIVATE include)
    add_test(NAME runtime_contract_test COMMAND runtime_contract_test)
endif()

add_executable(static_phone_spatial_policy_test tests/static_phone_spatial_policy_test.cpp)
target_include_directories(static_phone_spatial_policy_test PRIVATE include)
add_test(NAME static_phone_spatial_policy_test COMMAND static_phone_spatial_policy_test)

add_executable(static_phone_hotspot_test tests/static_phone_hotspot_test.cpp)
target_include_directories(static_phone_hotspot_test PRIVATE include)
target_compile_definitions(static_phone_hotspot_test PRIVATE
    JIANKONG_PIPELINE_SOURCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/src/jiankong_custom_pipeline.cu")
add_test(NAME static_phone_hotspot_test COMMAND static_phone_hotspot_test)

add_executable(gated_event_policy_test tests/gated_event_policy_test.cpp)
target_include_directories(gated_event_policy_test PRIVATE include)
add_test(NAME gated_event_policy_test COMMAND gated_event_policy_test)

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/scripts/build_container_50p2.sh" AND
       EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/docker/Dockerfile.samples-dev")
        add_test(NAME pipeline_contract_test
            COMMAND "${Python3_EXECUTABLE}" -I -B "${CMAKE_CURRENT_SOURCE_DIR}/tests/pipeline_contract_test.py")
    else()
        add_test(NAME pipeline_contract_test
            COMMAND "${Python3_EXECUTABLE}" -I -B -c
                    "compile(open(r'''${CMAKE_CURRENT_SOURCE_DIR}/tests/pipeline_contract_test.py''', encoding='utf-8').read(), r'''pipeline_contract_test.py''', 'exec')")
    endif()
endif()

find_program(BASH_EXECUTABLE bash)
if(BASH_EXECUTABLE AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/deployment_shell_contract_test.sh")
    add_test(NAME deployment_shell_contract_test
        COMMAND "${BASH_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/deployment_shell_contract_test.sh")
endif()

option(JIANKONG_BUILD_DEEPSTREAM_PIPELINE "Build the DeepStream/TensorRT live candidate" OFF)
set(DEEPSTREAM_ROOT "/opt/nvidia/deepstream/deepstream-7.0" CACHE PATH "DeepStream installation root")
set(TENSORRT_ROOT "/usr" CACHE PATH "TensorRT installation prefix")

if(JIANKONG_BUILD_DEEPSTREAM_PIPELINE)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    find_package(OpenCV REQUIRED)
    find_package(Threads REQUIRED)
    find_package(PkgConfig REQUIRED)
    find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp REQUIRED)
    pkg_check_modules(GSTREAMER REQUIRED IMPORTED_TARGET
        gstreamer-1.0
        gstreamer-app-1.0
        gstreamer-video-1.0)

    find_path(DEEPSTREAM_INCLUDE_DIR nvdsmeta.h
        HINTS "${DEEPSTREAM_ROOT}/sources/includes"
        REQUIRED)
    find_library(NVDS_META_LIBRARY NAMES nvds_meta
        HINTS "${DEEPSTREAM_ROOT}/lib"
        REQUIRED)
    find_library(NVDS_GST_META_LIBRARY NAMES nvdsgst_meta
        HINTS "${DEEPSTREAM_ROOT}/lib"
        REQUIRED)
    find_library(NVBUFSURFACE_LIBRARY NAMES nvbufsurface
        HINTS "${DEEPSTREAM_ROOT}/lib"
        REQUIRED)
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS "${TENSORRT_ROOT}/include" "/usr/include/x86_64-linux-gnu"
        REQUIRED)
    find_library(TENSORRT_LIBRARY NAMES nvinfer
        HINTS "${TENSORRT_ROOT}/lib" "${TENSORRT_ROOT}/lib/x86_64-linux-gnu"
              "/usr/lib/x86_64-linux-gnu"
        REQUIRED)

    add_executable(jiankong_custom_pipeline
        src/jiankong_custom_pipeline.cu
        src/deepstream_batch_reader.cpp
        src/rgba_letterbox_cuda.cu
        src/rgba_letterbox.cpp)
    target_include_directories(jiankong_custom_pipeline PRIVATE
        include
        ${OpenCV_INCLUDE_DIRS}
        "${DEEPSTREAM_INCLUDE_DIR}"
        "${TENSORRT_INCLUDE_DIR}"
        "${NLOHMANN_JSON_INCLUDE_DIR}")
    target_link_libraries(jiankong_custom_pipeline PRIVATE
        CUDA::cudart
        "${TENSORRT_LIBRARY}"
        "${NVDS_META_LIBRARY}"
        "${NVDS_GST_META_LIBRARY}"
        "${NVBUFSURFACE_LIBRARY}"
        PkgConfig::GSTREAMER
        ${OpenCV_LIBS}
        Threads::Threads)
    set_target_properties(jiankong_custom_pipeline PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES 86
        BUILD_RPATH "${DEEPSTREAM_ROOT}/lib")
    target_compile_options(jiankong_custom_pipeline PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endif()
