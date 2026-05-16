cmake_minimum_required(VERSION 3.18)
project(process_image LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CUDA architecture for T4 (Turing = 75)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_check_modules(LIBAV REQUIRED
    libavcodec
    libavformat
    libavutil
    libswscale
)

find_package(SQLite3 REQUIRED)

find_path(YAML_CPP_INCLUDE_DIR yaml-cpp/yaml.h)
find_library(YAML_CPP_LIBRARY yaml-cpp)

# Triton client (installed via triton client SDK)
find_path(TRITON_CLIENT_INCLUDE_DIR grpc_client.h
    PATHS /opt/tritonserver/client/include
          /usr/local/include/triton
          $ENV{TRITON_CLIENT_DIR}/include
          $ENV{CONDA_PREFIX}/include
)
find_library(TRITON_GRPC_CLIENT grpcclient
    PATHS /opt/tritonserver/client/lib
          /usr/local/lib
          $ENV{TRITON_CLIENT_DIR}/lib
          $ENV{CONDA_PREFIX}/lib
)

# FAISS
find_path(FAISS_INCLUDE_DIR faiss/IndexFlat.h
    PATHS /usr/local/include
          $ENV{FAISS_DIR}/include
          $ENV{CONDA_PREFIX}/include
)
find_library(FAISS_LIBRARY faiss
    PATHS /usr/local/lib
          $ENV{FAISS_DIR}/lib
          $ENV{CONDA_PREFIX}/lib
)
# In FAISS >= 1.12, GPU support is compiled into libfaiss directly (no separate libfaiss_gpu)
find_library(FAISS_GPU_LIBRARY faiss_gpu
    PATHS /usr/local/lib
          $ENV{FAISS_DIR}/lib
          $ENV{CONDA_PREFIX}/lib
)

# OpenCV (optional, for CPU fallback)
find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs dnn)

# spdlog (optional)
find_package(spdlog QUIET)

# Pipeline library sources (shared between CLI and service targets)
set(PIPELINE_SOURCES
    src/frame_source/ffmpeg_source.cpp
    src/frame_source/prefetch_source.cpp
    src/inference/triton_client.cpp
    src/inference/face_detector.cpp
    src/inference/face_recognizer.cpp
    src/matching/face_database.cpp
    src/matching/face_matcher.cpp
    src/matching/cpu_face_matcher.cpp
    src/postprocess/nms.cpp
    src/pipeline/pipeline.cpp
    src/pipeline/result_handler.cpp
    src/metrics/metrics_logger.cpp
    src/gpu/cpu_memory_pool.cpp
    src/gpu/cpu_preprocessor.cpp
    src/output/csv_result_writer.cpp
    src/output/console_result_writer.cpp
)

# CLI-only sources
set(SOURCES
    src/main.cpp
    ${PIPELINE_SOURCES}
)

set(CUDA_SOURCES
    src/gpu/memory_pool.cu
    src/gpu/preprocessor.cu
    src/postprocess/face_cropper.cu
    src/frame_source/nvdec_source.cu
)

add_executable(process_image ${SOURCES} ${CUDA_SOURCES})

target_compile_definitions(process_image PRIVATE TRITON_ENABLE_GPU=1)

target_include_directories(process_image PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${LIBAV_INCLUDE_DIRS}
    ${TRITON_CLIENT_INCLUDE_DIR}
    ${FAISS_INCLUDE_DIR}
    ${YAML_CPP_INCLUDE_DIR}
)

target_link_libraries(process_image PRIVATE
    ${LIBAV_LIBRARIES}
    SQLite::SQLite3
    ${YAML_CPP_LIBRARY}
    ${TRITON_GRPC_CLIENT}
    ${FAISS_LIBRARY}
    CUDA::cudart
    pthread
)

if(FAISS_GPU_LIBRARY)
    target_link_libraries(process_image PRIVATE ${FAISS_GPU_LIBRARY})
endif()

if(OpenCV_FOUND)
    target_compile_definitions(process_image PRIVATE HAS_OPENCV=1)
    target_link_libraries(process_image PRIVATE ${OpenCV_LIBS})
    target_include_directories(process_image PRIVATE ${OpenCV_INCLUDE_DIRS})
endif()

if(YAML_CPP_INCLUDE_DIR AND YAML_CPP_LIBRARY)
    target_compile_definitions(process_image PRIVATE HAS_YAML_CPP=1)
endif()

if(spdlog_FOUND)
    target_compile_definitions(process_image PRIVATE HAS_SPDLOG=1)
    target_link_libraries(process_image PRIVATE spdlog::spdlog)
endif()

target_link_directories(process_image PRIVATE ${LIBAV_LIBRARY_DIRS})

# =============================================================================
# Video Analytics Service (gRPC)
# =============================================================================
# Use pkg-config IMPORTED_TARGET for gRPC/protobuf. This avoids ABI mismatches
# between Conda headers (absl lts_20230125) and system libs (absl lts_20220623).
pkg_check_modules(GRPC_PC IMPORTED_TARGET grpc++)
pkg_check_modules(PROTOBUF_PC IMPORTED_TARGET protobuf)
find_package(CURL QUIET)

if(GRPC_PC_FOUND AND PROTOBUF_PC_FOUND AND CURL_FOUND)
    message(STATUS "Building video_analytics_service")
    message(STATUS "  gRPC includes: ${GRPC_PC_INCLUDE_DIRS}")
    message(STATUS "  gRPC libs:     ${GRPC_PC_LIBRARIES}")
    message(STATUS "  Protobuf libs: ${PROTOBUF_PC_LIBRARIES}")

    set(PROTO_FILE "${CMAKE_SOURCE_DIR}/proto/analytics.proto")
    set(PROTO_GEN_DIR "${CMAKE_BINARY_DIR}/proto_gen")
    file(MAKE_DIRECTORY ${PROTO_GEN_DIR})

    set(PROTO_SRCS "${PROTO_GEN_DIR}/analytics.pb.cc")
    set(PROTO_HDRS "${PROTO_GEN_DIR}/analytics.pb.h")
    set(GRPC_SRCS  "${PROTO_GEN_DIR}/analytics.grpc.pb.cc")
    set(GRPC_HDRS  "${PROTO_GEN_DIR}/analytics.grpc.pb.h")

    find_program(PROTOC_EXEC protoc)
    find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin)

    add_custom_command(
        OUTPUT ${PROTO_SRCS} ${PROTO_HDRS} ${GRPC_SRCS} ${GRPC_HDRS}
        COMMAND ${PROTOC_EXEC}
            --proto_path=${CMAKE_SOURCE_DIR}/proto
            --cpp_out=${PROTO_GEN_DIR}
            --grpc_out=${PROTO_GEN_DIR}
            --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
            ${PROTO_FILE}
        DEPENDS ${PROTO_FILE}
        COMMENT "Generating gRPC/Protobuf sources from analytics.proto"
    )

    set(SERVICE_SOURCES
        src/service_main.cpp
        src/service/shared_resources.cpp
        src/service/pipeline_session.cpp
        src/service/session_manager.cpp
        src/service/grpc_server.cpp
        src/output/webhook_result_writer.cpp
        ${PIPELINE_SOURCES}
        ${PROTO_SRCS}
        ${GRPC_SRCS}
    )

    add_executable(video_analytics_service ${SERVICE_SOURCES} ${CUDA_SOURCES})

    target_compile_definitions(video_analytics_service PRIVATE TRITON_ENABLE_GPU=1)

    # Project + gRPC/protobuf includes first (regular -I, searched first)
    target_include_directories(video_analytics_service BEFORE PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${PROTO_GEN_DIR}
        ${GRPC_PC_INCLUDE_DIRS}
        ${PROTOBUF_PC_INCLUDE_DIRS}
    )
    # Conda/external includes as SYSTEM (-isystem, searched AFTER -I dirs).
    # Prevents Conda's grpc/absl headers from shadowing the system ones.
    target_include_directories(video_analytics_service SYSTEM PRIVATE
        ${LIBAV_INCLUDE_DIRS}
        ${TRITON_CLIENT_INCLUDE_DIR}
        ${FAISS_INCLUDE_DIR}
        ${YAML_CPP_INCLUDE_DIR}
    )

    target_link_libraries(video_analytics_service PRIVATE
        ${LIBAV_LIBRARIES}
        SQLite::SQLite3
        ${YAML_CPP_LIBRARY}
        ${TRITON_GRPC_CLIENT}
        ${FAISS_LIBRARY}
        CUDA::cudart
        pthread
        PkgConfig::GRPC_PC
        PkgConfig::PROTOBUF_PC
        CURL::libcurl
    )

    target_link_directories(video_analytics_service PRIVATE ${LIBAV_LIBRARY_DIRS})

    if(FAISS_GPU_LIBRARY)
        target_link_libraries(video_analytics_service PRIVATE ${FAISS_GPU_LIBRARY})
    endif()

    if(OpenCV_FOUND)
        target_compile_definitions(video_analytics_service PRIVATE HAS_OPENCV=1)
        target_link_libraries(video_analytics_service PRIVATE ${OpenCV_LIBS})
        target_include_directories(video_analytics_service SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
    endif()

    if(YAML_CPP_INCLUDE_DIR AND YAML_CPP_LIBRARY)
        target_compile_definitions(video_analytics_service PRIVATE HAS_YAML_CPP=1)
    endif()
else()
    message(STATUS "Skipping video_analytics_service (requires gRPC, Protobuf, CURL)")
endif()

# Google Test
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(test_nms tests/test_nms.cpp src/postprocess/nms.cpp)
target_include_directories(test_nms PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_nms PRIVATE GTest::gtest_main)

add_executable(test_frame_source tests/test_frame_source.cpp src/frame_source/ffmpeg_source.cpp)
target_include_directories(test_frame_source PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${LIBAV_INCLUDE_DIRS}
)
target_link_libraries(test_frame_source PRIVATE
    GTest::gtest_main
    ${LIBAV_LIBRARIES}
)
target_link_directories(test_frame_source PRIVATE ${LIBAV_LIBRARY_DIRS})

add_executable(test_face_database tests/test_face_database.cpp src/matching/face_database.cpp)
target_include_directories(test_face_database PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_face_database PRIVATE GTest::gtest_main SQLite::SQLite3)

add_executable(test_face_matcher
    tests/test_face_matcher.cpp
    src/matching/face_matcher.cpp
    src/matching/face_database.cpp
)
target_include_directories(test_face_matcher PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${FAISS_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(test_face_matcher PRIVATE
    GTest::gtest_main
    SQLite::SQLite3
    ${FAISS_LIBRARY}
    CUDA::cudart
    CUDA::cublas
)

add_executable(test_triton_client
    tests/test_triton_client.cpp
    src/inference/triton_client.cpp
)
target_compile_definitions(test_triton_client PRIVATE TRITON_ENABLE_GPU=1)
target_include_directories(test_triton_client PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${TRITON_CLIENT_INCLUDE_DIR}
)
target_link_libraries(test_triton_client PRIVATE
    GTest::gtest_main
    ${TRITON_GRPC_CLIENT}
    CUDA::cudart
)

add_executable(test_cpu_preprocessor
    tests/test_cpu_preprocessor.cpp
    src/gpu/cpu_preprocessor.cpp
)
target_include_directories(test_cpu_preprocessor PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_cpu_preprocessor PRIVATE GTest::gtest_main)
if(OpenCV_FOUND)
    target_compile_definitions(test_cpu_preprocessor PRIVATE HAS_OPENCV=1)
    target_link_libraries(test_cpu_preprocessor PRIVATE ${OpenCV_LIBS})
    target_include_directories(test_cpu_preprocessor PRIVATE ${OpenCV_INCLUDE_DIRS})
endif()

include(GoogleTest)
gtest_discover_tests(test_nms)
gtest_discover_tests(test_frame_source)
gtest_discover_tests(test_face_database)
gtest_discover_tests(test_face_matcher)
gtest_discover_tests(test_triton_client)
gtest_discover_tests(test_cpu_preprocessor)
