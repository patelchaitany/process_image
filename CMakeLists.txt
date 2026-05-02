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

# Source files
set(SOURCES
    src/main.cpp
    src/frame_source/ffmpeg_source.cpp
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
)

set(CUDA_SOURCES
    src/gpu/memory_pool.cu
    src/gpu/preprocessor.cu
    src/postprocess/face_cropper.cu
)

add_executable(process_image ${SOURCES} ${CUDA_SOURCES})

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

if(spdlog_FOUND)
    target_compile_definitions(process_image PRIVATE HAS_SPDLOG=1)
    target_link_libraries(process_image PRIVATE spdlog::spdlog)
endif()

target_link_directories(process_image PRIVATE ${LIBAV_LIBRARY_DIRS})

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
)
target_link_libraries(test_face_matcher PRIVATE
    GTest::gtest_main
    SQLite::SQLite3
    ${FAISS_LIBRARY}
)

include(GoogleTest)
gtest_discover_tests(test_nms)
gtest_discover_tests(test_frame_source)
gtest_discover_tests(test_face_database)
gtest_discover_tests(test_face_matcher)
