cmake_minimum_required(VERSION 3.24)
project(fastpipe LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET libavformat libavcodec libavutil)
find_path(TRT_INCLUDE NvInfer.h PATHS /usr/include/x86_64-linux-gnu /usr/include REQUIRED)
find_library(TRT_LIB nvinfer PATHS /usr/lib/x86_64-linux-gnu REQUIRED)

add_executable(fastpipe
  src/main.cpp src/pipeline.cpp src/decoder.cpp src/trt_engine.cpp src/kernels.cu src/gputrack.cu)
target_include_directories(fastpipe PRIVATE ${TRT_INCLUDE} src)
# -lineinfo: Nsight Compute maps samples back to source lines
target_compile_options(fastpipe PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo --use_fast_math -Xptxas=-v>
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wno-deprecated-declarations>)
target_link_libraries(fastpipe PRIVATE CUDA::cudart CUDA::cuda_driver ${TRT_LIB} PkgConfig::FFMPEG pthread dl)
