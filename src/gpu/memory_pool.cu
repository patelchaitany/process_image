#include "memory_pool.h"
#include <cstdio>

GPUMemoryPool::~GPUMemoryPool() {
    release();
}

bool GPUMemoryPool::init(int frame_width, int frame_height, int max_faces) {
    if (initialized_) return true;

    cudaError_t err = cudaStreamCreate(&stream_);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to create CUDA stream: %s\n", cudaGetErrorString(err));
        return false;
    }

    raw_frame_size_ = static_cast<size_t>(frame_width) * frame_height * 3;
    size_t yolo_size = 1 * 3 * 640 * 640 * sizeof(float);
    size_t arcface_size = static_cast<size_t>(max_faces) * 3 * 112 * 112 * sizeof(float);
    size_t yolo_out_size = 8400 * 5 * sizeof(float);
    size_t arcface_out_size = static_cast<size_t>(max_faces) * 512 * sizeof(float);

    err = cudaMalloc(&buffers_.raw_frame, raw_frame_size_);
    if (err != cudaSuccess) goto fail;

    err = cudaMalloc(&buffers_.yolo_input, yolo_size);
    if (err != cudaSuccess) goto fail;

    err = cudaMalloc(&buffers_.arcface_input, arcface_size);
    if (err != cudaSuccess) goto fail;

    err = cudaMalloc(&buffers_.yolo_output, yolo_out_size);
    if (err != cudaSuccess) goto fail;

    err = cudaMalloc(&buffers_.arcface_output, arcface_out_size);
    if (err != cudaSuccess) goto fail;

    initialized_ = true;
    return true;

fail:
    fprintf(stderr, "CUDA allocation failed: %s\n", cudaGetErrorString(err));
    release();
    return false;
}

void GPUMemoryPool::release() {
    if (buffers_.raw_frame) { cudaFree(buffers_.raw_frame); buffers_.raw_frame = nullptr; }
    if (buffers_.yolo_input) { cudaFree(buffers_.yolo_input); buffers_.yolo_input = nullptr; }
    if (buffers_.arcface_input) { cudaFree(buffers_.arcface_input); buffers_.arcface_input = nullptr; }
    if (buffers_.yolo_output) { cudaFree(buffers_.yolo_output); buffers_.yolo_output = nullptr; }
    if (buffers_.arcface_output) { cudaFree(buffers_.arcface_output); buffers_.arcface_output = nullptr; }
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
    initialized_ = false;
}

bool GPUMemoryPool::upload_frame(const uint8_t* cpu_data, size_t size, cudaStream_t s) {
    if (!initialized_ || !cpu_data) return false;
    cudaStream_t use_stream = s ? s : stream_;
    cudaError_t err = cudaMemcpyAsync(buffers_.raw_frame, cpu_data, size,
                                       cudaMemcpyHostToDevice, use_stream);
    return err == cudaSuccess;
}

bool GPUMemoryPool::register_triton_shm(const std::string& /*triton_url*/) {
    // Will be implemented in Phase 3 with TritonClient
    return true;
}

void GPUMemoryPool::unregister_triton_shm() {
    // Will be implemented in Phase 3
}
