#include "memory_pool.h"
#include "inference/triton_client.h"
#include <cstdio>
#include <cstring>

GPUMemoryPool::~GPUMemoryPool() {
    release();
}

bool GPUMemoryPool::init(int frame_width, int frame_height, int max_faces) {
    if (initialized_) return true;
    max_faces_ = max_faces;

    cudaError_t err = cudaStreamCreate(&stream_);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to create CUDA stream: %s\n", cudaGetErrorString(err));
        return false;
    }

    raw_frame_size_ = static_cast<size_t>(frame_width) * frame_height * 3;
    yolo_input_bytes_ = 1 * 3 * 640 * 640 * sizeof(float);
    arcface_input_bytes_ = static_cast<size_t>(max_faces) * 3 * 112 * 112 * sizeof(float);
    yolo_output_bytes_ = 8400 * 5 * sizeof(float);
    arcface_output_bytes_ = static_cast<size_t>(max_faces) * 512 * sizeof(float);

    err = cudaMalloc(&buffers_.raw_frame, raw_frame_size_);
    if (err != cudaSuccess) goto fail;

    err = cudaMalloc(&buffers_.yolo_input, yolo_input_bytes_);
    if (err != cudaSuccess) goto fail;

    err = cudaMalloc(&buffers_.arcface_input, arcface_input_bytes_);
    if (err != cudaSuccess) goto fail;

    err = cudaMalloc(&buffers_.yolo_output, yolo_output_bytes_);
    if (err != cudaSuccess) goto fail;

    err = cudaMalloc(&buffers_.arcface_output, arcface_output_bytes_);
    if (err != cudaSuccess) goto fail;

    err = cudaMallocHost(&pinned_staging_, raw_frame_size_);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate pinned staging buffer (%zu bytes): %s\n",
                raw_frame_size_, cudaGetErrorString(err));
        goto fail;
    }

    initialized_ = true;
    return true;

fail:
    fprintf(stderr, "CUDA allocation failed: %s\n", cudaGetErrorString(err));
    release();
    return false;
}

void GPUMemoryPool::release() {
    if (pinned_staging_) { cudaFreeHost(pinned_staging_); pinned_staging_ = nullptr; }
    if (buffers_.raw_frame) { cudaFree(buffers_.raw_frame); buffers_.raw_frame = nullptr; }
    if (buffers_.yolo_input) { cudaFree(buffers_.yolo_input); buffers_.yolo_input = nullptr; }
    if (buffers_.arcface_input) { cudaFree(buffers_.arcface_input); buffers_.arcface_input = nullptr; }
    if (buffers_.yolo_output) { cudaFree(buffers_.yolo_output); buffers_.yolo_output = nullptr; }
    if (buffers_.arcface_output) { cudaFree(buffers_.arcface_output); buffers_.arcface_output = nullptr; }
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
    initialized_ = false;
    shm_registered_ = false;
}

bool GPUMemoryPool::upload_frame(const uint8_t* cpu_data, size_t size, cudaStream_t s) {
    if (!initialized_ || !cpu_data) return false;
    if (size > raw_frame_size_) return false;

    cudaStream_t use_stream = s ? s : stream_;

    std::memcpy(pinned_staging_, cpu_data, size);

    cudaError_t err = cudaMemcpyAsync(buffers_.raw_frame, pinned_staging_, size,
                                       cudaMemcpyHostToDevice, use_stream);
    return err == cudaSuccess;
}

bool GPUMemoryPool::register_triton_shm(TritonClient& client) {
    if (!initialized_ || shm_registered_) return shm_registered_;

    bool ok = true;
    ok &= client.registerCudaShm("yolo_input_shm",   buffers_.yolo_input,   yolo_input_bytes_);
    ok &= client.registerCudaShm("yolo_output_shm",  buffers_.yolo_output,  yolo_output_bytes_);
    ok &= client.registerCudaShm("arcface_input_shm", buffers_.arcface_input, arcface_input_bytes_);
    ok &= client.registerCudaShm("arcface_output_shm", buffers_.arcface_output, arcface_output_bytes_);

    if (!ok) {
        fprintf(stderr, "GPUMemoryPool: partial SHM registration failure, unregistering all\n");
        client.unregisterAllShm();
        return false;
    }

    shm_registered_ = true;
    fprintf(stderr, "GPUMemoryPool: all 4 CUDA SHM regions registered with Triton\n");
    return true;
}

void GPUMemoryPool::unregister_triton_shm(TritonClient& client) {
    if (!shm_registered_) return;
    client.unregisterAllShm();
    shm_registered_ = false;
}
