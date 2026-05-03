#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <string>

class TritonClient;

struct GPUBuffers {
    void* raw_frame = nullptr;       // 1920*1080*3 uint8
    void* yolo_input = nullptr;      // 1*3*640*640 float32
    void* arcface_input = nullptr;   // M*3*112*112 float32
    void* yolo_output = nullptr;     // output tensor
    void* arcface_output = nullptr;  // M*512 float32
};

class GPUMemoryPool {
public:
    GPUMemoryPool() = default;
    ~GPUMemoryPool();

    bool init(int frame_width, int frame_height, int max_faces);
    void release();

    bool upload_frame(const uint8_t* cpu_data, size_t size, cudaStream_t stream = nullptr);

    GPUBuffers& buffers() { return buffers_; }
    const GPUBuffers& buffers() const { return buffers_; }

    cudaStream_t stream() const { return stream_; }

    /// @brief Register all GPU I/O buffers as CUDA shared memory with Triton.
    bool register_triton_shm(TritonClient& client);

    /// @brief Unregister all CUDA shared memory regions from Triton.
    void unregister_triton_shm(TritonClient& client);

private:
    GPUBuffers buffers_;
    cudaStream_t stream_ = nullptr;
    bool initialized_ = false;
    size_t raw_frame_size_ = 0;
    int max_faces_ = 0;

    uint8_t* pinned_staging_ = nullptr;

    size_t yolo_input_bytes_ = 0;
    size_t yolo_output_bytes_ = 0;
    size_t arcface_input_bytes_ = 0;
    size_t arcface_output_bytes_ = 0;
    bool shm_registered_ = false;
};
