#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <string>

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

    // Triton CUDA shared memory registration
    bool register_triton_shm(const std::string& triton_url);
    void unregister_triton_shm();

private:
    GPUBuffers buffers_;
    cudaStream_t stream_ = nullptr;
    bool initialized_ = false;
    size_t raw_frame_size_ = 0;
};
