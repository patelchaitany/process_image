#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct CPUBuffers {
    std::vector<uint8_t> raw_frame;
    std::vector<float> yolo_input;      // [1,3,640,640]
    std::vector<float> arcface_input;   // [M,3,112,112]
    std::vector<float> yolo_output;
    std::vector<float> arcface_output;  // [M,512]
};

class CPUMemoryPool {
public:
    CPUMemoryPool() = default;

    bool init(int frame_width, int frame_height, int max_faces);
    void release();

    bool upload_frame(const uint8_t* data, size_t size);

    CPUBuffers& buffers() { return buffers_; }
    const CPUBuffers& buffers() const { return buffers_; }

    bool register_system_shm(const std::string& triton_url);
    void unregister_system_shm();

private:
    CPUBuffers buffers_;
    bool initialized_ = false;
};
