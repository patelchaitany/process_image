#include "cpu_memory_pool.h"
#include <cstring>
#include <cstdio>

bool CPUMemoryPool::init(int frame_width, int frame_height, int max_faces) {
    size_t frame_size = static_cast<size_t>(frame_width) * frame_height * 3;
    size_t yolo_size = 1 * 3 * 640 * 640;
    size_t arcface_size = static_cast<size_t>(max_faces) * 3 * 112 * 112;
    size_t yolo_out_size = 8400 * 5;
    size_t arcface_out_size = static_cast<size_t>(max_faces) * 512;

    buffers_.raw_frame.resize(frame_size);
    buffers_.yolo_input.resize(yolo_size);
    buffers_.arcface_input.resize(arcface_size);
    buffers_.yolo_output.resize(yolo_out_size);
    buffers_.arcface_output.resize(arcface_out_size);

    initialized_ = true;
    return true;
}

void CPUMemoryPool::release() {
    buffers_.raw_frame.clear();
    buffers_.yolo_input.clear();
    buffers_.arcface_input.clear();
    buffers_.yolo_output.clear();
    buffers_.arcface_output.clear();
    initialized_ = false;
}

bool CPUMemoryPool::upload_frame(const uint8_t* data, size_t size) {
    if (!initialized_ || !data) return false;
    if (size > buffers_.raw_frame.size()) return false;
    std::memcpy(buffers_.raw_frame.data(), data, size);
    return true;
}
