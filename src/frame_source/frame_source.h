#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct Frame {
    std::vector<uint8_t> data;  // BGR24 interleaved (CPU path)
    void* gpu_data = nullptr;   // BGR24 on GPU (NVDEC path, owned by source)
    bool on_gpu = false;
    int width = 0;
    int height = 0;
    int channels = 3;
    uint64_t frame_index = 0;

    size_t size_bytes() const {
        if (on_gpu) return static_cast<size_t>(width) * height * channels;
        return data.size();
    }
    bool empty() const {
        if (on_gpu) return gpu_data == nullptr;
        return data.empty();
    }
};

class FrameSource {
public:
    virtual ~FrameSource() = default;
    virtual bool open(const std::string& source) = 0;
    virtual bool read(Frame& frame) = 0;
    virtual void close() = 0;
    virtual int width() const = 0;
    virtual int height() const = 0;
    virtual double fps() const = 0;
    virtual bool is_open() const = 0;
};
