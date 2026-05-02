#pragma once

#include <cstdint>
#include <vector>

struct CPUPreprocessParams {
    int src_width;
    int src_height;
    int dst_size;
    float scale;
    int pad_x;
    int pad_y;
};

class CPUPreprocessor {
public:
    bool preprocess_yolo(const uint8_t* src_bgr, float* dst_chw,
                         int src_width, int src_height, int dst_size);

    bool preprocess_arcface(const uint8_t* src_bgr, float* dst_chw,
                            const float* bboxes, int num_faces,
                            int src_width, int src_height);

    CPUPreprocessParams last_params() const { return last_params_; }

private:
    CPUPreprocessParams last_params_{};
};
