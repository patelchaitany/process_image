#pragma once

#include <cuda_runtime.h>
#include <cstdint>

struct PreprocessParams {
    int src_width;
    int src_height;
    int dst_size;       // square output (640 for YOLO)
    float scale;        // letterbox scale factor
    int pad_x;          // letterbox x offset
    int pad_y;          // letterbox y offset
};

class Preprocessor {
public:
    bool preprocess_yolo(const void* src_bgr_gpu, void* dst_chw_gpu,
                         int src_width, int src_height, int dst_size,
                         cudaStream_t stream = nullptr);

    bool preprocess_arcface(const void* src_bgr_gpu, void* dst_chw_gpu,
                            const float* bboxes, int num_faces,
                            int src_width, int src_height,
                            cudaStream_t stream = nullptr);

    PreprocessParams last_params() const { return last_params_; }

private:
    PreprocessParams last_params_{};
};
