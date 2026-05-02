#pragma once

#include "nms.h"
#include <cuda_runtime.h>
#include <vector>

class FaceCropper {
public:
    bool crop_and_preprocess(const void* src_frame_gpu,
                             const std::vector<Detection>& detections,
                             void* dst_batch_gpu,
                             int src_width, int src_height,
                             cudaStream_t stream = nullptr);
};
