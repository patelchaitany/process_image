#pragma once

#include <vector>

struct Detection {
    float x1, y1, x2, y2;
    float confidence;
};

std::vector<Detection> apply_nms(const std::vector<Detection>& detections,
                                  float iou_threshold = 0.45f);

std::vector<Detection> filter_and_decode(const float* raw_output,
                                          int num_detections,
                                          float confidence_threshold,
                                          int model_size,
                                          int orig_width,
                                          int orig_height,
                                          float scale, int pad_x, int pad_y);
