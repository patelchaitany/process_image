#include "nms.h"
#include <algorithm>
#include <cmath>

static float compute_iou(const Detection& a, const Detection& b) {
    float inter_x1 = std::max(a.x1, b.x1);
    float inter_y1 = std::max(a.y1, b.y1);
    float inter_x2 = std::min(a.x2, b.x2);
    float inter_y2 = std::min(a.y2, b.y2);

    float inter_w = std::max(0.0f, inter_x2 - inter_x1);
    float inter_h = std::max(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;

    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    float union_area = area_a + area_b - inter_area;

    return (union_area > 0) ? inter_area / union_area : 0.0f;
}

std::vector<Detection> apply_nms(const std::vector<Detection>& detections,
                                  float iou_threshold) {
    if (detections.empty()) return {};

    std::vector<int> indices(detections.size());
    for (size_t i = 0; i < indices.size(); ++i) indices[i] = static_cast<int>(i);

    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return detections[a].confidence > detections[b].confidence;
    });

    std::vector<bool> suppressed(detections.size(), false);
    std::vector<Detection> result;

    for (int idx : indices) {
        if (suppressed[idx]) continue;
        result.push_back(detections[idx]);

        for (int other : indices) {
            if (suppressed[other] || other == idx) continue;
            if (compute_iou(detections[idx], detections[other]) > iou_threshold) {
                suppressed[other] = true;
            }
        }
    }

    return result;
}

std::vector<Detection> filter_and_decode(const float* raw_output,
                                          int num_detections,
                                          float confidence_threshold,
                                          int model_size,
                                          int orig_width, int orig_height,
                                          float scale, int pad_x, int pad_y) {
    std::vector<Detection> candidates;

    for (int i = 0; i < num_detections; ++i) {
        float conf = raw_output[i * 5 + 4];
        if (conf < confidence_threshold) continue;

        float cx = raw_output[i * 5 + 0];
        float cy = raw_output[i * 5 + 1];
        float w = raw_output[i * 5 + 2];
        float h = raw_output[i * 5 + 3];

        // cxcywh in model coords -> xyxy in original image coords
        float x1 = (cx - w / 2.0f - pad_x) / scale;
        float y1 = (cy - h / 2.0f - pad_y) / scale;
        float x2 = (cx + w / 2.0f - pad_x) / scale;
        float y2 = (cy + h / 2.0f - pad_y) / scale;

        // Clip to image bounds
        x1 = std::max(0.0f, std::min(x1, static_cast<float>(orig_width)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(orig_height)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(orig_width)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(orig_height)));

        if (x2 > x1 && y2 > y1) {
            candidates.push_back({x1, y1, x2, y2, conf});
        }
    }

    return candidates;
}
