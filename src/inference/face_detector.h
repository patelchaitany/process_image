#pragma once

#include "triton_client.h"
#include <vector>
#include <string>

struct RawDetection {
    float cx, cy, w, h;
    float confidence;
};

class FaceDetector {
public:
    explicit FaceDetector(TritonClient& client, const std::string& model_name = "yolo26_face");

    bool detect(const std::string& input_shm_name,
                const std::string& output_shm_name,
                std::vector<RawDetection>& detections,
                float confidence_threshold = 0.5f);

private:
    TritonClient& client_;
    std::string model_name_;
};
