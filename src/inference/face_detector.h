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

    /// @brief Run detection via CUDA shared memory (GPU path).
    bool detect(const std::string& input_shm_name,
                const std::string& output_shm_name,
                std::vector<RawDetection>& detections,
                float confidence_threshold = 0.5f,
                InferResult* outStats = nullptr);

    /// @brief Run detection with inline data (CPU path).
    bool detectDirect(const float* inputData,
                      std::vector<RawDetection>& detections,
                      float confidence_threshold = 0.5f,
                      InferResult* outStats = nullptr);

private:
    TritonClient& client_;
    std::string model_name_;
    static constexpr int NUM_DETECTIONS = 8400;
    static constexpr int DETECTION_STRIDE = 5;

    void parseDetections(const InferResult& result,
                         std::vector<RawDetection>& detections,
                         float confidence_threshold);
};
