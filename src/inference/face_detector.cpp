#include "face_detector.h"

FaceDetector::FaceDetector(TritonClient& client, const std::string& model_name)
    : client_(client), model_name_(model_name) {}

bool FaceDetector::detect(const std::string& input_shm_name,
                           const std::string& output_shm_name,
                           std::vector<RawDetection>& detections,
                           float confidence_threshold) {
    std::vector<int64_t> inputShape = {1, 3, 640, 640};
    std::vector<int64_t> outputShape = {1, 8400, 5};
    InferResult result = client_.infer(model_name_, input_shm_name,
                                        inputShape, output_shm_name, outputShape);
    if (!result.isSuccess) return false;
    detections.clear();
    const float* data = result.outputData;
    if (!data) return true;
    int numDetections = static_cast<int>(result.outputShape[1]);
    for (int i = 0; i < numDetections; ++i) {
        float conf = data[i * 5 + 4];
        if (conf >= confidence_threshold) {
            RawDetection det;
            det.cx = data[i * 5 + 0];
            det.cy = data[i * 5 + 1];
            det.w = data[i * 5 + 2];
            det.h = data[i * 5 + 3];
            det.confidence = conf;
            detections.push_back(det);
        }
    }
    return true;
}
