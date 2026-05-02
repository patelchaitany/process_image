#include "face_detector.h"

FaceDetector::FaceDetector(TritonClient& client, const std::string& model_name)
    : client_(client), model_name_(model_name) {}

void FaceDetector::parseDetections(const InferResult& result,
                                    std::vector<RawDetection>& detections,
                                    float confidence_threshold) {
    detections.clear();
    const float* data = result.outputData;
    if (!data) return;

    int numDetections = static_cast<int>(result.outputShape[1]);
    for (int i = 0; i < numDetections; ++i) {
        float conf = data[i * DETECTION_STRIDE + 4];
        if (conf >= confidence_threshold) {
            RawDetection det;
            det.cx = data[i * DETECTION_STRIDE + 0];
            det.cy = data[i * DETECTION_STRIDE + 1];
            det.w  = data[i * DETECTION_STRIDE + 2];
            det.h  = data[i * DETECTION_STRIDE + 3];
            det.confidence = conf;
            detections.push_back(det);
        }
    }
}

bool FaceDetector::detect(const std::string& input_shm_name,
                           const std::string& output_shm_name,
                           std::vector<RawDetection>& detections,
                           float confidence_threshold,
                           InferResult* outStats) {
    std::vector<int64_t> inputShape = {1, 3, 640, 640};
    std::vector<int64_t> outputShape = {1, NUM_DETECTIONS, DETECTION_STRIDE};

    InferResult result = client_.infer(model_name_, input_shm_name,
                                        inputShape, output_shm_name, outputShape);
    if (outStats) *outStats = result;
    if (!result.isSuccess) return false;

    parseDetections(result, detections, confidence_threshold);
    return true;
}

bool FaceDetector::detectDirect(const float* inputData,
                                 std::vector<RawDetection>& detections,
                                 float confidence_threshold,
                                 InferResult* outStats) {
    std::vector<int64_t> inputShape = {1, 3, 640, 640};
    std::vector<int64_t> outputShape = {1, NUM_DETECTIONS, DETECTION_STRIDE};

    InferResult result = client_.inferDirect(model_name_, inputData,
                                              inputShape, outputShape);
    if (outStats) *outStats = result;
    if (!result.isSuccess) return false;

    parseDetections(result, detections, confidence_threshold);
    return true;
}
