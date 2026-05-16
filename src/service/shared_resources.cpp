#include "service/shared_resources.h"

#include <cstdio>

#include <cuda_runtime.h>

namespace {

bool detectUseGpu(const ServiceConfig& config, bool* useGpu) {
    if (config.device_mode == "cpu") {
        *useGpu = false;
        return true;
    }
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (config.device_mode == "gpu") {
        if (err != cudaSuccess || deviceCount == 0) {
            fprintf(stderr,
                    "SharedResources: device_mode \"gpu\" requires a CUDA device "
                    "(cudaGetDeviceCount failed or returned 0)\n");
            return false;
        }
        *useGpu = true;
        return true;
    }
    *useGpu = (err == cudaSuccess && deviceCount > 0);
    return true;
}

bool connectTriton(SharedResources& res, const ServiceConfig& config) {
    res.triton_client = std::make_unique<TritonClient>();
    if (res.triton_client->connect(config.triton_url, 3, 1000)) {
        return true;
    }
    fprintf(stderr,
            "SharedResources: failed to connect to Triton at %s\n",
            config.triton_url.c_str());
    res.triton_client.reset();
    return false;
}

bool openFaceDatabase(SharedResources& res, const ServiceConfig& config) {
    res.face_database = std::make_unique<FaceDatabase>();
    if (res.face_database->open(config.db_path)) {
        return true;
    }
    fprintf(stderr,
            "SharedResources: failed to open face database at %s\n",
            config.db_path.c_str());
    res.shutdown();
    return false;
}

bool initFaceMatcher(SharedResources& res, const ServiceConfig& config) {
    res.face_matcher = std::make_unique<FaceMatcher>();
    if (res.face_matcher->init(*res.face_database,
                               config.match_threshold,
                               res.use_gpu)) {
        return true;
    }
    fprintf(stderr, "SharedResources: FaceMatcher::init failed\n");
    res.shutdown();
    return false;
}

void createCropAndInferWrappers(SharedResources& res,
                                const ServiceConfig& config) {
    res.face_detector =
        std::make_unique<FaceDetector>(*res.triton_client, config.yolo_model);
    res.face_recognizer =
        std::make_unique<FaceRecognizer>(*res.triton_client,
                                         config.arcface_model);
    res.face_cropper = std::make_unique<FaceCropper>();
}

void copySessionConfig(SharedResources& res, const ServiceConfig& config) {
    res.confidence_threshold = config.confidence_threshold;
    res.match_threshold = config.match_threshold;
    res.nms_iou_threshold = config.nms_iou_threshold;
    res.yolo_input_size = config.yolo_input_size;
    res.max_faces_per_frame = config.max_faces_per_frame;
}

}  // namespace

bool SharedResources::init(const ServiceConfig& config) {
    shutdown();
    bool gpu = false;
    if (!detectUseGpu(config, &gpu)) {
        return false;
    }
    use_gpu = gpu;
    printf("SharedResources: device policy \"%s\" -> %s\n",
           config.device_mode.c_str(),
           use_gpu ? "GPU" : "CPU");
    if (!connectTriton(*this, config)) {
        return false;
    }
    if (!openFaceDatabase(*this, config)) {
        return false;
    }
    const int faces = face_database->count();
    printf("SharedResources: face database contains %d faces\n", faces);
    if (!initFaceMatcher(*this, config)) {
        return false;
    }
    createCropAndInferWrappers(*this, config);
    copySessionConfig(*this, config);
    printf("SharedResources: initialized (Triton=%s, YOLO=%s, ArcFace=%s)\n",
           config.triton_url.c_str(),
           config.yolo_model.c_str(),
           config.arcface_model.c_str());
    return true;
}

void SharedResources::shutdown() {
    if (face_matcher) {
        face_matcher->release();
    }
    face_matcher.reset();
    if (face_database) {
        face_database->close();
    }
    if (triton_client) {
        triton_client->disconnect();
    }
    face_detector.reset();
    face_recognizer.reset();
    face_cropper.reset();
    face_database.reset();
    triton_client.reset();
}
