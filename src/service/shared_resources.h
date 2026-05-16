#pragma once

#include <memory>
#include <string>

#include "inference/triton_client.h"
#include "matching/face_database.h"
#include "matching/face_matcher.h"
#include "inference/face_detector.h"
#include "inference/face_recognizer.h"
#include "postprocess/face_cropper.h"

/// @brief Configuration for the video analytics service process.
struct ServiceConfig {
    std::string triton_url = "localhost:8001";
    std::string db_path = "./faces.db";
    std::string device_mode = "auto";
    std::string yolo_model = "yolo26_face";
    std::string arcface_model = "arcface";
    float confidence_threshold = 0.5f;
    float match_threshold = 0.6f;
    float nms_iou_threshold = 0.45f;
    int yolo_input_size = 640;
    int max_faces_per_frame = 32;
    int max_sessions = 30;
    int grpc_port = 50051;
};

/// @brief Owns inference, matching, and GPU helpers shared by all pipeline sessions.
///
/// Holds one Triton client, face database and matcher, detector, recognizer, and cropper.
/// Initialized once per service process for up to ServiceConfig::max_sessions cameras.
struct SharedResources {
    std::unique_ptr<TritonClient> triton_client;
    std::unique_ptr<FaceDatabase> face_database;
    std::unique_ptr<FaceMatcher> face_matcher;
    std::unique_ptr<FaceDetector> face_detector;
    std::unique_ptr<FaceRecognizer> face_recognizer;
    std::unique_ptr<FaceCropper> face_cropper;

    float confidence_threshold = 0.5f;
    float match_threshold = 0.6f;
    float nms_iou_threshold = 0.45f;
    int yolo_input_size = 640;
    int max_faces_per_frame = 32;
    bool use_gpu = true;

    /// @brief Builds clients and loads the face index from configuration.
    /// @param config Service endpoints, thresholds, and model names.
    /// @return true if all steps succeeded.
    bool init(const ServiceConfig& config);

    /// @brief Releases matcher, closes the database, disconnects Triton, and frees objects.
    void shutdown();
};
