#pragma once

#include "service/pipeline_session.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

/// @brief Outcome of attempting to start a new video pipeline session.
struct StartResult {
    std::string session_id;
    bool success = false;
    std::string error;
};

/// @brief Outcome of stopping an existing session; includes the final processed frame count.
struct StopResult {
    bool success = false;
    uint64_t frames_processed = 0;
};

/// @brief Outcome of registering a new face into the gallery.
struct RegisterFaceResult {
    bool success = false;
    int64_t person_id = -1;
    std::string error;
};

/// @brief A single detected face bounding box and confidence.
struct DetectedFace {
    float x1, y1, x2, y2;
    float confidence;
};

/// @brief Outcome of running face detection on a single image.
struct DetectFacesResult {
    bool success = false;
    std::vector<DetectedFace> detections;
    int image_width = 0;
    int image_height = 0;
    std::string error;
};

/// @brief Manages concurrent @ref PipelineSession instances and owns @ref SharedResources.
///
/// Thread-safe entry point for gRPC workers: caps session count, assigns UUID session ids,
/// and coordinates shared inference and database resources.
class SessionManager {
public:
    SessionManager() = default;
    ~SessionManager();

    /// @brief Initializes shared resources (Triton, face DB, matcher).
    bool init(const ServiceConfig& config);

    /// @brief Creates and starts a new pipeline session.
    StartResult startSession(const std::string& source_uri,
                             const std::string& callback_url,
                             const std::string& session_name,
                             const SessionConfig& config);

    /// @brief Stops and removes a session.
    StopResult stopSession(const std::string& session_id);

    /// @brief Returns status of all sessions.
    std::vector<SessionStatus> listSessions() const;

    /// @brief Returns status of a single session.
    /// @return Status with state="unknown" if session not found.
    SessionStatus getSessionStatus(const std::string& session_id) const;

    /// @brief Runs face detection on an encoded image and returns bounding boxes.
    /// No database writes or embedding extraction — purely detection + NMS.
    DetectFacesResult detectFaces(const std::string& image_data,
                                  float confidence_threshold = 0.0f);

    /// @brief Registers a face from an encoded image (JPEG/PNG).
    /// Runs YOLO detection + ArcFace embedding via Triton, writes to DB, and
    /// updates the live FAISS index. The image must contain exactly one face.
    RegisterFaceResult registerFace(const std::string& name,
                                    const std::string& image_data);

    /// @brief Stops all sessions and shuts down shared resources.
    void shutdownAll();

private:
    ServiceConfig config_;
    SharedResources shared_;
    std::unordered_map<std::string, std::unique_ptr<PipelineSession>> sessions_;
    mutable std::mutex sessionsMutex_;
    std::mutex registrationMutex_;

    std::string generateSessionId() const;

    /// @brief Shared helper: decode image + run YOLO + NMS, returns detections
    /// in original image coordinates. Used by both detectFaces() and registerFace().
    struct ImageDetections {
        std::vector<Detection> detections;
        std::vector<uint8_t> bgr_data;
        int width = 0;
        int height = 0;
        bool ok = false;
        std::string error;
    };
    ImageDetections runDetection(const std::string& image_data,
                                 float confidence_threshold);
};
