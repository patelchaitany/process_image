#include "service/session_manager.h"
#include "gpu/cpu_preprocessor.h"
#include "postprocess/nms.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#ifdef HAS_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#endif

namespace {

std::string formatUuidBytes(const unsigned char bytes[16]) {
    char buf[37];
    snprintf(buf,
             sizeof(buf),
             "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
             bytes[0],
             bytes[1],
             bytes[2],
             bytes[3],
             bytes[4],
             bytes[5],
             bytes[6],
             bytes[7],
             bytes[8],
             bytes[9],
             bytes[10],
             bytes[11],
             bytes[12],
             bytes[13],
             bytes[14],
             bytes[15]);
    return std::string(buf);
}

}  // namespace

SessionManager::~SessionManager() {
    shutdownAll();
}

bool SessionManager::init(const ServiceConfig& config) {
    config_ = config;
    return shared_.init(config);
}

std::string SessionManager::generateSessionId() const {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned int> octet(0, 255);
    unsigned char bytes[16];
    for (int i = 0; i < 16; ++i) {
        bytes[i] = static_cast<unsigned char>(octet(gen));
    }
    bytes[6] = static_cast<unsigned char>((bytes[6] & 0x0fU) | 0x40U);
    bytes[8] = static_cast<unsigned char>((bytes[8] & 0x3fU) | 0x80U);
    return formatUuidBytes(bytes);
}

StartResult SessionManager::startSession(const std::string& source_uri,
                                         const std::string& callback_url,
                                         const std::string& session_name,
                                         const SessionConfig& config) {
    std::lock_guard<std::mutex> lock(sessionsMutex_);
    StartResult result;
    if (static_cast<int>(sessions_.size()) >= config_.max_sessions) {
        result.success = false;
        result.error = "max_sessions limit reached";
        return result;
    }
    std::string session_id;
    bool unique = false;
    for (int n = 0; n < 64 && !unique; ++n) {
        session_id = generateSessionId();
        unique = (sessions_.find(session_id) == sessions_.end());
    }
    if (!unique) {
        result.success = false;
        result.error = "failed to allocate unique session id";
        return result;
    }
    auto session = std::make_unique<PipelineSession>(session_id,
                                                     session_name,
                                                     source_uri,
                                                     callback_url,
                                                     config,
                                                     shared_);
    if (!session->start()) {
        SessionStatus st = session->status();
        result.success = false;
        result.error = st.error_message.empty() ? "session start failed"
                                                : st.error_message;
        return result;
    }
    sessions_.emplace(session_id, std::move(session));
    result.session_id = session_id;
    result.success = true;
    return result;
}

StopResult SessionManager::stopSession(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(sessionsMutex_);
    StopResult result;
    auto it = sessions_.find(session_id);
    if (it == sessions_.end()) {
        result.success = false;
        result.frames_processed = 0;
        return result;
    }
    uint64_t frames = it->second->status().frames_processed;
    it->second->stop();
    sessions_.erase(it);
    result.success = true;
    result.frames_processed = frames;
    return result;
}

std::vector<SessionStatus> SessionManager::listSessions() const {
    std::lock_guard<std::mutex> lock(sessionsMutex_);
    std::vector<SessionStatus> out;
    out.reserve(sessions_.size());
    for (const auto& entry : sessions_) {
        out.push_back(entry.second->status());
    }
    return out;
}

SessionStatus SessionManager::getSessionStatus(const std::string& session_id) const {
    std::lock_guard<std::mutex> lock(sessionsMutex_);
    auto it = sessions_.find(session_id);
    if (it == sessions_.end()) {
        SessionStatus unknown;
        unknown.session_id = session_id;
        unknown.state = "unknown";
        return unknown;
    }
    return it->second->status();
}

SessionManager::ImageDetections SessionManager::runDetection(
        const std::string& image_data, float confidence_threshold) {
    ImageDetections det;

#ifndef HAS_OPENCV
    det.error = "server built without OpenCV; image-based RPCs require OpenCV";
    return det;
#else
    std::vector<uint8_t> buf(image_data.begin(), image_data.end());
    cv::Mat image = cv::imdecode(buf, cv::IMREAD_COLOR);
    if (image.empty()) {
        det.error = "failed to decode image (expected JPEG or PNG)";
        return det;
    }

    det.width = image.cols;
    det.height = image.rows;
    int yolo_size = shared_.yolo_input_size;

    float conf = (confidence_threshold > 0.0f) ? confidence_threshold
                                               : shared_.confidence_threshold;

    CPUPreprocessor preprocessor;
    std::vector<float> yolo_input(3 * yolo_size * yolo_size);
    if (!preprocessor.preprocess_yolo(image.data, yolo_input.data(),
                                      det.width, det.height, yolo_size)) {
        det.error = "YOLO preprocessing failed";
        return det;
    }
    CPUPreprocessParams params = preprocessor.last_params();

    std::vector<RawDetection> raw_dets;
    if (!shared_.face_detector->detectDirect(yolo_input.data(), raw_dets, conf)) {
        det.error = "face detection inference failed";
        return det;
    }

    det.detections.reserve(raw_dets.size());
    for (const auto& rd : raw_dets) {
        Detection d;
        d.x1 = (rd.cx - rd.w / 2.0f - params.pad_x) / params.scale;
        d.y1 = (rd.cy - rd.h / 2.0f - params.pad_y) / params.scale;
        d.x2 = (rd.cx + rd.w / 2.0f - params.pad_x) / params.scale;
        d.y2 = (rd.cy + rd.h / 2.0f - params.pad_y) / params.scale;
        d.x1 = std::max(0.0f, std::min(d.x1, static_cast<float>(det.width)));
        d.y1 = std::max(0.0f, std::min(d.y1, static_cast<float>(det.height)));
        d.x2 = std::max(0.0f, std::min(d.x2, static_cast<float>(det.width)));
        d.y2 = std::max(0.0f, std::min(d.y2, static_cast<float>(det.height)));
        d.confidence = rd.confidence;
        if (d.x2 > d.x1 && d.y2 > d.y1)
            det.detections.push_back(d);
    }

    det.detections = apply_nms(det.detections, shared_.nms_iou_threshold);
    det.bgr_data.assign(image.data, image.data + image.total() * image.elemSize());
    det.ok = true;
    return det;
#endif  // HAS_OPENCV
}

DetectFacesResult SessionManager::detectFaces(const std::string& image_data,
                                               float confidence_threshold) {
    DetectFacesResult result;

    if (image_data.empty()) {
        result.error = "image_data is required";
        return result;
    }

    ImageDetections det = runDetection(image_data, confidence_threshold);
    if (!det.ok) {
        result.error = det.error;
        return result;
    }

    result.image_width = det.width;
    result.image_height = det.height;
    result.detections.reserve(det.detections.size());
    for (const auto& d : det.detections) {
        result.detections.push_back({d.x1, d.y1, d.x2, d.y2, d.confidence});
    }

    result.success = true;
    return result;
}

RegisterFaceResult SessionManager::registerFace(const std::string& name,
                                                 const std::string& image_data) {
    std::lock_guard<std::mutex> lock(registrationMutex_);
    RegisterFaceResult result;

    if (name.empty()) {
        result.error = "name is required";
        return result;
    }
    if (image_data.empty()) {
        result.error = "image_data is required";
        return result;
    }

    ImageDetections det = runDetection(image_data, 0.0f);
    if (!det.ok) {
        result.error = det.error;
        return result;
    }

    if (det.detections.empty()) {
        result.error = "no face detected in image";
        return result;
    }
    if (det.detections.size() > 1) {
        result.error = "multiple faces detected (" +
                       std::to_string(det.detections.size()) +
                       "); image must contain exactly one face";
        return result;
    }

#ifndef HAS_OPENCV
    result.error = "server built without OpenCV";
    return result;
#else
    // Preprocess face crop for ArcFace (112x112, normalized)
    CPUPreprocessor preprocessor;
    float bbox[4] = {det.detections[0].x1, det.detections[0].y1,
                     det.detections[0].x2, det.detections[0].y2};
    std::vector<float> arcface_input(3 * 112 * 112);
    if (!preprocessor.preprocess_arcface(det.bgr_data.data(), arcface_input.data(),
                                         bbox, 1, det.width, det.height)) {
        result.error = "ArcFace preprocessing failed";
        return result;
    }

    std::vector<std::vector<float>> embeddings;
    if (!shared_.face_recognizer->recognizeDirect(arcface_input.data(), 1, embeddings)) {
        result.error = "face recognition inference failed";
        return result;
    }
    if (embeddings.empty() || embeddings[0].empty()) {
        result.error = "embedding extraction returned empty result";
        return result;
    }

    const std::vector<float>& embedding = embeddings[0];

    int64_t new_id = -1;
    if (!shared_.face_database->add_face(name, embedding, &new_id)) {
        result.error = "failed to write face to database";
        return result;
    }

    if (!shared_.face_matcher->addFace(new_id, name, embedding)) {
        result.error = "face saved to database but failed to update live index; "
                       "restart the service to pick it up";
        result.person_id = new_id;
        return result;
    }

    result.success = true;
    result.person_id = new_id;
    fprintf(stderr, "RegisterFace: enrolled \"%s\" as person_id=%lld\n",
            name.c_str(), static_cast<long long>(new_id));
    return result;
#endif  // HAS_OPENCV
}

void SessionManager::shutdownAll() {
    std::lock_guard<std::mutex> lock(sessionsMutex_);
    for (auto& entry : sessions_) {
        entry.second->stop();
    }
    sessions_.clear();
    shared_.shutdown();
}
