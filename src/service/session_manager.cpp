#include "service/session_manager.h"

#include <cstdio>
#include <random>

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

void SessionManager::shutdownAll() {
    std::lock_guard<std::mutex> lock(sessionsMutex_);
    for (auto& entry : sessions_) {
        entry.second->stop();
    }
    sessions_.clear();
    shared_.shutdown();
}
