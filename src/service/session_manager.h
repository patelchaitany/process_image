#pragma once

#include "service/pipeline_session.h"

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

    /// @brief Stops all sessions and shuts down shared resources.
    void shutdownAll();

private:
    ServiceConfig config_;
    SharedResources shared_;
    std::unordered_map<std::string, std::unique_ptr<PipelineSession>> sessions_;
    mutable std::mutex sessionsMutex_;

    std::string generateSessionId() const;
};
