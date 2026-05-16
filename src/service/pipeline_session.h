#pragma once

#include "service/shared_resources.h"
#include "frame_source/frame_source.h"
#include "inference/face_detector.h"
#include "matching/face_matcher.h"
#include "output/result_writer.h"
#include "postprocess/nms.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

class FrameSource;
class GPUMemoryPool;
class Preprocessor;
class CPUMemoryPool;
class CPUPreprocessor;
class CPUFaceMatcher;
class ResultHandler;

/// @brief Per-session thresholds and throughput limits for one pipeline session.
struct SessionConfig {
    float confidence_threshold = 0.0f;  ///< Override confidence; @c 0 uses service default.
    float match_threshold = 0.0f;       ///< Override match threshold; @c 0 uses service default.
    int max_fps = 0;                    ///< Cap processed FPS; @c 0 means unlimited.
};

/// @brief Snapshot of session identity, health, and counters for observability APIs.
struct SessionStatus {
    std::string session_id;
    std::string session_name;
    std::string source_uri;
    std::string state;              ///< @c starting, @c running, @c stopped, @c error, or @c finished.
    uint64_t frames_processed = 0;
    uint64_t frames_dropped = 0;
    double uptime_seconds = 0.0;
    std::string error_message;
};

/// @brief One camera or file decoded and analyzed on a worker thread using shared inference objects.
///
/// Owns decode path buffers, preprocessors, and result writers while borrowing @ref SharedResources
/// for Triton-backed detectors and matchers.
class PipelineSession {
public:
    PipelineSession(const std::string& session_id,
                    const std::string& session_name,
                    const std::string& source_uri,
                    const std::string& callback_url,
                    const SessionConfig& config,
                    SharedResources& shared);
    ~PipelineSession();

    /// @brief Opens sources, GPU or CPU pools, webhook writer, and starts @ref runLoop on a thread.
    /// @return false if initialization failed; session moves to @c error state.
    bool start();

    /// @brief Signals the worker to exit, joins it, and releases session-local resources.
    void stop();

    /// @brief Thread-safe snapshot of counters and coarse lifecycle state.
    SessionStatus status() const;

    PipelineSession(const PipelineSession&) = delete;
    PipelineSession& operator=(const PipelineSession&) = delete;

private:
    std::string sessionId_;
    std::string sessionName_;
    std::string sourceUri_;
    std::string callbackUrl_;
    SessionConfig config_;

    std::atomic<bool> running_{false};
    std::string state_;
    mutable std::mutex stateMutex_;
    std::atomic<uint64_t> framesProcessed_{0};
    std::atomic<uint64_t> framesDropped_{0};
    std::chrono::steady_clock::time_point startTime_;
    std::string errorMessage_;

    std::thread workerThread_;

    bool useGpu_ = true;
    std::unique_ptr<FrameSource> frameSource_;
    std::unique_ptr<GPUMemoryPool> memoryPool_;
    std::unique_ptr<Preprocessor> preprocessor_;
    std::unique_ptr<CPUMemoryPool> cpuMemoryPool_;
    std::unique_ptr<CPUPreprocessor> cpuPreprocessor_;
    std::unique_ptr<CPUFaceMatcher> cpuFaceMatcher_;
    std::unique_ptr<ResultHandler> resultHandler_;

    SharedResources& shared_;

    std::string shmYoloIn_;
    std::string shmYoloOut_;
    std::string shmArcIn_;
    std::string shmArcOut_;

    void runLoop();
    void setState(const std::string& state);
    bool initFrameSource();
    bool initGpuResources();
    bool initCpuResources();
    bool initResultHandler();
    void processFrameGpu(Frame& frame);
    void processFrameCpu(Frame& frame);
    FrameResult makeResult(uint64_t frame_id) const;
    std::string utcTimestamp() const;
    float effectiveConfidenceThreshold() const;
    std::vector<Detection> decodeRawDetections(
        const std::vector<RawDetection>& raw, float scale,
        int pad_x, int pad_y, int orig_width, int orig_height) const;
    void cleanupResources();
};
