#pragma once

#include <string>
#include <vector>
#include <cstdint>

struct Detection;
struct MatchResult;

/// @brief Per-frame result bundle passed to all writers.
struct FrameResult {
    uint64_t frame_id = 0;
    std::string timestamp_utc;
    std::string source_id;
    std::vector<Detection> detections;
    std::vector<MatchResult> matches;
};

/// @brief Abstract interface for consuming pipeline results.
///
/// Implement this to add new output formats (CSV, JSON, database, websocket, etc.)
/// without touching the pipeline or result handler. Register instances with
/// ResultHandler::addWriter().
class ResultWriter {
public:
    virtual ~ResultWriter() = default;

    /// @brief Called once before the first write to perform any setup.
    /// @return true if initialization succeeded.
    virtual bool open() { return true; }

    /// @brief Write a single frame's results.
    virtual void write(const FrameResult& result) = 0;

    /// @brief Flush any buffered data and release resources.
    virtual void close() {}
};
