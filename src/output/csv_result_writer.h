#pragma once

#include "output/result_writer.h"
#include "metrics/ring_buffer.h"
#include <fstream>
#include <string>
#include <thread>
#include <atomic>

/// @brief Non-blocking CSV writer for per-detection bounding-box data.
///
/// write() pushes into a lock-free ring buffer and returns immediately.
/// A background thread drains the buffer and writes to disk, so pipeline
/// latency is never affected by file I/O.
///
/// One CSV row per detection per frame:
///   frame_id, timestamp_utc, source_id, det_idx,
///   x1, y1, x2, y2, bbox_width, bbox_height, det_confidence,
///   identity, face_id, match_confidence
///
/// File is rotated when it exceeds rotateSizeMb.
class CsvResultWriter : public ResultWriter {
public:
    /// @param path            Output CSV file path.
    /// @param rotateSizeMb    Rotate the file once it exceeds this size (0 = never).
    /// @param flushIntervalMs How often the writer thread flushes to disk.
    explicit CsvResultWriter(const std::string& path,
                             int rotateSizeMb = 100,
                             int flushIntervalMs = 500);
    ~CsvResultWriter() override;

    bool open() override;
    void write(const FrameResult& result) override;
    void close() override;

private:
    std::string basePath_;
    int rotateSizeBytes_;
    int flushIntervalMs_;

    std::ofstream file_;
    bool headerWritten_ = false;

    RingBuffer<FrameResult, 4096> buffer_;
    std::atomic<bool> running_{false};
    std::thread writerThread_;

    void writerLoop();
    void writeEntry(const FrameResult& result);
    void drain();
    void writeHeader();
    void rotateIfNeeded();
    std::string generateRotatedPath() const;
};
