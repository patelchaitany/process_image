#pragma once

#include "frame_metrics.h"
#include "ring_buffer.h"
#include <string>
#include <thread>
#include <atomic>
#include <fstream>

class MetricsLogger {
public:
    MetricsLogger() = default;
    ~MetricsLogger();

    bool init(const std::string& output_dir, int flush_interval_ms = 1000,
              int rotate_size_mb = 100);
    void shutdown();

    void log_frame(const FrameMetrics& metrics);

private:
    void writer_thread_func();
    void rotate_file_if_needed();
    std::string generate_filename() const;

    RingBuffer<FrameMetrics, 128> buffer_;
    std::thread writer_thread_;
    std::atomic<bool> running_{false};
    std::string output_dir_;
    std::ofstream file_;
    int flush_interval_ms_ = 1000;
    size_t rotate_size_bytes_ = 100 * 1024 * 1024;
    size_t current_size_ = 0;
    bool header_written_ = false;
};
