#pragma once

#include "frame_source.h"
#include <memory>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

/// @brief Wraps any FrameSource and decodes frames on a background thread.
///
/// The pipeline thread calls read() which pops a pre-decoded frame from
/// the buffer in ~0ms. A separate decode thread continuously fills the
/// buffer up to `bufferSize` frames ahead. This hides the entire decode
/// latency (20-30ms) behind the pipeline's GPU/inference work.
class PrefetchSource : public FrameSource {
public:
    /// @param inner      The actual FrameSource to wrap (takes ownership).
    /// @param bufferSize Max frames to decode ahead (2-4 is typical).
    explicit PrefetchSource(std::unique_ptr<FrameSource> inner, int bufferSize = 3);
    ~PrefetchSource() override;

    PrefetchSource(const PrefetchSource&) = delete;
    PrefetchSource& operator=(const PrefetchSource&) = delete;

    bool open(const std::string& source) override;

    /// @brief Returns a pre-decoded frame from the buffer.
    /// Blocks only if the decode thread hasn't produced one yet.
    bool read(Frame& frame) override;

    void close() override;
    int width() const override;
    int height() const override;
    double fps() const override;
    bool is_open() const override;

private:
    std::unique_ptr<FrameSource> inner_;
    int bufferSize_;

    std::queue<Frame> queue_;
    std::mutex mu_;
    std::condition_variable notEmpty_;
    std::condition_variable notFull_;

    std::thread decodeThread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> eof_{false};

    void decodeLoop();
};
