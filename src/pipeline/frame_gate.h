#pragma once

#include <cstdint>
#include <optional>

/// @brief Single-slot realtime frame gate for live ingest (RTSP).
///
/// Ingest never blocks; at most one frame is retained and newer arrivals overwrite.
/// The worker picks the newest eligible frame when idle: first pick ignores stride;
/// later picks require slot.seq >= last_processed_seq + stride.
struct GatedFrame {
    uint64_t seq = 0;
    double arrival_time_ms = 0.0;
};

class FrameGate {
public:
    /// @param stride Minimum gap between processed sequence numbers (>= 1). Default 2 = skip alternate.
    explicit FrameGate(int stride = 2);

    /// @brief Record a new frame from the decoder. Out-of-order or duplicate seq is ignored.
    void on_arrival(uint64_t seq, double arrival_time_ms);

    /// @brief Attempt to take a frame for processing when the worker is idle.
    /// @param end_of_stream If true, pick a held frame only when stride rules allow (same as live).
    std::optional<GatedFrame> try_pick(bool end_of_stream = false);

    /// @brief Call after inference completes for the picked seq.
    void on_finished(uint64_t seq);

    int stride() const { return stride_; }
    bool has_slot() const { return slot_.has_value(); }
    uint64_t frames_completed() const { return frames_completed_; }
    uint64_t last_processed_seq() const { return last_processed_seq_; }
    uint64_t drops() const { return drops_; }
    uint64_t last_accepted_arrival_seq() const { return last_accepted_arrival_seq_; }

private:
    int stride_;
    std::optional<GatedFrame> slot_;
    uint64_t last_processed_seq_ = 0;
    bool has_processed_ = false;
    uint64_t frames_completed_ = 0;
    uint64_t drops_ = 0;
    uint64_t last_accepted_arrival_seq_ = 0;
    bool any_arrival_ = false;
};

inline FrameGate::FrameGate(int stride) : stride_(stride < 1 ? 1 : stride) {}

inline void FrameGate::on_arrival(uint64_t seq, double arrival_time_ms) {
    if (any_arrival_ && seq <= last_accepted_arrival_seq_) {
        return;
    }
    any_arrival_ = true;
    last_accepted_arrival_seq_ = seq;
    slot_ = GatedFrame{seq, arrival_time_ms};
}

inline std::optional<GatedFrame> FrameGate::try_pick(bool end_of_stream) {
    if (!slot_) {
        return std::nullopt;
    }

    if (!has_processed_) {
        GatedFrame picked = *slot_;
        slot_ = std::nullopt;
        return picked;
    }

    const uint64_t latest = slot_->seq;
    const uint64_t min_seq = last_processed_seq_ + static_cast<uint64_t>(stride_);
    if (latest < min_seq) {
        if (end_of_stream) {
            return std::nullopt;
        }
        return std::nullopt;
    }

    drops_ += latest - last_processed_seq_ - 1;
    GatedFrame picked = *slot_;
    slot_ = std::nullopt;
    return picked;
}

inline void FrameGate::on_finished(uint64_t seq) {
    last_processed_seq_ = seq;
    has_processed_ = true;
    frames_completed_++;
}
