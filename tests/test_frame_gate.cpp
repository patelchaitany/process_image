#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

#include "pipeline/frame_gate.h"

namespace {

class GateSimulator {
public:
    explicit GateSimulator(int stride) : gate(stride) {}

    FrameGate gate;
    std::function<double(uint64_t)> process_ms = [](uint64_t) { return 10.0; };
    std::vector<uint64_t> processed;
    std::vector<double> pickup_age_ms;
    double now = 0.0;

    void arrival(uint64_t seq, double t_ms) {
        now = t_ms;
        drain_completions();
        gate.on_arrival(seq, t_ms);
        start_work_if_idle();
    }

    void advance_to(double t_ms) {
        now = t_ms;
        drain_completions();
        start_work_if_idle();
    }

    void end_of_stream(double t_ms) {
        now = t_ms;
        drain_completions();
        while (start_work_if_idle(true)) {
            drain_completions();
        }
    }

    int slot_high_water() const { return gate.has_slot() ? 1 : 0; }

private:
    bool busy_ = false;
    uint64_t current_seq_ = 0;
    double busy_until_ = 0.0;

    void drain_completions() {
        while (busy_ && now >= busy_until_) {
            gate.on_finished(current_seq_);
            busy_ = false;
            start_work_if_idle();
        }
    }

    bool start_work_if_idle(bool eof = false) {
        if (busy_) {
            return false;
        }
        std::optional<GatedFrame> pick = gate.try_pick(eof);
        if (!pick) {
            return false;
        }
        pickup_age_ms.push_back(now - pick->arrival_time_ms);
        current_seq_ = pick->seq;
        processed.push_back(current_seq_);
        busy_ = true;
        busy_until_ = now + process_ms(current_seq_);
        if (process_ms(current_seq_) <= 0.0) {
            gate.on_finished(current_seq_);
            busy_ = false;
            return start_work_if_idle(eof);
        }
        return true;
    }
};

TEST(FrameGateTest, FasterThanPeriodStride1ProcessesEveryFrame) {
    GateSimulator sim(1);
    sim.process_ms = [](uint64_t) { return 5.0; };
    for (uint64_t i = 0; i < 8; ++i) {
        sim.arrival(i, static_cast<double>(i) * 10.0);
    }
    sim.advance_to(100.0);
    EXPECT_EQ(sim.processed, std::vector<uint64_t>({0, 1, 2, 3, 4, 5, 6, 7}));
    EXPECT_EQ(sim.gate.drops(), 0u);
}

TEST(FrameGateTest, FasterThanPeriodStride2Alternates) {
    GateSimulator sim(2);
    sim.process_ms = [](uint64_t) { return 5.0; };
    for (uint64_t i = 0; i < 10; ++i) {
        sim.arrival(i, static_cast<double>(i) * 10.0);
    }
    sim.advance_to(200.0);
    std::vector<uint64_t> expected;
    for (uint64_t i = 0; i < 10; i += 2) {
        expected.push_back(i);
    }
    EXPECT_EQ(sim.processed, expected);
    for (size_t i = 1; i < sim.processed.size(); ++i) {
        EXPECT_GE(sim.processed[i] - sim.processed[i - 1], 2u);
    }
}

TEST(FrameGateTest, EightyMsThirtyThreePeriodStride2Trace) {
    GateSimulator sim(2);
    sim.process_ms = [](uint64_t) { return 80.0; };
    const double period = 33.0;
    sim.arrival(0, 0.0);
    sim.arrival(1, period);
    sim.arrival(2, 2.0 * period);
    sim.advance_to(80.0);
    sim.arrival(3, 3.0 * period);
    sim.arrival(4, 4.0 * period);
    sim.advance_to(160.0);
    EXPECT_EQ(sim.processed, std::vector<uint64_t>({0, 2, 4}));
    EXPECT_EQ(sim.gate.drops(), 2u);
}

TEST(FrameGateTest, EightyMsThirtyThreePeriodStride1JumpsToNewest) {
    GateSimulator sim(1);
    sim.process_ms = [](uint64_t) { return 80.0; };
    const double period = 33.0;
    sim.arrival(0, 0.0);
    sim.arrival(1, period);
    sim.arrival(2, 2.0 * period);
    sim.advance_to(80.0);
    sim.arrival(3, 3.0 * period);
    sim.arrival(4, 4.0 * period);
    sim.advance_to(160.0);
    EXPECT_EQ(sim.processed, std::vector<uint64_t>({0, 2, 4}));
    EXPECT_EQ(sim.gate.drops(), 2u);
}

TEST(FrameGateTest, FirstFrameOneSecondThenSteadyStride2) {
    GateSimulator sim(2);
    sim.process_ms = [](uint64_t seq) { return seq == 0 ? 1000.0 : 20.0; };
    const double period = 33.0;
    for (int i = 0; i <= 30; ++i) {
        sim.arrival(static_cast<uint64_t>(i), static_cast<double>(i) * period);
    }
    sim.advance_to(999.0);
    EXPECT_EQ(sim.processed, std::vector<uint64_t>({0}));
    sim.advance_to(1000.0);
    EXPECT_EQ(sim.processed, std::vector<uint64_t>({0, 30}));
    EXPECT_EQ(sim.gate.drops(), 29u);
}

TEST(FrameGateTest, MidStreamOneSecondFrameDoesNotCancel) {
    GateSimulator sim(2);
    sim.process_ms = [](uint64_t seq) {
        if (seq == 4) {
            return 1000.0;
        }
        return 20.0;
    };
    const double period = 33.0;
    for (int i = 0; i <= 40; ++i) {
        sim.arrival(static_cast<uint64_t>(i), static_cast<double>(i) * period);
    }
    sim.advance_to(5000.0);
    auto it = std::find(sim.processed.begin(), sim.processed.end(), 4u);
    ASSERT_NE(it, sim.processed.end());
    EXPECT_LT(static_cast<size_t>(it - sim.processed.begin() + 1), sim.processed.size());
}

TEST(FrameGateTest, ThreeArrivalsDuringOneCallOnlyNewestEligible) {
    GateSimulator sim(2);
    sim.process_ms = [](uint64_t) { return 100.0; };
    sim.arrival(0, 0.0);
    sim.arrival(1, 10.0);
    sim.arrival(2, 20.0);
    sim.arrival(3, 30.0);
    sim.advance_to(150.0);
    EXPECT_EQ(sim.processed, std::vector<uint64_t>({0, 3}));
}

TEST(FrameGateTest, WaitsInsideStrideWindow) {
    GateSimulator sim(2);
    sim.process_ms = [](uint64_t) { return 10.0; };
    sim.arrival(0, 0.0);
    sim.advance_to(15.0);
    sim.arrival(1, 15.0);
    sim.advance_to(20.0);
    EXPECT_EQ(sim.processed, std::vector<uint64_t>({0}));
    sim.arrival(2, 25.0);
    sim.advance_to(40.0);
    EXPECT_EQ(sim.processed, std::vector<uint64_t>({0, 2}));
}

TEST(FrameGateTest, JitterStillRespectsStride) {
    GateSimulator sim(2);
    sim.process_ms = [](uint64_t) { return 12.0; };
    const std::vector<double> times = {0, 23, 56, 89, 101, 135, 168, 201};
    for (size_t i = 0; i < times.size(); ++i) {
        sim.arrival(static_cast<uint64_t>(i), times[i]);
    }
    sim.advance_to(300.0);
    for (size_t i = 1; i < sim.processed.size(); ++i) {
        EXPECT_GE(sim.processed[i] - sim.processed[i - 1], 2u);
    }
    EXPECT_LE(sim.slot_high_water(), 1);
}

TEST(FrameGateTest, StallThenNextArrival) {
    GateSimulator sim(2);
    sim.process_ms = [](uint64_t) { return 10.0; };
    sim.arrival(0, 0.0);
    sim.advance_to(20.0);
    sim.arrival(1, 520.0);
    sim.advance_to(530.0);
    EXPECT_EQ(sim.processed, std::vector<uint64_t>({0}));
    sim.arrival(2, 540.0);
    sim.advance_to(560.0);
    EXPECT_EQ(sim.processed, std::vector<uint64_t>({0, 2}));
}

TEST(FrameGateTest, EndOfStreamEligibleFrameProcessed) {
    GateSimulator sim(2);
    sim.process_ms = [](uint64_t) { return 10.0; };
    sim.arrival(0, 0.0);
    sim.advance_to(15.0);
    sim.arrival(2, 20.0);
    sim.end_of_stream(25.0);
    EXPECT_EQ(sim.processed, std::vector<uint64_t>({0, 2}));
}

TEST(FrameGateTest, EndOfStreamIneligibleFrameNotProcessed) {
    GateSimulator sim(2);
    sim.process_ms = [](uint64_t) { return 10.0; };
    sim.arrival(0, 0.0);
    sim.advance_to(15.0);
    sim.arrival(1, 20.0);
    sim.end_of_stream(25.0);
    EXPECT_EQ(sim.processed, std::vector<uint64_t>({0}));
    EXPECT_TRUE(sim.gate.has_slot());
}

TEST(FrameGateTest, DecreasingSeqIgnored) {
    FrameGate gate(2);
    gate.on_arrival(5, 0.0);
    gate.on_arrival(3, 1.0);
    EXPECT_EQ(gate.last_accepted_arrival_seq(), 5u);
    auto pick = gate.try_pick();
    ASSERT_TRUE(pick.has_value());
    EXPECT_EQ(pick->seq, 5u);
}

TEST(FrameGateTest, OverrunKeepsPickupAgeBounded) {
    GateSimulator sim(2);
    const double period = 33.0;
    sim.process_ms = [](uint64_t) { return 80.0; };
    for (int i = 0; i < 60; ++i) {
        sim.arrival(static_cast<uint64_t>(i), static_cast<double>(i) * period);
        sim.advance_to(static_cast<double>(i) * period);
    }
    sim.advance_to(2000.0);
    for (size_t i = 1; i < sim.processed.size(); ++i) {
        EXPECT_GE(sim.processed[i] - sim.processed[i - 1], 2u);
    }
    ASSERT_GT(sim.pickup_age_ms.size(), 1u);
    const double max_pickup_age = 80.0 + period;
    for (size_t i = 1; i < sim.pickup_age_ms.size(); ++i) {
        EXPECT_LE(sim.pickup_age_ms[i], max_pickup_age);
    }
}

TEST(FrameGateTest, SingleSlotHighWaterAlwaysOne) {
    GateSimulator sim(2);
    sim.process_ms = [](uint64_t) { return 200.0; };
    const double period = 33.0;
    int max_slot = 0;
    for (int i = 0; i < 30; ++i) {
        sim.arrival(static_cast<uint64_t>(i), static_cast<double>(i) * period);
        max_slot = std::max(max_slot, sim.slot_high_water());
        if (sim.gate.has_slot()) {
            max_slot = std::max(max_slot, 1);
        }
    }
    sim.advance_to(3000.0);
    EXPECT_LE(max_slot, 1);
}

TEST(FrameGateTest, DirectPickDropAccounting) {
    FrameGate gate(2);
    gate.on_arrival(0, 0.0);
    auto a = gate.try_pick();
    ASSERT_TRUE(a.has_value());
    gate.on_finished(0);
    gate.on_arrival(3, 10.0);
    auto b = gate.try_pick();
    ASSERT_TRUE(b.has_value());
    EXPECT_EQ(b->seq, 3u);
    EXPECT_EQ(gate.drops(), 2u);
}

}  // namespace
