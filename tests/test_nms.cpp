#include <gtest/gtest.h>
#include "postprocess/nms.h"

TEST(NMSTest, EmptyInput) {
    std::vector<Detection> empty;
    auto result = apply_nms(empty, 0.45f);
    EXPECT_TRUE(result.empty());
}

TEST(NMSTest, SingleDetection) {
    std::vector<Detection> dets = {{10, 10, 50, 50, 0.9f}};
    auto result = apply_nms(dets, 0.45f);
    ASSERT_EQ(result.size(), 1);
    EXPECT_FLOAT_EQ(result[0].confidence, 0.9f);
}

TEST(NMSTest, NoOverlap) {
    std::vector<Detection> dets = {
        {10, 10, 50, 50, 0.9f},
        {100, 100, 150, 150, 0.8f},
        {200, 200, 250, 250, 0.7f},
    };
    auto result = apply_nms(dets, 0.45f);
    EXPECT_EQ(result.size(), 3);
}

TEST(NMSTest, FullOverlap) {
    std::vector<Detection> dets = {
        {10, 10, 50, 50, 0.9f},
        {10, 10, 50, 50, 0.8f},
        {10, 10, 50, 50, 0.7f},
    };
    auto result = apply_nms(dets, 0.45f);
    ASSERT_EQ(result.size(), 1);
    EXPECT_FLOAT_EQ(result[0].confidence, 0.9f);
}

TEST(NMSTest, PartialOverlap) {
    std::vector<Detection> dets = {
        {10, 10, 50, 50, 0.9f},
        {20, 20, 60, 60, 0.85f},  // overlaps with first
        {100, 100, 140, 140, 0.8f},  // no overlap
    };
    auto result = apply_nms(dets, 0.45f);
    EXPECT_EQ(result.size(), 2);
}

TEST(NMSTest, HigherConfidenceKept) {
    std::vector<Detection> dets = {
        {10, 10, 50, 50, 0.5f},
        {15, 15, 55, 55, 0.95f},  // higher confidence, overlaps
    };
    auto result = apply_nms(dets, 0.3f);
    ASSERT_EQ(result.size(), 1);
    EXPECT_FLOAT_EQ(result[0].confidence, 0.95f);
}

TEST(FilterDecodeTest, ConfidenceFiltering) {
    float raw[] = {
        320, 320, 100, 100, 0.9f,  // above threshold
        160, 160, 50, 50, 0.3f,    // below threshold
        480, 480, 80, 80, 0.7f,    // above threshold
    };
    auto result = filter_and_decode(raw, 3, 0.5f, 640, 1920, 1080, 0.333f, 0, 0);
    EXPECT_EQ(result.size(), 2);
}

TEST(FilterDecodeTest, EmptyOutput) {
    float raw[] = {
        320, 320, 100, 100, 0.1f,
    };
    auto result = filter_and_decode(raw, 1, 0.5f, 640, 1920, 1080, 0.333f, 0, 0);
    EXPECT_TRUE(result.empty());
}
