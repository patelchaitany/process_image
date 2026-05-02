#include <gtest/gtest.h>
#include "gpu/cpu_preprocessor.h"
#include <vector>
#include <cmath>

TEST(CPUPreprocessorTest, YoloOutputShape) {
    CPUPreprocessor pp;
    int src_w = 640, src_h = 480;
    int dst_size = 640;
    std::vector<uint8_t> src(src_w * src_h * 3, 128);
    std::vector<float> dst(3 * dst_size * dst_size, 0.0f);

    ASSERT_TRUE(pp.preprocess_yolo(src.data(), dst.data(), src_w, src_h, dst_size));

    auto params = pp.last_params();
    EXPECT_EQ(params.dst_size, dst_size);
    EXPECT_GT(params.scale, 0.0f);
}

TEST(CPUPreprocessorTest, YoloLetterboxParams) {
    CPUPreprocessor pp;
    int src_w = 1920, src_h = 1080;
    int dst_size = 640;
    std::vector<uint8_t> src(src_w * src_h * 3, 0);
    std::vector<float> dst(3 * dst_size * dst_size, 0.0f);

    pp.preprocess_yolo(src.data(), dst.data(), src_w, src_h, dst_size);

    auto params = pp.last_params();
    float expected_scale = std::min(640.0f / 1920, 640.0f / 1080);
    EXPECT_NEAR(params.scale, expected_scale, 1e-5f);
    EXPECT_GE(params.pad_x, 0);
    EXPECT_GE(params.pad_y, 0);
}

TEST(CPUPreprocessorTest, YoloNormalizedRange) {
    CPUPreprocessor pp;
    int src_w = 64, src_h = 64;
    int dst_size = 64;
    std::vector<uint8_t> src(src_w * src_h * 3, 255);
    std::vector<float> dst(3 * dst_size * dst_size, -1.0f);

    pp.preprocess_yolo(src.data(), dst.data(), src_w, src_h, dst_size);

    for (float v : dst) {
        EXPECT_GE(v, 0.0f);
        EXPECT_LE(v, 1.0f);
    }
}

TEST(CPUPreprocessorTest, ArcfaceOutputShape) {
    CPUPreprocessor pp;
    int src_w = 640, src_h = 480;
    int num_faces = 2;
    std::vector<uint8_t> src(src_w * src_h * 3, 100);
    std::vector<float> dst(num_faces * 3 * 112 * 112, 0.0f);
    float bboxes[] = {10.0f, 10.0f, 100.0f, 100.0f,
                      200.0f, 50.0f, 350.0f, 200.0f};

    ASSERT_TRUE(pp.preprocess_arcface(src.data(), dst.data(),
                                       bboxes, num_faces, src_w, src_h));
}
