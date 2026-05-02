#include <gtest/gtest.h>
#include "frame_source/ffmpeg_source.h"

TEST(FrameSourceTest, OpenInvalidPath) {
    FFmpegSource source;
    EXPECT_FALSE(source.open("nonexistent_file.mp4"));
    EXPECT_FALSE(source.is_open());
}

TEST(FrameSourceTest, CloseWithoutOpen) {
    FFmpegSource source;
    source.close();  // should not crash
    EXPECT_FALSE(source.is_open());
}

TEST(FrameSourceTest, ReadWithoutOpen) {
    FFmpegSource source;
    Frame frame;
    EXPECT_FALSE(source.read(frame));
}

// Integration test - requires a test video file
TEST(FrameSourceTest, DISABLED_OpenAndReadVideo) {
    FFmpegSource source;
    ASSERT_TRUE(source.open("test_data/sample.mp4"));
    EXPECT_TRUE(source.is_open());
    EXPECT_GT(source.width(), 0);
    EXPECT_GT(source.height(), 0);
    EXPECT_GT(source.fps(), 0.0);

    Frame frame;
    ASSERT_TRUE(source.read(frame));
    EXPECT_EQ(frame.width, source.width());
    EXPECT_EQ(frame.height, source.height());
    EXPECT_EQ(frame.channels, 3);
    EXPECT_EQ(frame.frame_index, 0u);
    EXPECT_EQ(frame.data.size(),
              static_cast<size_t>(frame.width) * frame.height * frame.channels);

    source.close();
    EXPECT_FALSE(source.is_open());
}
