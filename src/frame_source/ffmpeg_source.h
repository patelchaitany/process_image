#pragma once

#include "frame_source.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

class FFmpegSource : public FrameSource {
public:
    FFmpegSource() = default;
    ~FFmpegSource() override;

    bool open(const std::string& source) override;
    bool read(Frame& frame) override;
    void close() override;
    int width() const override;
    int height() const override;
    double fps() const override;
    bool is_open() const override;

private:
    AVFormatContext* fmt_ctx_ = nullptr;
    AVCodecContext* codec_ctx_ = nullptr;
    SwsContext* sws_ctx_ = nullptr;
    AVFrame* av_frame_ = nullptr;
    AVFrame* bgr_frame_ = nullptr;
    AVPacket* pkt_ = nullptr;
    int video_stream_idx_ = -1;
    uint64_t frame_count_ = 0;
    bool opened_ = false;

    bool decode_next_frame();
};
