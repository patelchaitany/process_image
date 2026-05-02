#include "ffmpeg_source.h"
#include <cstdio>

FFmpegSource::~FFmpegSource() {
    close();
}

bool FFmpegSource::open(const std::string& source) {
    close();

    if (avformat_open_input(&fmt_ctx_, source.c_str(), nullptr, nullptr) < 0) {
        fprintf(stderr, "Failed to open input: %s\n", source.c_str());
        return false;
    }

    if (avformat_find_stream_info(fmt_ctx_, nullptr) < 0) {
        fprintf(stderr, "Failed to find stream info\n");
        close();
        return false;
    }

    video_stream_idx_ = -1;
    for (unsigned i = 0; i < fmt_ctx_->nb_streams; ++i) {
        if (fmt_ctx_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_idx_ = static_cast<int>(i);
            break;
        }
    }

    if (video_stream_idx_ < 0) {
        fprintf(stderr, "No video stream found\n");
        close();
        return false;
    }

    const AVCodec* codec = avcodec_find_decoder(
        fmt_ctx_->streams[video_stream_idx_]->codecpar->codec_id);
    if (!codec) {
        fprintf(stderr, "Unsupported codec\n");
        close();
        return false;
    }

    codec_ctx_ = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(codec_ctx_,
        fmt_ctx_->streams[video_stream_idx_]->codecpar);

    if (avcodec_open2(codec_ctx_, codec, nullptr) < 0) {
        fprintf(stderr, "Failed to open codec\n");
        close();
        return false;
    }

    sws_ctx_ = sws_getContext(
        codec_ctx_->width, codec_ctx_->height, codec_ctx_->pix_fmt,
        codec_ctx_->width, codec_ctx_->height, AV_PIX_FMT_BGR24,
        SWS_BILINEAR, nullptr, nullptr, nullptr);

    if (!sws_ctx_) {
        fprintf(stderr, "Failed to create SwsContext\n");
        close();
        return false;
    }

    av_frame_ = av_frame_alloc();
    bgr_frame_ = av_frame_alloc();
    pkt_ = av_packet_alloc();

    int num_bytes = av_image_get_buffer_size(AV_PIX_FMT_BGR24,
        codec_ctx_->width, codec_ctx_->height, 1);
    uint8_t* buffer = (uint8_t*)av_malloc(num_bytes);
    av_image_fill_arrays(bgr_frame_->data, bgr_frame_->linesize,
        buffer, AV_PIX_FMT_BGR24,
        codec_ctx_->width, codec_ctx_->height, 1);

    frame_count_ = 0;
    opened_ = true;
    return true;
}

bool FFmpegSource::read(Frame& frame) {
    if (!opened_) return false;

    if (!decode_next_frame()) return false;

    sws_scale(sws_ctx_,
        av_frame_->data, av_frame_->linesize, 0, codec_ctx_->height,
        bgr_frame_->data, bgr_frame_->linesize);

    int w = codec_ctx_->width;
    int h = codec_ctx_->height;
    size_t row_bytes = static_cast<size_t>(w) * 3;

    frame.width = w;
    frame.height = h;
    frame.channels = 3;
    frame.frame_index = frame_count_++;
    frame.data.resize(static_cast<size_t>(w) * h * 3);

    for (int y = 0; y < h; ++y) {
        memcpy(frame.data.data() + y * row_bytes,
               bgr_frame_->data[0] + y * bgr_frame_->linesize[0],
               row_bytes);
    }

    return true;
}

bool FFmpegSource::decode_next_frame() {
    while (true) {
        int ret = av_read_frame(fmt_ctx_, pkt_);
        if (ret < 0) return false;  // EOF or error

        if (pkt_->stream_index != video_stream_idx_) {
            av_packet_unref(pkt_);
            continue;
        }

        ret = avcodec_send_packet(codec_ctx_, pkt_);
        av_packet_unref(pkt_);
        if (ret < 0) return false;

        ret = avcodec_receive_frame(codec_ctx_, av_frame_);
        if (ret == AVERROR(EAGAIN)) continue;
        if (ret < 0) return false;

        return true;
    }
}

void FFmpegSource::close() {
    if (bgr_frame_) {
        if (bgr_frame_->data[0]) av_free(bgr_frame_->data[0]);
        av_frame_free(&bgr_frame_);
    }
    if (av_frame_) av_frame_free(&av_frame_);
    if (pkt_) av_packet_free(&pkt_);
    if (sws_ctx_) { sws_freeContext(sws_ctx_); sws_ctx_ = nullptr; }
    if (codec_ctx_) { avcodec_free_context(&codec_ctx_); }
    if (fmt_ctx_) { avformat_close_input(&fmt_ctx_); }
    video_stream_idx_ = -1;
    opened_ = false;
}

int FFmpegSource::width() const {
    return codec_ctx_ ? codec_ctx_->width : 0;
}

int FFmpegSource::height() const {
    return codec_ctx_ ? codec_ctx_->height : 0;
}

double FFmpegSource::fps() const {
    if (!fmt_ctx_ || video_stream_idx_ < 0) return 0.0;
    AVRational r = fmt_ctx_->streams[video_stream_idx_]->avg_frame_rate;
    return (r.den > 0) ? static_cast<double>(r.num) / r.den : 0.0;
}

bool FFmpegSource::is_open() const {
    return opened_;
}
