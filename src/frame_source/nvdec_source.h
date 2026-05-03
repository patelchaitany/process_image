#pragma once

#include "frame_source.h"
#include <cuda_runtime.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
}

/// @brief FrameSource using NVIDIA NVDEC hardware decoder.
///
/// Decodes video entirely on the GPU using the dedicated NVDEC ASIC.
/// Decoded NV12 frames are converted to BGR24 on-GPU via a CUDA kernel,
/// so the frame data never touches CPU memory.
class NvdecSource : public FrameSource {
public:
    NvdecSource();
    ~NvdecSource() override;

    NvdecSource(const NvdecSource&) = delete;
    NvdecSource& operator=(const NvdecSource&) = delete;

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
    AVBufferRef* hw_device_ctx_ = nullptr;
    AVFrame* hw_frame_ = nullptr;
    AVPacket* pkt_ = nullptr;
    int video_stream_idx_ = -1;
    uint64_t frame_count_ = 0;
    bool opened_ = false;

    uint8_t* gpu_bgr_ = nullptr;
    size_t gpu_bgr_size_ = 0;
    cudaStream_t stream_ = nullptr;

    bool decode_next_frame();

    /// @brief CUDA kernel launcher: convert NV12 planes to BGR24 on GPU.
    void launch_nv12_to_bgr(const uint8_t* y_plane, const uint8_t* uv_plane,
                             int y_pitch, int uv_pitch, int w, int h);

    static enum AVPixelFormat get_hw_format(AVCodecContext* ctx,
                                             const enum AVPixelFormat* pix_fmts);
};
