#include "nvdec_source.h"
#include <cstdio>

extern "C" {
#include <libavutil/hwcontext_cuda.h>
}

// ---- NV12 → BGR24 CUDA kernel ----

__global__ void nv12ToBgrKernel(const uint8_t* __restrict__ y_plane,
                                 const uint8_t* __restrict__ uv_plane,
                                 uint8_t* __restrict__ bgr,
                                 int width, int height,
                                 int y_pitch, int uv_pitch) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float Y = static_cast<float>(y_plane[y * y_pitch + x]);
    int uv_idx = (y / 2) * uv_pitch + (x / 2) * 2;
    float U = static_cast<float>(uv_plane[uv_idx])     - 128.0f;
    float V = static_cast<float>(uv_plane[uv_idx + 1]) - 128.0f;

    // BT.601 YUV → RGB
    float R = Y + 1.402f * V;
    float G = Y - 0.344136f * U - 0.714136f * V;
    float B = Y + 1.772f * U;

    int bgr_idx = (y * width + x) * 3;
    bgr[bgr_idx + 0] = static_cast<uint8_t>(fminf(fmaxf(B, 0.0f), 255.0f));
    bgr[bgr_idx + 1] = static_cast<uint8_t>(fminf(fmaxf(G, 0.0f), 255.0f));
    bgr[bgr_idx + 2] = static_cast<uint8_t>(fminf(fmaxf(R, 0.0f), 255.0f));
}

// ---- NvdecSource implementation ----

NvdecSource::NvdecSource() = default;

NvdecSource::~NvdecSource() {
    close();
}

enum AVPixelFormat NvdecSource::get_hw_format(AVCodecContext* /*ctx*/,
                                               const enum AVPixelFormat* pix_fmts) {
    for (const auto* p = pix_fmts; *p != AV_PIX_FMT_NONE; p++) {
        if (*p == AV_PIX_FMT_CUDA) return *p;
    }
    fprintf(stderr, "NvdecSource: CUDA pixel format not offered by decoder\n");
    return AV_PIX_FMT_NONE;
}

bool NvdecSource::open(const std::string& source) {
    close();

    if (avformat_open_input(&fmt_ctx_, source.c_str(), nullptr, nullptr) < 0) {
        fprintf(stderr, "NvdecSource: failed to open %s\n", source.c_str());
        return false;
    }
    if (avformat_find_stream_info(fmt_ctx_, nullptr) < 0) {
        fprintf(stderr, "NvdecSource: no stream info\n");
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
        fprintf(stderr, "NvdecSource: no video stream\n");
        close();
        return false;
    }

    const AVCodec* codec = avcodec_find_decoder(
        fmt_ctx_->streams[video_stream_idx_]->codecpar->codec_id);
    if (!codec) {
        fprintf(stderr, "NvdecSource: unsupported codec\n");
        close();
        return false;
    }

    // Check that the codec supports CUDA hwaccel
    bool cuda_supported = false;
    for (int i = 0;; i++) {
        const AVCodecHWConfig* cfg = avcodec_get_hw_config(codec, i);
        if (!cfg) break;
        if (cfg->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX &&
            cfg->device_type == AV_HWDEVICE_TYPE_CUDA) {
            cuda_supported = true;
            break;
        }
    }
    if (!cuda_supported) {
        fprintf(stderr, "NvdecSource: codec %s does not support CUDA hwaccel\n", codec->name);
        close();
        return false;
    }

    codec_ctx_ = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(codec_ctx_,
        fmt_ctx_->streams[video_stream_idx_]->codecpar);

    // Create CUDA hardware device context
    if (av_hwdevice_ctx_create(&hw_device_ctx_, AV_HWDEVICE_TYPE_CUDA,
                                nullptr, nullptr, 0) < 0) {
        fprintf(stderr, "NvdecSource: failed to create CUDA hw device context\n");
        close();
        return false;
    }
    codec_ctx_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
    codec_ctx_->get_format = get_hw_format;

    if (avcodec_open2(codec_ctx_, codec, nullptr) < 0) {
        fprintf(stderr, "NvdecSource: failed to open codec with NVDEC\n");
        close();
        return false;
    }

    hw_frame_ = av_frame_alloc();
    pkt_ = av_packet_alloc();

    int w = codec_ctx_->width;
    int h = codec_ctx_->height;
    gpu_bgr_size_ = static_cast<size_t>(w) * h * 3;

    cudaError_t err = cudaMalloc(&gpu_bgr_, gpu_bgr_size_);
    if (err != cudaSuccess) {
        fprintf(stderr, "NvdecSource: cudaMalloc BGR buffer failed: %s\n",
                cudaGetErrorString(err));
        close();
        return false;
    }

    err = cudaStreamCreate(&stream_);
    if (err != cudaSuccess) {
        fprintf(stderr, "NvdecSource: cudaStreamCreate failed: %s\n",
                cudaGetErrorString(err));
        close();
        return false;
    }

    frame_count_ = 0;
    opened_ = true;

    fprintf(stderr, "NvdecSource: NVDEC decoder opened – %s, %dx%d (zero CPU threads)\n",
            codec->name, w, h);
    return true;
}

bool NvdecSource::read(Frame& frame) {
    if (!opened_) return false;
    if (!decode_next_frame()) return false;

    // hw_frame_->format == AV_PIX_FMT_CUDA
    // data[0] = Y plane (device ptr), data[1] = UV plane (device ptr)
    const auto* y_ptr  = reinterpret_cast<const uint8_t*>(hw_frame_->data[0]);
    const auto* uv_ptr = reinterpret_cast<const uint8_t*>(hw_frame_->data[1]);
    int y_pitch  = hw_frame_->linesize[0];
    int uv_pitch = hw_frame_->linesize[1];
    int w = hw_frame_->width;
    int h = hw_frame_->height;

    launch_nv12_to_bgr(y_ptr, uv_ptr, y_pitch, uv_pitch, w, h);

    cudaStreamSynchronize(stream_);
    av_frame_unref(hw_frame_);

    frame.gpu_data = gpu_bgr_;
    frame.on_gpu = true;
    frame.data.clear();
    frame.width = w;
    frame.height = h;
    frame.channels = 3;
    frame.frame_index = frame_count_++;

    return true;
}

void NvdecSource::launch_nv12_to_bgr(const uint8_t* y_plane, const uint8_t* uv_plane,
                                       int y_pitch, int uv_pitch, int w, int h) {
    dim3 block(32, 8);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    nv12ToBgrKernel<<<grid, block, 0, stream_>>>(
        y_plane, uv_plane, gpu_bgr_, w, h, y_pitch, uv_pitch);
}

bool NvdecSource::decode_next_frame() {
    while (true) {
        int ret = av_read_frame(fmt_ctx_, pkt_);
        if (ret < 0) return false;

        if (pkt_->stream_index != video_stream_idx_) {
            av_packet_unref(pkt_);
            continue;
        }

        ret = avcodec_send_packet(codec_ctx_, pkt_);
        av_packet_unref(pkt_);
        if (ret < 0) return false;

        ret = avcodec_receive_frame(codec_ctx_, hw_frame_);
        if (ret == AVERROR(EAGAIN)) continue;
        if (ret < 0) return false;

        return true;
    }
}

void NvdecSource::close() {
    opened_ = false;

    if (gpu_bgr_) { cudaFree(gpu_bgr_); gpu_bgr_ = nullptr; }
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
    if (hw_frame_) { av_frame_free(&hw_frame_); }
    if (pkt_) { av_packet_free(&pkt_); }
    if (codec_ctx_) { avcodec_free_context(&codec_ctx_); }
    if (hw_device_ctx_) { av_buffer_unref(&hw_device_ctx_); hw_device_ctx_ = nullptr; }
    if (fmt_ctx_) { avformat_close_input(&fmt_ctx_); }
    video_stream_idx_ = -1;
    gpu_bgr_size_ = 0;
}

int NvdecSource::width() const {
    return codec_ctx_ ? codec_ctx_->width : 0;
}

int NvdecSource::height() const {
    return codec_ctx_ ? codec_ctx_->height : 0;
}

double NvdecSource::fps() const {
    if (!fmt_ctx_ || video_stream_idx_ < 0) return 0.0;
    AVRational r = fmt_ctx_->streams[video_stream_idx_]->avg_frame_rate;
    return (r.den > 0) ? static_cast<double>(r.num) / r.den : 0.0;
}

bool NvdecSource::is_open() const {
    return opened_;
}
