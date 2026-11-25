#include "decoder.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <thread>

#include "common.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
}
#include <cuda.h>
#include <fcntl.h>
#include <poll.h>
#include <unistd.h>

#include <cstring>

// ------------------------------------------------------------------------------------------------ FrameRing
void FrameRing::alloc(int w, int h, int slots, bool fifo, bool zero_copy) {
    w_ = w; h_ = h; fifo_ = fifo; zero_copy_ = zero_copy;
    own_.assign(slots, nullptr); y_.assign(slots, nullptr); uv_.assign(slots, nullptr);
    pitch_.assign(slots, 0); held_.assign(slots, nullptr);
    refs_.assign(slots, 0); writing_.assign(slots, 0); queued_.assign(slots, 0);
    fid_.assign(slots, -1); t_.assign(slots, 0.0); tc_.assign(slots, 0.0);
    if (zero_copy) return;
    size_t pitch = 0;
    for (int s = 0; s < slots; ++s) {
        void* p = nullptr;
        CK(cudaMallocPitch(&p, &pitch, (size_t)w, (size_t)h * 3 / 2));   // Y rows then UV rows, same pitch
        own_[s] = y_[s] = static_cast<uint8_t*>(p);
        uv_[s] = own_[s] + pitch * h;
    }
    own_pitch_ = (int)pitch;
    for (auto& p : pitch_) p = own_pitch_;
}

void FrameRing::free_all() {
    for (auto& b : own_) if (b) { cudaFree(b); b = nullptr; }
    for (auto& f : held_) if (f) av_frame_free(&f);
}

int FrameRing::begin_write() {
    std::lock_guard<std::mutex> g(m_);
    const int S = (int)refs_.size();
    int pick = -1;
    for (int s = 0; s < S && pick < 0; ++s)
        if (refs_[s] == 0 && !writing_[s] && !queued_[s] && (fifo_ || s != latest_)) pick = s;
    if (pick < 0 && fifo_) {   // buffer full: evict the oldest queued frame (processing is more than buffer_s behind)
        for (int s = 0; s < S; ++s)
            if (queued_[s] && refs_[s] == 0 && (pick < 0 || fid_[s] < fid_[pick])) pick = s;
        if (pick >= 0) { queued_[pick] = 0; overflow++; }
    }
    if (pick < 0) return -1;
    writing_[pick] = 1;
    if (held_[pick]) av_frame_unref(held_[pick]);     // give the old surface back to FFmpeg's pool
    return pick;
}

void FrameRing::end_write(int s, int64_t fid, double t_arr, double t_cap, AVFrame* src) {
    if (zero_copy_) {
        if (!held_[s]) held_[s] = av_frame_alloc();
        av_frame_ref(held_[s], src);
    }
    std::lock_guard<std::mutex> g(m_);
    if (zero_copy_) {
        y_[s] = held_[s]->data[0];
        uv_[s] = held_[s]->data[1];
        pitch_[s] = held_[s]->linesize[0];
    }
    writing_[s] = 0;
    fid_[s] = fid;
    t_[s] = t_arr;
    tc_[s] = t_cap;
    latest_ = s;
    if (fifo_) queued_[s] = 1;
    arrived++;
}

int FrameRing::acquire_latest(int64_t after, int64_t& fid, double& t_arr, double& t_cap) {
    std::lock_guard<std::mutex> g(m_);
    if (latest_ < 0 || fid_[latest_] <= after || writing_[latest_]) return -1;
    refs_[latest_]++;
    fid = fid_[latest_]; t_arr = t_[latest_]; t_cap = tc_[latest_];
    return latest_;
}

int FrameRing::acquire_next(int64_t after, int64_t& fid, double& t_arr, double& t_cap) {
    std::lock_guard<std::mutex> g(m_);
    int best = -1;
    for (int s = 0; s < (int)refs_.size(); ++s)
        if (queued_[s] && fid_[s] > after && (best < 0 || fid_[s] < fid_[best])) best = s;
    if (best < 0) return -1;
    queued_[best] = 0;
    refs_[best]++;
    fid = fid_[best]; t_arr = t_[best]; t_cap = tc_[best];
    return best;
}

int64_t FrameRing::peek(int64_t after, double& t_cap) {
    std::lock_guard<std::mutex> g(m_);
    if (!fifo_) {
        if (latest_ < 0 || writing_[latest_] || fid_[latest_] <= after) return -1;
        t_cap = tc_[latest_];
        return fid_[latest_];
    }
    int best = -1;
    for (int s = 0; s < (int)refs_.size(); ++s)
        if (queued_[s] && fid_[s] > after && (best < 0 || fid_[s] < fid_[best])) best = s;
    if (best < 0) return -1;
    t_cap = tc_[best];
    return fid_[best];
}

int FrameRing::queued() {
    std::lock_guard<std::mutex> g(m_);
    int n = 0;
    for (char q : queued_) n += q;
    return n;
}

void FrameRing::addref(int s) {
    std::lock_guard<std::mutex> g(m_);
    refs_[s]++;
}

void FrameRing::release(int s) {
    std::lock_guard<std::mutex> g(m_);
    refs_[s]--;
}

// ------------------------------------------------------------------------------------------------ VideoDecoder
static enum AVPixelFormat pick_cuda(AVCodecContext*, const enum AVPixelFormat* fmts) {
    for (const enum AVPixelFormat* p = fmts; *p != AV_PIX_FMT_NONE; ++p)
        if (*p == AV_PIX_FMT_CUDA) return *p;
    std::fprintf(stderr, "decoder: CUDA output format not offered\n");
    return AV_PIX_FMT_NONE;
}

VideoDecoder::VideoDecoder(const std::string& path, AVBufferRef* hw) {
    if (avformat_open_input(&fmt_, path.c_str(), nullptr, nullptr) < 0) throw std::runtime_error("open " + path);
    if (avformat_find_stream_info(fmt_, nullptr) < 0) throw std::runtime_error("stream info " + path);
    const AVCodec* codec = nullptr;
    stream_ = av_find_best_stream(fmt_, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
    if (stream_ < 0) throw std::runtime_error("no video stream " + path);
    AVStream* st = fmt_->streams[stream_];
    cc_ = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(cc_, st->codecpar);
    cc_->hw_device_ctx = av_buffer_ref(hw);
    cc_->get_format = pick_cuda;
    cc_->thread_count = 1;
    cc_->extra_hw_frames = 2;
    if (avcodec_open2(cc_, codec, nullptr) < 0) throw std::runtime_error("codec open " + path);
    AVRational r = st->avg_frame_rate.num ? st->avg_frame_rate : st->r_frame_rate;
    fps_ = r.num ? av_q2d(r) : 10.0;
    w_ = cc_->width;
    h_ = cc_->height;
    // colour matrix from the stream tags; untagged streams: BT.709 for HD, BT.601 for SD (FFmpeg's convention)
    matrix_ = cc_->colorspace == AVCOL_SPC_BT709 ? 1
            : (cc_->colorspace == AVCOL_SPC_UNSPECIFIED ? (h_ >= 720 ? 1 : 0) : 0);
    pkt_ = av_packet_alloc();
    frame_ = av_frame_alloc();
}

VideoDecoder::VideoDecoder(const std::string& pipe_path, AVBufferRef* hw, int w, int h, double fps) {
    // non-blocking open: the sender may connect later; next_live() polls
    fd_ = open(pipe_path.c_str(), O_RDONLY | O_NONBLOCK);
    if (fd_ < 0) throw std::runtime_error("open " + pipe_path);
    const AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_H264);
    if (!codec) throw std::runtime_error("no H.264 decoder");
    cc_ = avcodec_alloc_context3(codec);
    cc_->hw_device_ctx = av_buffer_ref(hw);
    cc_->get_format = pick_cuda;
    cc_->thread_count = 1;
    cc_->extra_hw_frames = 2;
    cc_->flags |= AV_CODEC_FLAG_LOW_DELAY;      // camera streams have no reordering: output each frame at once
    cc_->pkt_timebase = AVRational{1, 1000000};
    if (avcodec_open2(cc_, codec, nullptr) < 0) throw std::runtime_error("codec open (live)");
    fps_ = fps;
    w_ = w;
    h_ = h;
    matrix_ = h_ >= 720 ? 1 : 0;
    pkt_ = av_packet_alloc();
    frame_ = av_frame_alloc();
    buf_.resize(1 << 20);
}

VideoDecoder::~VideoDecoder() {
    av_frame_free(&frame_);
    av_packet_free(&pkt_);
    avcodec_free_context(&cc_);
    if (fmt_) avformat_close_input(&fmt_);
    if (fd_ >= 0) close(fd_);
}

AVFrame* VideoDecoder::next_live() {
    // a frame may already be waiting in the decoder
    av_frame_unref(frame_);
    if (avcodec_receive_frame(cc_, frame_) == 0) goto got;
    for (int guard = 0; guard < 64; ++guard) {
        size_t need = 16;
        if (have_ >= 16) {
            uint32_t magic, len;
            std::memcpy(&magic, buf_.data(), 4);
            std::memcpy(&len, buf_.data() + 4, 4);
            if (magic != 0x55415046u || len > (64u << 20)) {       // 'FPAU' little-endian; stream out of step
                std::fprintf(stderr, "[live] bad frame header, closing camera stream\n");
                eof_ = true;
                return nullptr;
            }
            need = 16 + (size_t)len;
            if (buf_.size() < need + AV_INPUT_BUFFER_PADDING_SIZE) buf_.resize(need + AV_INPUT_BUFFER_PADDING_SIZE);
        }
        if (have_ < need) {
            pollfd p{fd_, POLLIN, 0};
            const int pr = poll(&p, 1, 100);
            if (pr <= 0) return nullptr;                               // nothing yet
            const ssize_t r = read(fd_, buf_.data() + have_, need - have_);
            if (r > 0) { have_ += (size_t)r; started_ = true; continue; }
            if (r == 0 || (p.revents & (POLLHUP | POLLERR))) {         // no writer (yet), or the writer left
                if (started_) eof_ = true;
                else std::this_thread::sleep_for(std::chrono::milliseconds(20));
                return nullptr;
            }
            return nullptr;
        }
        if (have_ < 16) continue;
        // one complete access unit
        uint32_t len;
        uint64_t seq;
        std::memcpy(&len, buf_.data() + 4, 4);
        std::memcpy(&seq, buf_.data() + 8, 8);
        t_arr_ = now_s();
        std::memset(buf_.data() + 16 + len, 0, AV_INPUT_BUFFER_PADDING_SIZE);
        av_packet_unref(pkt_);
        pkt_->data = buf_.data() + 16;
        pkt_->size = (int)len;
        pkt_->pts = pkt_->dts = (int64_t)seq;
        const int sr = avcodec_send_packet(cc_, pkt_);
        pkt_->data = nullptr;
        pkt_->size = 0;
        have_ = 0;
        if (sr < 0) continue;                                          // undecodable unit (e.g. before the first keyframe)
        av_frame_unref(frame_);
        if (avcodec_receive_frame(cc_, frame_) == 0) goto got;
    }
    return nullptr;
got:
    seq_ = frame_->pts;
    if (frame_->width != w_ || frame_->height != h_) {
        std::fprintf(stderr, "[live] camera sends %dx%d, expected %dx%d: frame ignored\n", frame_->width, frame_->height, w_, h_);
        return nullptr;
    }
    matrix_ = frame_->colorspace == AVCOL_SPC_BT709 ? 1 : (frame_->colorspace == AVCOL_SPC_UNSPECIFIED ? (h_ >= 720 ? 1 : 0) : 0);
    return frame_;
}

AVFrame* VideoDecoder::next() {
    if (fd_ >= 0) return next_live();
    for (int guard = 0; guard < 100000; ++guard) {
        av_frame_unref(frame_);
        int r = avcodec_receive_frame(cc_, frame_);
        if (r == 0) return frame_;
        if (r != AVERROR(EAGAIN)) return nullptr;
        int rr = av_read_frame(fmt_, pkt_);
        if (rr < 0) {
            // end of the test clip: rewind the demuxer only. Flushing or draining the decoder at every loop makes
            // all cameras on the same clip reset NVDEC at once (a multi-second stall); a live camera never does this.
            av_seek_frame(fmt_, stream_, 0, AVSEEK_FLAG_BACKWARD);
            continue;
        }
        if (pkt_->stream_index == stream_) avcodec_send_packet(cc_, pkt_);
        av_packet_unref(pkt_);
    }
    return nullptr;
}

// ------------------------------------------------------------------------------------------------ DecodePool
namespace {
struct HwDev {
    AVBufferRef* ref = nullptr;
    cudaStream_t stream = nullptr;
};
std::mutex g_hw_m;
std::vector<HwDev> g_hw;
}  // namespace

static HwDev& hw_dev(int t) {
    std::lock_guard<std::mutex> g(g_hw_m);
    if ((int)g_hw.size() <= t) g_hw.resize(t + 1);
    HwDev& d = g_hw[t];
    if (!d.ref) {
        CK(cudaFree(nullptr));                         // runtime primary context exists (flags set in main)
        CUcontext ctx = nullptr;
        CUdevice dev = 0;
        if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS || cuDevicePrimaryCtxRetain(&ctx, dev) != CUDA_SUCCESS)
            throw std::runtime_error("cuDevicePrimaryCtxRetain failed");
        int lo, hi;
        CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
        CK(cudaStreamCreateWithPriority(&d.stream, cudaStreamNonBlocking, hi));
        d.ref = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_CUDA);
        auto* dc = reinterpret_cast<AVHWDeviceContext*>(d.ref->data);
        auto* hw = static_cast<AVCUDADeviceContext*>(dc->hwctx);
        hw->cuda_ctx = ctx;
        hw->stream = reinterpret_cast<CUstream>(d.stream);
        if (av_hwdevice_ctx_init(d.ref) < 0) throw std::runtime_error("av_hwdevice_ctx_init(cuda) failed");
    }
    return d;
}

AVBufferRef* DecodePool::hw_device(int t) { return hw_dev(t).ref; }
cudaStream_t DecodePool::hw_stream(int t) { return hw_dev(t).stream; }

DecodePool::DecodePool(std::vector<CamSource*> cams, int threads, DecodedSink* sink)
    : cams_(std::move(cams)), nthreads_(std::max(1, std::min<int>(threads, (int)cams_.size()))), sink_(sink) {}

DecodePool::~DecodePool() { stop(); }

void DecodePool::start(double t0) {
    for (size_t i = 0; i < cams_.size(); ++i) {   // stagger cameras across one frame period, like unsynchronised cameras
        cams_[i]->period = 1.0 / cams_[i]->dec->fps();
        cams_[i]->next_due = t0 + cams_[i]->period * (double)(i % 97) / 97.0;
    }
    run_ = true;
    for (int t = 0; t < nthreads_; ++t) th_.emplace_back(&DecodePool::loop, this, t);
}

void DecodePool::stop() {
    run_ = false;
    for (auto& t : th_) if (t.joinable()) t.join();
    th_.clear();
}

void DecodePool::loop(int t) {
    std::vector<CamSource*> mine;
    for (size_t i = t; i < cams_.size(); i += nthreads_) mine.push_back(cams_[i]);
    cudaStream_t ff = hw_stream(t);     // FFmpeg's copies for this thread's decoders run here (high priority)
    char name[32];
    std::snprintf(name, sizeof(name), "decode%d", t);
    name_thread(name);
    while (run_ && !mine.empty() && mine[0]->dec->live()) {
        // live camera (one per thread): frames arrive at the camera's own pace; capture time = arrival time
        CamSource* c = mine[0];
        if (c->dec->eof()) { std::this_thread::sleep_for(std::chrono::milliseconds(50)); continue; }
        AVFrame* f = c->dec->next();
        if (!f) continue;
        CK(cudaStreamSynchronize(ff));
        c->fid = c->dec->seq();
        if (c->fid > 0 && std::floor(c->fid * c->ratio) == std::floor((c->fid - 1) * c->ratio)) continue;   // not due
        int slot = c->ring.begin_write();
        if (slot < 0) { c->ring.dropped++; continue; }
        c->ring.end_write(slot, c->fid, now_s(), c->dec->t_arrival(), f);
        if (sink_) sink_->on_frame();
    }
    while (run_) {
        CamSource* c = *std::min_element(mine.begin(), mine.end(),
                                         [](CamSource* a, CamSource* b) { return a->next_due < b->next_due; });
        double wait = c->next_due - now_s();
        if (wait > 0) {
            std::this_thread::sleep_for(std::chrono::duration<double>(std::min(wait, 0.02)));
            continue;
        }
        const double t_cap = c->next_due;      // when the camera would have delivered this frame
        c->next_due += c->period;
        AVFrame* f;
        {
            Nvtx r("nvdec");
            f = c->dec->next();
            CK(cudaStreamSynchronize(ff));      // FFmpeg's surface copy is complete before anyone reads the frame
        }
        if (!f) continue;
        c->fid++;
        if (c->ring.fifo() && c->fid > 0 &&
            std::floor(c->fid * c->ratio) == std::floor((c->fid - 1) * c->ratio)) continue;   // not due
        int slot = c->ring.begin_write();
        if (slot < 0) { c->ring.dropped++; continue; }
        if (!c->ring.zero_copy()) {
            Nvtx r("ring_copy");
            CK(cudaMemcpy2DAsync(c->ring.wy(slot), c->ring.copy_pitch(), f->data[0], f->linesize[0], c->ring.w(),
                                 c->ring.h(), cudaMemcpyDeviceToDevice, ff));
            CK(cudaMemcpy2DAsync(c->ring.wuv(slot), c->ring.copy_pitch(), f->data[1], f->linesize[1], c->ring.w(),
                                 c->ring.h() / 2, cudaMemcpyDeviceToDevice, ff));
            CK(cudaStreamSynchronize(ff));
        }
        c->ring.end_write(slot, c->fid, now_s(), t_cap, f);
        if (sink_) sink_->on_frame();
    }
}
