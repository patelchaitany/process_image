#pragma once
// Per-camera NVDEC decode (libavcodec "cuda" hwaccel) into a device frame ring.
#include <cuda_runtime.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

extern "C" {
struct AVBufferRef;
struct AVFormatContext;
struct AVCodecContext;
struct AVPacket;
struct AVFrame;
}

// Device frame buffer for one camera (NV12, one pitched allocation per slot). Two policies:
//  latest : 4 slots; the scheduler always takes the newest frame and older ones are skipped (bounded delay,
//           frame rate degrades under overload)
//  fifo   : a buffer of buffer_s seconds; every due frame is processed in order. If processing falls more than
//           buffer_s behind, the oldest queued frame is evicted and counted as an overflow (real-time violation).
// A slot is pinned (refs > 0) from scheduling until the frame's last kernel has finished.
class FrameRing {
public:
    // zero_copy: slots hold references to FFmpeg's decoded CUDA frames (no device-to-device copy)
    void alloc(int w, int h, int slots, bool fifo, bool zero_copy);
    void free_all();
    int begin_write();                                   // slot to decode into, or -1 (drop)
    // publish; zero-copy rings take a reference to `src`, copy rings expect the data already in wy()/wuv()
    void end_write(int s, int64_t fid, double t_arr, double t_cap, AVFrame* src);
    int acquire_latest(int64_t after, int64_t& fid, double& t_arr, double& t_cap);   // latest policy
    int acquire_next(int64_t after, int64_t& fid, double& t_arr, double& t_cap);     // fifo: oldest queued frame
    // readiness without pinning: newest (latest) / oldest queued (fifo) frame id after `after`, and its capture time
    int64_t peek(int64_t after, double& t_cap);
    int queued();
    void release(int s);
    void addref(int s);
    const uint8_t* y(int s) const { return y_[s]; }
    const uint8_t* uv(int s) const { return uv_[s]; }
    int pitch(int s) const { return pitch_[s]; }
    uint8_t* wy(int s) { return y_[s]; }
    uint8_t* wuv(int s) { return uv_[s]; }
    int copy_pitch() const { return own_pitch_; }
    int w() const { return w_; }
    int h() const { return h_; }
    bool fifo() const { return fifo_; }
    bool zero_copy() const { return zero_copy_; }
    std::atomic<int64_t> dropped{0}, overflow{0}, arrived{0};

private:
    std::mutex m_;
    std::vector<uint8_t*> own_, y_, uv_;
    std::vector<int> pitch_;
    std::vector<AVFrame*> held_;
    std::vector<int> refs_;
    std::vector<char> writing_, queued_;
    std::vector<int64_t> fid_;
    std::vector<double> t_, tc_;
    int latest_ = -1;
    int own_pitch_ = 0, w_ = 0, h_ = 0;
    bool fifo_ = false, zero_copy_ = false;
};

class VideoDecoder {
public:
    VideoDecoder(const std::string& path, AVBufferRef* hw_device);
    // Live camera: H.264 access units arrive on a pipe, each behind a 16-byte header
    // {u32 'FPAU', u32 bytes, u64 frame number}. The frame number travels through the decoder (packet pts), so every
    // result can be matched to the frame the sender kept.
    VideoDecoder(const std::string& pipe_path, AVBufferRef* hw_device, int w, int h, double fps);
    ~VideoDecoder();
    // Decode the next frame (loops at end of file). Returns the hardware frame (owned by the decoder).
    // Live: waits up to 100 ms for data and returns null if no complete frame arrived yet.
    AVFrame* next();
    bool live() const { return fd_ >= 0; }
    bool eof() const { return eof_; }
    int64_t seq() const { return seq_; }            // live: frame number of the frame next() just returned
    double t_arrival() const { return t_arr_; }     // live: when that frame's last byte arrived
    double fps() const { return fps_; }
    int width() const { return w_; }
    int matrix() const { return matrix_; }
    int height() const { return h_; }

private:
    AVFormatContext* fmt_ = nullptr;
    AVCodecContext* cc_ = nullptr;
    AVPacket* pkt_ = nullptr;
    AVFrame* frame_ = nullptr;
    int stream_ = -1;
    double fps_ = 10;
    int w_ = 0, h_ = 0;
    int matrix_ = 0;
    // live source
    AVFrame* next_live();
    int fd_ = -1;
    bool eof_ = false, started_ = false;
    std::vector<uint8_t> buf_;
    size_t have_ = 0;
    int64_t seq_ = -1;
    double t_arr_ = 0;
};

struct DecodedSink {
    virtual void on_frame() = 0;   // wake-up hint for the scheduler
    virtual ~DecodedSink() = default;
};

// One camera = decoder + ring + real-time schedule
struct CamSource {
    int id = 0;
    double ratio = 1.0;      // analysis rate / stream rate (fifo: decimation at the decoder)
    std::unique_ptr<VideoDecoder> dec;
    FrameRing ring;
    double next_due = 0;      // capture time of the next frame (real-time schedule)
    double period = 0.1;
    int64_t fid = -1;
};

// Pool of decoder threads; each thread serves its cameras in due-time order, at real-time pace.
class DecodePool {
public:
    DecodePool(std::vector<CamSource*> cams, int threads, DecodedSink* sink);
    ~DecodePool();
    void start(double t0);
    void stop();
    // One FFmpeg CUDA device per decoder thread: primary context + its own high-priority stream, so FFmpeg's
    // surface copies are not queued behind TensorRT kernels.
    static AVBufferRef* hw_device(int thread = 0);
    static cudaStream_t hw_stream(int thread = 0);

private:
    void loop(int t);
    std::vector<CamSource*> cams_;
    int nthreads_;
    DecodedSink* sink_;
    std::vector<std::thread> th_;
    std::atomic<bool> run_{false};
};
