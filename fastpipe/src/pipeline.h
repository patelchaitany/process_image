#pragma once
// Stage graph: decode -> [K1, TRT det, K2] -> CPU tracker -> [K3, TRT rec, K4] -> finish (ARCHITECTURE.md 2, 6)
#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "decoder.h"
#include "gputrack.cuh"
#include "kernels.cuh"
#include "tracker.h"
#include "trt_engine.h"

struct Config {
    std::string det_plan, rec_plan;
    std::string video = "/mnt/e/bench/sub_rot{r}.mp4";
    std::string out_dir = "results/fastpipe";
    std::string tag = "fastpipe";
    std::vector<int> counts{1, 2, 16, 32, 64, 128};
    double analyze_fps = 15, warmup = 10, window = 30;   // detect every 2nd frame of a 30 fps stream
    int det_batch = 32, rec_batch = 128, inflight = 2, decoders = 8, gallery = 300;
    int det_ctxs = 2, rec_ctxs = 2;
    float conf = 0.5f, iou = 0.45f;
    int max_faces = 64;
    double reverify = 1.0;
    float dist_k = 1.5f, min_rec_px = 0;
    bool track = true, budget = false, graphs = true, mapped = true;
    std::string policy = "fifo";     // latest: newest frame, skip stale | fifo: every frame, buffer_s seconds buffer
    double buffer_s = 3.0;
    int fifo_inflight = 40;           // fifo: frames per camera between scheduling and result
    int zones = 4;                    // zone mask per camera: N vertical sections (0 = none)
    bool zero_copy = true;            // frame ring holds FFmpeg's CUDA frames instead of copying them
    int min_batch = 16;               // detector batching: launch when this many frames are ready ...
    double max_wait_ms = 25;          // ... or the oldest ready frame has waited this long (or the GPU is idle)
    bool kalman = false;              // detect on keyframes only; Kalman-smoothed boxes for every frame in between
    double window_s = 0;              // >0: collect this many seconds of keyframes, then process them as one burst
    float kf_acc = 16.0f, kf_meas = 0.02f;   // Kalman: acceleration noise (box heights/s^2), measurement noise (box heights)
    bool gpu_track = false;           // tracker, Kalman fill, recognition triggers and identity write-back on the GPU
    double recheck_s = 3.0, retry_s = 1.0;
    float grow = 0.5f, cross_iou = 0.5f;
    // Cosine score against the gallery: >= match_thr is a match, < unknown_thr is clearly unknown, between is retried.
    // Scores are not comparable between recognisers: each engine carries its own pair in <engine>.thr (see main.cpp).
    // These defaults are ArcFace R50's.
    float unknown_thr = 0.25f, match_thr = 0.45f;
    bool unknown_thr_set = false, match_thr_set = false;   // given on the command line
    int confirm_hits = 3, pool = 4096, pool_margin = 1024;
    float trk_iou = 0.3f, ttl = 1.5f; // GPU tracker: IoU needed to match a detection to a track; seconds a lost track is kept
    bool unified = true;              // one giant pipeline: the whole tick is a single CUDA graph on one stream
    int uni_rmax = 128;               // unified: most recognitions per tick
    bool uni_split = true;            // two launches per tick (detect+track, then recognise at the exact accepted count)
    double uni_headroom = 1.0;        // unified: recognition slots per tick = demand x this, rounded up (< 1: rounded down)
    bool cb = false;
    bool cb_idle = false;             // continuous batching: launch the detector at once when it is idle                  // continuous batching: stream priorities, bucket-aligned batches, deadline windows
    int rec_min = 32;                 // GPU tracker: send crops to the recogniser once this many are waiting ...
    double rec_wait_ms = 150;         // ... or the oldest has waited this long
    bool bestshot = false;            // recognise each face once per reverify window, from its best frame in the window
    int kfeval = 0;                   // evaluation mode: detect every frame of N frames, score Kalman fill vs detections
    int det_chw = 1;          // detector output layout: 1 = [B,5,A], 0 = [B,A,5]
    int kbench = 0;           // kernel micro-benchmark mode
    int boxeval = 0;          // accuracy mode: N frames through the real GPU pipeline, skipped boxes vs detecting every frame
    int dump = 0;             // validation mode: N frames of camera 0, all faces, write results
    std::string dump_path = "dump.bin";
    // live mode: cameras arrive as H.264 on one pipe each ({r} = camera index); results leave as JSON lines on stdout
    std::string live;
    // < 1: the detector sees the frame shrunk by this factor (padded to its input size). For close-up cameras: the
    // detector stops responding to faces wider than about 200 px of its 640 px input, i.e. 400 px of a 720p frame.
    float det_zoom = 1.0f;
    int live_w = 1280, live_h = 720;
    double live_fps = 30;
};

// Host buffer that the GPU reads or writes: zero-copy mapped memory, or pinned host + device copy.
template <class T>
struct HostDev {
    T* h = nullptr;
    T* d = nullptr;
    bool mapped = true;
    size_t n = 0;
    void alloc(size_t count, bool map, bool write_combined);
    void free_all();
    void to_device(size_t count, cudaStream_t s);   // no-op when mapped
    void to_host(size_t count, cudaStream_t s);     // no-op when mapped
};

struct Cam;
struct Item {
    Cam* cam;
    int slot;
    int64_t fid;
    double t_arr, t_read;   // decoded frame published / taken into a detector batch
    int due;
    double t_cap;           // when the camera produced the frame (real-time schedule)
    int gap = 1;            // frames since this camera's previous analysed frame (GPU tracker)
};

struct Job {                 // one frame waiting for its face recognitions (or one camera's best-frame window)
    Item item;
    int n_faces;
    int first, count;        // range in the recognition batch
    std::vector<double> t_first;   // first-seen time per crop for new tracks, < 0 for re-checks
    bool frame_job = true;   // false: best-frame window job (frames already finished; release pins when done)
    Cam* cam = nullptr;
    std::vector<int> pins;   // ring slots pinned by the window's best frames
    double t_window_end = 0; // capture-time end of the window
};

struct DetCtx {
    // GPU-tracker mode: this batch's crop requests and their count (device), count mirrored to host-mapped memory
    CropReq* creq = nullptr;
    int* creq_slot = nullptr;
    int* ccount = nullptr;
    int* ccount_host = nullptr;
    __half* in = nullptr;
    __half* out = nullptr;
    HostDev<FrameDets> dets;
    cudaEvent_t ev0{}, ev1{};
    int n = 0, bucket = 0;
    Item items[MAX_DET_B];
    // unified pipeline: this tick's parameters and results (pinned host memory), and its recognition cap
    TickDesc* desc_h = nullptr;
    TickStats* stats_h = nullptr;
    int R = 0;
    int* cnt_h = nullptr;            // accepted count of this tick, copied after graph A
    cudaEvent_t evA{}, evB{};       // end of graph A (count ready), start of graph B
    // live mode: this tick's delivered boxes (pinned), and the enrolment it tried, if any
    LiveBox* live_h = nullptr;
    int* enroll_h = nullptr;
    int enroll_label = -1;
    FrameDets* dbg_h = nullptr;     // FP_LIVE_DEBUG: the detector's raw result for the tick's first frame
};

struct RecCtx {
    __half* in = nullptr;
    __half* emb = nullptr;
    HostDev<CropReq> req;
    HostDev<MatchOut> match;
    cudaEvent_t ev0{}, ev1{};
    int n = 0, bucket = 0;
    std::vector<Job> jobs;
    unsigned long long pool_start = 0;   // GPU-tracker mode: crop-pool range [pool_start, pool_start + n)
};

struct Row {
    int cam;
    int64_t fid;
    double t_arr, t_read, t_done;
    int n_rec, n_faces;
    double t_cap;
};

struct Cam {
    CamSource src;
    uint8_t* zone_mask = nullptr;   // device, w x h labels
    int64_t zone_count[8] = {};      // faces seen per zone (presence counting)
    double ratio = 1.0;
    int64_t last = -1;       // last frame id taken (or skipped as not due)
    bool in_det = false;
    int inflight = 0;
    Tracker tracker;
    KalmanTracker ktracker;
    int64_t last_key = -1;           // previous keyframe id (Kalman mode)
    int64_t cur_w = -1;              // current best-frame window index
    struct Best { float q; int slot; float4 box; double t_first, t_cap, prev_rec; };
    std::unordered_map<uint64_t, Best> best;   // per track: best frame so far in this window
    int64_t out_frames = 0, filled_frames = 0;
    // end-to-end accounting per second of video: when the last result of that second was ready, frames delivered
    std::vector<double> chunk_done;
    std::vector<int> chunk_frames;
    std::vector<Row> rows;
    int64_t eligible = 0, skipped = 0;
};

class Pipeline : public DecodedSink {
public:
    Pipeline(const Config& cfg, TrtEngine& det, TrtEngine& rec, const __half* gallery_d, int G);
    ~Pipeline() override;
    std::string run_count(int n);     // one stress run; returns the JSON summary line
    std::string eval_boxes(int frames);   // skipped-box accuracy of the unified pipeline on one clip
    void run_live(int n);                 // n live cameras until stdin closes; control lines on stdin, results on stdout
    void on_frame() override;

private:
    void sched_loop();
    void track_loop();
    void finish_loop();
    void process_det(DetCtx* c);
    void submit_rec(RecCtx* r);
    void finish_item(const Item& it, int n_rec, int n_faces, double t_done);
    void flush_best(Cam* cam, double t_window_end);
    void alloc_gpu_track(int max_cams);
    void reset_gpu_track();
    void submit_rec_gpu(RecCtx* r);
    void alloc_unified();
    void capture_unified(int B, int R);
    void capture_split_A(int B);
    void capture_split_B(int R);
    CropPool uni_cp(int R) const;
    void submit_tick(DetCtx* c, const DetBatch& b, const ZoneMasks& zm);
    void live_tail(DetCtx* c);            // live mode: gather this tick's rows, try a pending enrolment
    void emit_live(DetCtx* c, double now);
    // zones: (label 1..255, polygon as x0 y0 x1 y1 ... in 0..1 of the frame); later zones lie on top of earlier ones
    void set_zones(int cam, const std::vector<std::pair<int, std::vector<float>>>& zones);
    void sample_gpu();

    const Config& cfg_;
    TrtEngine& det_;
    TrtEngine& rec_;
    const __half* gallery_;
    int G_;
    cudaStream_t det_s_{}, rec_s_{};
    std::vector<std::unique_ptr<DetCtx>> det_ctx_;
    std::vector<std::unique_ptr<RecCtx>> rec_ctx_;

    // run state
    std::vector<std::unique_ptr<Cam>> cams_;
    std::atomic<bool> run_{false}, recording_{false};
    std::mutex m_;                                     // camera scheduling state + rows
    std::condition_variable sched_cv_;
    int rr_ = 0;
    std::mutex q_m_;                                   // queues between stages
    std::condition_variable q_cv_, det_free_cv_, rec_done_cv_;
    std::deque<DetCtx*> det_free_, det_done_;
    std::deque<RecCtx*> rec_free_, rec_done_;
    // pending recognition work (track thread only)
    std::vector<FillBox> fills_;                     // Kalman fills of the current keyframe (track thread)
    std::vector<CropReq> pend_req_;
    std::vector<Job> pend_jobs_;
    std::atomic<int> backlog_{0};
    std::atomic<int> det_inflight_{0};                 // detector batches submitted, not yet tracked                      // crops pending or in flight in the recogniser
    // metrics
    std::mutex stat_m_;
    std::vector<std::pair<int, float>> det_calls_, rec_calls_;   // (batch, GPU ms)
    double track_cpu_s_ = 0;
    std::vector<float> out_delay_;     // capture -> boxes available, for every output frame (Kalman mode)
    bool draining_ = false;
    // GPU-tracker state
    int max_cams_ = 0;
    TrackTable tt_{};
    TrackOut tout_{};
    CropPool cp_{};
    TrackParams tp_{};
    __half* pool_tensor_ = nullptr;
    unsigned long long* released_host_ = nullptr;   // host-mapped, read by the tracker kernel
    unsigned long long pool_head_ = 0, pool_tail_ = 0;
    double t0_run_ = 0;
    // unified pipeline: one set of device buffers (ticks run one after another on one stream) and one graph per
    // (frames bucket, recognitions bucket)
    struct Uni {
        TickDesc* desc = nullptr;
        TickStats* stats = nullptr;
        __half *det_in = nullptr, *det_out = nullptr, *rec_in = nullptr, *emb = nullptr;
        FrameDets* dets = nullptr;
        CropReq* req = nullptr;
        int* req_slot = nullptr;
        PoolMeta* meta = nullptr;
        MatchOut* match = nullptr;
        RecCand* cand = nullptr;
        LiveBox* live = nullptr;
        int* enroll_done = nullptr;
    } u_;
    struct Enroll { int cam; unsigned tid; int label; double t_req; };
    std::mutex enroll_m_;
    std::deque<Enroll> enroll_q_;
    std::map<std::pair<int, int>, cudaGraphExec_t> uni_graphs_;
    std::vector<int> uni_R_;
    std::map<int, cudaGraphExec_t> uniA_, uniB_;   // graph A per frames bucket, graph B per recognition size
    double splitA_ms_ = 0, splitB_ms_ = 0, split_gap_ms_ = 0;
    std::atomic<double> denied_ema_{0.0};          // refused requests per tick (they come back with priority)
    std::atomic<double> req_per_frame_ema_{3.0};   // recognition demand per detected frame (requests incl. denied)
    double tick_ms_sum_ = 0;
    long ticks_ = 0, uni_slots_ = 0, uni_used_ = 0, uni_denied_ = 0, uni_boxes_ = 0, uni_named_ = 0;
    int name_wait_max_ms_ = 0;
    std::vector<double> name_delay_;      // request (frame capture) -> name written
    double pend_since_ = -1;          // when the oldest crop still waiting for the recogniser was queued
    unsigned long long gpu_c0_[11] = {}, gpu_c1_[11] = {};
    std::vector<double> first_id_, refresh_, id_after_window_;
    int64_t deferred_ = 0;
    std::vector<std::array<double, 5>> gpu_samples_;
};
