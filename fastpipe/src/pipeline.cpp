#include "pipeline.h"
#include <functional>
#include <iostream>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <sstream>

#include "common.h"

extern "C" {
#include <libavutil/frame.h>
}

// ------------------------------------------------------------------------------------------------ HostDev
template <class T>
void HostDev<T>::alloc(size_t count, bool map, bool write_combined) {
    n = count;
    mapped = map;
    unsigned flags = cudaHostAllocDefault;
    if (map) flags |= cudaHostAllocMapped;
    if (write_combined) flags |= cudaHostAllocWriteCombined;
    CK(cudaHostAlloc(reinterpret_cast<void**>(&h), sizeof(T) * count, flags));
    if (map) CK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d), h, 0));
    else CK(cudaMalloc(reinterpret_cast<void**>(&d), sizeof(T) * count));
}
template <class T>
void HostDev<T>::free_all() {
    if (h) cudaFreeHost(h);
    if (d && !mapped) cudaFree(d);
    h = d = nullptr;
}
template <class T>
void HostDev<T>::to_device(size_t count, cudaStream_t s) {
    if (!mapped && count) CK(cudaMemcpyAsync(d, h, sizeof(T) * count, cudaMemcpyHostToDevice, s));
}
template <class T>
void HostDev<T>::to_host(size_t count, cudaStream_t s) {
    if (!mapped && count) CK(cudaMemcpyAsync(h, d, sizeof(T) * count, cudaMemcpyDeviceToHost, s));
}
template struct HostDev<FrameDets>;
template struct HostDev<CropReq>;
template struct HostDev<MatchOut>;

// ------------------------------------------------------------------------------------------------ setup
Pipeline::Pipeline(const Config& cfg, TrtEngine& det, TrtEngine& rec, const __half* gallery_d, int G)
    : cfg_(cfg), det_(det), rec_(rec), gallery_(gallery_d), G_(G) {
    int lo, hi;
    CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
    // continuous batching: frames (tight deadline) get the high-priority stream; recognition back-fills idle SMs
    CK(cudaStreamCreateWithPriority(&det_s_, cudaStreamNonBlocking, cfg_.cb ? hi : lo));
    CK(cudaStreamCreateWithPriority(&rec_s_, cudaStreamNonBlocking, lo));
    const int DB = det_.max_batch(), RB = rec_.max_batch();
    for (int i = 0; i < cfg_.det_ctxs; ++i) {
        auto c = std::make_unique<DetCtx>();
        CK(cudaMalloc(&c->in, sizeof(__half) * det_.in_item_elems() * DB));
        CK(cudaMalloc(&c->out, sizeof(__half) * det_.out_item_elems() * DB));
        CK(cudaMemset(c->in, 0, sizeof(__half) * det_.in_item_elems() * DB));
        c->dets.alloc(MAX_DET_B, cfg_.mapped && !cfg_.gpu_track, false);   // GPU tracker reads them on the device
        if (cfg_.gpu_track) {
            CK(cudaMalloc(&c->creq, sizeof(CropReq) * MAX_DET_B * MAX_FACES));
            CK(cudaMalloc(&c->creq_slot, sizeof(int) * MAX_DET_B * MAX_FACES));
            CK(cudaMalloc(&c->ccount, sizeof(int)));
            CK(cudaHostAlloc(reinterpret_cast<void**>(&c->ccount_host), sizeof(int), cudaHostAllocMapped));
        }
        if (cfg_.unified) {
            CK(cudaHostAlloc(reinterpret_cast<void**>(&c->desc_h), sizeof(TickDesc), cudaHostAllocDefault));
            CK(cudaHostAlloc(reinterpret_cast<void**>(&c->stats_h), sizeof(TickStats), cudaHostAllocDefault));
            CK(cudaHostAlloc(reinterpret_cast<void**>(&c->cnt_h), sizeof(int), cudaHostAllocDefault));
            CK(cudaEventCreate(&c->evA));
            CK(cudaEventCreate(&c->evB));
            if (!cfg_.live.empty()) {
                CK(cudaHostAlloc(reinterpret_cast<void**>(&c->live_h), sizeof(LiveBox) * MAX_DET_B * LIVE_ROWS * TRK, cudaHostAllocDefault));
                CK(cudaHostAlloc(reinterpret_cast<void**>(&c->enroll_h), sizeof(int), cudaHostAllocDefault));
                if (std::getenv("FP_LIVE_DEBUG"))
                    CK(cudaHostAlloc(reinterpret_cast<void**>(&c->dbg_h), sizeof(FrameDets), cudaHostAllocDefault));
            }
        }
        CK(cudaEventCreate(&c->ev0));
        CK(cudaEventCreate(&c->ev1));
        det_ctx_.push_back(std::move(c));
    }
    for (int i = 0; i < cfg_.rec_ctxs; ++i) {
        auto r = std::make_unique<RecCtx>();
        CK(cudaMalloc(&r->in, sizeof(__half) * rec_.in_item_elems() * RB));
        CK(cudaMalloc(&r->emb, sizeof(__half) * rec_.out_item_elems() * RB));
        CK(cudaMemset(r->in, 0, sizeof(__half) * rec_.in_item_elems() * RB));
        r->req.alloc(RB, cfg_.mapped, true);       // CPU writes, GPU reads once: write-combined
        r->match.alloc(RB, cfg_.mapped && !cfg_.gpu_track, false);
        CK(cudaEventCreate(&r->ev0));
        CK(cudaEventCreate(&r->ev1));
        rec_ctx_.push_back(std::move(r));
    }
    if (cfg_.gpu_track) alloc_gpu_track(256);
    const double t0 = now_s();
    if (cfg_.unified) {
        alloc_unified();     // captures the whole-pipeline graphs (includes the engines' first runs)
        std::fprintf(stderr, "[pipeline] unified: %zu whole-pipeline graphs captured in %.1f s\n", uni_graphs_.size(), now_s() - t0);
    } else {
        // capture all CUDA graphs up front (one thread, no concurrent work), so no capture happens mid-run
        for (auto& c : det_ctx_) det_.warm(c->in, c->out, det_s_);
        for (auto& r : rec_ctx_) rec_.warm(r->in, r->emb, rec_s_);
        std::fprintf(stderr, "[pipeline] engines warmed%s in %.1f s\n", cfg_.graphs ? " and graphs captured" : "", now_s() - t0);
    }
}

Pipeline::~Pipeline() {
    cudaDeviceSynchronize();
    for (auto& c : det_ctx_) { cudaFree(c->in); cudaFree(c->out); c->dets.free_all(); cudaEventDestroy(c->ev0); cudaEventDestroy(c->ev1); }
    for (auto& r : rec_ctx_) { cudaFree(r->in); cudaFree(r->emb); r->req.free_all(); r->match.free_all(); cudaEventDestroy(r->ev0); cudaEventDestroy(r->ev1); }
    cudaStreamDestroy(det_s_);
    cudaStreamDestroy(rec_s_);
}

void Pipeline::on_frame() { sched_cv_.notify_one(); }

// ------------------------------------------------------------------------------------------------ GPU tracker
void Pipeline::alloc_gpu_track(int max_cams) {
    max_cams_ = max_cams;
    const size_t N = (size_t)max_cams * TRK;
    auto fa = [&](float*& p, size_t n) { CK(cudaMalloc(&p, n * sizeof(float))); };
    fa(tt_.x, 4 * N); fa(tt_.v, 4 * N); fa(tt_.P00, 4 * N); fa(tt_.P01, 4 * N); fa(tt_.P11, 4 * N);
    fa(tt_.seen, N); fa(tt_.first, N); fa(tt_.rec_t, N); fa(tt_.rec_area, N); fa(tt_.id_score, N);
    fa(tt_.pend_t, N);
    CK(cudaMalloc(&tt_.hits, N * sizeof(int)));
    CK(cudaMalloc(&tt_.label, N * sizeof(int)));
    CK(cudaMalloc(&tt_.gen, N * sizeof(unsigned)));
    CK(cudaMalloc(&tt_.tid, N * sizeof(unsigned)));
    CK(cudaMalloc(&tt_.alive, N));
    CK(cudaMalloc(&tt_.known, N));
    CK(cudaMalloc(&tt_.next_tid, max_cams * sizeof(unsigned)));
    CK(cudaMalloc(&tout_.box, (size_t)max_cams * OUT_RING * TRK * sizeof(float4)));
    CK(cudaMalloc(&tout_.id, (size_t)max_cams * OUT_RING * TRK * sizeof(int)));
    CK(cudaMalloc(&tout_.frames, max_cams * sizeof(unsigned long long)));
    CK(cudaMalloc(&tout_.faces, max_cams * sizeof(unsigned long long)));
    CK(cudaMalloc(&tout_.requests, max_cams * sizeof(unsigned long long)));
    CK(cudaMalloc(&tout_.reasons, 5 * sizeof(unsigned long long)));
    CK(cudaMalloc(&tout_.missed, sizeof(unsigned long long)));
    CK(cudaMalloc(&tout_.zone, (size_t)max_cams * OUT_RING * TRK));
    cp_.pool = cfg_.pool;
    cp_.margin = cfg_.pool_margin;
    CK(cudaMalloc(&cp_.head, sizeof(unsigned long long)));
    CK(cudaHostAlloc(reinterpret_cast<void**>(&released_host_), sizeof(unsigned long long), cudaHostAllocMapped));
    cp_.released = released_host_;
    CK(cudaMalloc(&cp_.meta, cfg_.pool * sizeof(PoolMeta)));
    CK(cudaHostAlloc(reinterpret_cast<void**>(&cp_.tfirst_host), cfg_.pool * sizeof(float), cudaHostAllocMapped));
    // + one recogniser batch of padding: the last call of the ring may read up to a bucket past its range
    CK(cudaMalloc(&pool_tensor_, (size_t)(cfg_.pool + rec_.max_batch()) * rec_.in_item_elems() * sizeof(__half)));
    CK(cudaMemset(pool_tensor_, 0, (size_t)(cfg_.pool + rec_.max_batch()) * rec_.in_item_elems() * sizeof(__half)));
    tp_ = TrackParams{cfg_.trk_iou, cfg_.dist_k, cfg_.kf_meas, cfg_.kf_acc, 0.3f, cfg_.ttl, (float)cfg_.recheck_s, (float)cfg_.retry_s,
                      cfg_.grow, cfg_.min_rec_px, cfg_.unknown_thr, cfg_.match_thr, cfg_.cross_iou, cfg_.confirm_hits};
    tt_.ncam = max_cams;
    reset_gpu_track();
    std::fprintf(stderr, "[gpu-track] %d cameras x %d tracks, crop pool %d (%.0f MB)\n", max_cams, TRK, cfg_.pool,
                 (cfg_.pool + rec_.max_batch()) * rec_.in_item_elems() * 2 / 1e6);
}

void Pipeline::reset_gpu_track() {
    const size_t N = (size_t)max_cams_ * TRK;
    CK(cudaMemset(tt_.alive, 0, N));
    CK(cudaMemset(tt_.known, 0, N));
    CK(cudaMemset(tt_.gen, 0, N * sizeof(unsigned)));
    CK(cudaMemset(tt_.next_tid, 0, max_cams_ * sizeof(unsigned)));
    CK(cudaMemset(tout_.id, 0, (size_t)max_cams_ * OUT_RING * TRK * sizeof(int)));
    CK(cudaMemset(tout_.zone, 0, (size_t)max_cams_ * OUT_RING * TRK));
    CK(cudaMemset(tout_.frames, 0, max_cams_ * sizeof(unsigned long long)));
    CK(cudaMemset(tout_.faces, 0, max_cams_ * sizeof(unsigned long long)));
    CK(cudaMemset(tout_.requests, 0, max_cams_ * sizeof(unsigned long long)));
    CK(cudaMemset(tout_.reasons, 0, 5 * sizeof(unsigned long long)));
    CK(cudaMemset(tout_.missed, 0, sizeof(unsigned long long)));
    CK(cudaMemset(cp_.head, 0, sizeof(unsigned long long)));
    *released_host_ = 0;
    pool_head_ = pool_tail_ = 0;
    CK(cudaDeviceSynchronize());
}

// ------------------------------------------------------------------------------------------------ unified pipeline
// One tick = one batch of frames taken all the way to names:
//   K1 prepare -> detector -> K2 NMS -> K5 track + decide -> crop -> recogniser -> K4 match -> K6 write names
// captured once per (frames bucket B, recognitions bucket R) as a single CUDA graph. Everything that changes per tick
// is read from device memory (TickDesc), so launching a tick is: copy 3 KB of parameters, one cudaGraphLaunch, copy
// back 520 bytes of statistics. No second stream, no crop pool, no CPU decision between stages.
void Pipeline::alloc_unified() {
    const int DB = det_.max_batch(), RB = rec_.max_batch();
    auto dm = [](auto*& p, size_t bytes) {
        CK(cudaMalloc(reinterpret_cast<void**>(&p), bytes));
        CK(cudaMemset(p, 0, bytes));
    };
    dm(u_.desc, sizeof(TickDesc));
    dm(u_.stats, sizeof(TickStats));
    dm(u_.det_in, sizeof(__half) * det_.in_item_elems() * DB);
    dm(u_.det_out, sizeof(__half) * det_.out_item_elems() * DB);
    dm(u_.dets, sizeof(FrameDets) * MAX_DET_B);
    dm(u_.req, sizeof(CropReq) * RB);
    dm(u_.req_slot, sizeof(int) * RB);
    dm(u_.meta, sizeof(PoolMeta) * RB);
    dm(u_.rec_in, sizeof(__half) * rec_.in_item_elems() * RB);
    dm(u_.emb, sizeof(__half) * rec_.out_item_elems() * RB);
    dm(u_.match, sizeof(MatchOut) * RB);
    dm(u_.cand, sizeof(RecCand) * MAX_DET_B * MAX_FACES);
    if (!cfg_.live.empty()) {
        dm(u_.live, sizeof(LiveBox) * MAX_DET_B * LIVE_ROWS * TRK);
        dm(u_.enroll_done, sizeof(int));
    }
    if (cfg_.uni_split) {
        // graph A per frames bucket (detector .. K5b at the largest cap), graph B per recogniser size (every 4)
        const int rmax = std::min(RB, cfg_.uni_rmax);
        for (int r = 0; r < rmax; r += 4) uni_R_.push_back(r);
        uni_R_.push_back(rmax);
        for (int B : det_.buckets()) capture_split_A(B);
        for (int R : uni_R_) capture_split_B(R);
        return;
    }
    for (int r : {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 88, 96, 112, 128})
        if (r <= std::min(RB, cfg_.uni_rmax)) uni_R_.push_back(r);
    for (int B : det_.buckets())
        for (int R : uni_R_) capture_unified(B, R);
}

CropPool Pipeline::uni_cp(int R) const {
    CropPool cp{};
    cp.req = u_.req;
    cp.req_slot = u_.req_slot;
    cp.batch_count = &u_.stats->count;
    cp.denied = &u_.stats->denied;
    cp.meta = u_.meta;
    cp.tfirst_host = u_.stats->tfirst;
    cp.treq = u_.stats->treq;
    cp.rcam = u_.stats->rcam;
    cp.max_pend_ms = &u_.stats->max_pend_ms;
    cp.cand = u_.cand;
    cp.cand_count = &u_.stats->cands;
    cp.pool = 0;
    cp.margin = R;
    return cp;
}

static cudaGraphExec_t capture_twice(cudaStream_t s, const std::function<void()>& enqueue) {
    enqueue();                                   // eager run first: TensorRT finishes any deferred set-up
    CK(cudaStreamSynchronize(s));
    cudaGraph_t g;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    enqueue();
    CK(cudaStreamEndCapture(s, &g));
    cudaGraphExec_t ge;
    CK(cudaGraphInstantiate(&ge, g, 0));
    CK(cudaGraphDestroy(g));
    CK(cudaStreamSynchronize(s));
    return ge;
}

// Graph A(B): K1 -> detector -> K2 -> K5 -> K5b with the largest cap. The accepted count is then known on the GPU.
void Pipeline::capture_split_A(int B) {
    const int A = (int)(det_.out_item_elems() / 5);
    const CropPool cp = uni_cp(uni_R_.back());
    uniA_[B] = capture_twice(det_s_, [&] {
        CK(cudaMemsetAsync(u_.stats, 0, 6 * sizeof(int), det_s_));
        launch_prep_det_u(u_.desc, B, u_.det_in, det_s_);
        det_.run_direct(B, u_.det_in, u_.det_out, det_s_);
        launch_decode_nms_u(u_.det_out, B, A, cfg_.det_chw, cfg_.conf, cfg_.iou, u_.desc, u_.dets, det_s_);
        launch_track_update_u(u_.dets, B, u_.desc, tt_, tp_, cp, tout_, det_s_);
        launch_assign_slots(u_.desc, tt_, cp, tout_, det_s_);
    });
}

// Graph B(R): crop -> recogniser at R -> K4 -> K6 -> K7. K7's grid covers the largest frames bucket (blocks beyond
// the tick's frame count exit at once), so B depends on R only.
void Pipeline::capture_split_B(int R) {
    const CropPool cp = uni_cp(uni_R_.back());
    const int BM = det_.buckets().back();
    uniB_[R] = capture_twice(det_s_, [&] {
        if (R > 0) {
            launch_crop_to_pool(cp, R, u_.rec_in, det_s_);
            rec_.run_direct(R, u_.rec_in, u_.emb, det_s_);
            launch_match(u_.emb, R, gallery_, G_, u_.match, det_s_);
            launch_scatter_identity(u_.match, R, 0, cp, tt_, cfg_.match_thr, &u_.stats->count, det_s_);
        }
        launch_finalize_output(u_.desc, BM, tt_, tout_, u_.stats, det_s_);
    });
}

void Pipeline::capture_unified(int B, int R) {
    const int A = (int)(det_.out_item_elems() / 5);
    auto enqueue = [&] {
        CK(cudaMemsetAsync(u_.stats, 0, 6 * sizeof(int), det_s_));          // the tick's counters
        launch_prep_det_u(u_.desc, B, u_.det_in, det_s_);
        det_.run_direct(B, u_.det_in, u_.det_out, det_s_);
        launch_decode_nms_u(u_.det_out, B, A, cfg_.det_chw, cfg_.conf, cfg_.iou, u_.desc, u_.dets, det_s_);
        CropPool cp{};
        cp.req = u_.req;
        cp.req_slot = u_.req_slot;
        cp.batch_count = &u_.stats->count;
        cp.denied = &u_.stats->denied;
        cp.meta = u_.meta;
        cp.tfirst_host = u_.stats->tfirst;
        cp.treq = u_.stats->treq;
        cp.rcam = u_.stats->rcam;
        cp.max_pend_ms = &u_.stats->max_pend_ms;
        cp.cand = u_.cand;
        cp.cand_count = &u_.stats->cands;
        cp.pool = 0;          // crops go straight into this tick's recogniser input
        cp.margin = R;        // this tick's recognition cap, enforced by the tracker kernel
        launch_track_update_u(u_.dets, B, u_.desc, tt_, tp_, cp, tout_, det_s_);
        launch_assign_slots(u_.desc, tt_, cp, tout_, det_s_);               // K5b: longest-waiting faces first
        if (R > 0) {
            launch_crop_to_pool(cp, R, u_.rec_in, det_s_);
            rec_.run_direct(R, u_.rec_in, u_.emb, det_s_);
            launch_match(u_.emb, R, gallery_, G_, u_.match, det_s_);
            launch_scatter_identity(u_.match, R, 0, cp, tt_, cfg_.match_thr, &u_.stats->count, det_s_);
        }
        launch_finalize_output(u_.desc, B, tt_, tout_, u_.stats, det_s_);   // names into the delivered rows
    };
    enqueue();                                   // eager run first: TensorRT finishes any deferred set-up
    CK(cudaStreamSynchronize(det_s_));
    cudaGraph_t g;
    CK(cudaStreamBeginCapture(det_s_, cudaStreamCaptureModeThreadLocal));
    enqueue();
    CK(cudaStreamEndCapture(det_s_, &g));
    cudaGraphExec_t ge;
    CK(cudaGraphInstantiate(&ge, g, 0));
    CK(cudaGraphDestroy(g));
    uni_graphs_[{B, R}] = ge;
}

void Pipeline::submit_tick(DetCtx* c, const DetBatch& b, const ZoneMasks& zm) {
    const int n = c->n;
    TickDesc& d = *c->desc_h;
    d.det = b;
    d.zones = zm;
    d.n = n;
    for (int i = 0; i < n; ++i) {
        const Item& it = c->items[i];
        d.trk[i] = TrackItem{it.cam->src.id, (int)it.fid, it.gap, (float)(it.t_cap - t0_run_), (float)it.cam->src.period, b.f[i]};
    }
    // Recognition slots this tick: recent demand per frame x frames x headroom, rounded UP to a captured size, so
    // refusals are rare. A request over the cap is refused on the GPU and comes back on that face's next detected frame.
    const double want = req_per_frame_ema_.load() * n * cfg_.uni_headroom;
    int R = 0;
    if (cfg_.uni_headroom >= 1.0) {
        R = uni_R_.back();
        for (auto it = uni_R_.rbegin(); it != uni_R_.rend(); ++it) if (*it >= want) R = *it;
    } else {
        for (int r : uni_R_) if (r <= want) R = r;
        if (R == 0 && want >= 1.0 && uni_R_.size() > 1) R = uni_R_[1];
    }
    d.reserve = 0;
    if (cfg_.uni_split) {
        // A -> 4-byte count to the host -> B at the smallest captured size >= count. Ticks stay strictly in
        // order on the stream: this tick's B is enqueued before the next tick's A (same thread, same stream).
        CK(cudaEventRecord(c->ev0, det_s_));
        CK(cudaMemcpyAsync(u_.desc, c->desc_h, sizeof(TickDesc), cudaMemcpyHostToDevice, det_s_));
        CK(cudaGraphLaunch(uniA_.at(c->bucket), det_s_));
        CK(cudaMemcpyAsync(c->cnt_h, &u_.stats->count, sizeof(int), cudaMemcpyDeviceToHost, det_s_));
        CK(cudaEventRecord(c->evA, det_s_));
        cudaError_t e;
        while ((e = cudaEventQuery(c->evA)) == cudaErrorNotReady) { }   // spin: the GPU waits for B meanwhile
        CK(e);
        const int cnt = *(volatile int*)c->cnt_h;
        int Rx = uni_R_.back();
        for (int r : uni_R_) if (r >= cnt) { Rx = r; break; }
        c->R = Rx;
        CK(cudaEventRecord(c->evB, det_s_));
        CK(cudaGraphLaunch(uniB_.at(Rx), det_s_));
        CK(cudaMemcpyAsync(c->stats_h, u_.stats, sizeof(TickStats), cudaMemcpyDeviceToHost, det_s_));
        live_tail(c);
        CK(cudaEventRecord(c->ev1, det_s_));
        return;
    }
    c->R = R;
    CK(cudaEventRecord(c->ev0, det_s_));
    CK(cudaMemcpyAsync(u_.desc, c->desc_h, sizeof(TickDesc), cudaMemcpyHostToDevice, det_s_));
    CK(cudaGraphLaunch(uni_graphs_.at({c->bucket, R}), det_s_));
    CK(cudaMemcpyAsync(c->stats_h, u_.stats, sizeof(TickStats), cudaMemcpyDeviceToHost, det_s_));
    live_tail(c);
    CK(cudaEventRecord(c->ev1, det_s_));
}

// ------------------------------------------------------------------------------------------------ live mode
// After a tick's graphs: K8 gathers the rows this tick delivered (boxes, ids, names, zones) into one block and a
// single copy brings them to the host. This is the only per-box data that ever leaves the GPU.
void Pipeline::live_tail(DetCtx* c) {
    if (cfg_.live.empty()) return;
    launch_gather_rows(u_.desc, c->n, tout_, u_.live, det_s_);
    CK(cudaMemcpyAsync(c->live_h, u_.live, sizeof(LiveBox) * (size_t)c->n * LIVE_ROWS * TRK, cudaMemcpyDeviceToHost, det_s_));
    if (c->dbg_h) CK(cudaMemcpyAsync(c->dbg_h, u_.dets, sizeof(FrameDets), cudaMemcpyDeviceToHost, det_s_));
    c->enroll_label = -1;
    Enroll e{};
    bool have = false;
    {
        std::lock_guard<std::mutex> g(enroll_m_);
        if (!enroll_q_.empty()) { e = enroll_q_.front(); have = true; }
    }
    if (!have) return;
    CK(cudaMemsetAsync(u_.enroll_done, 0, sizeof(int), det_s_));
    launch_enroll(u_.meta, &u_.stats->count, c->R, u_.emb, tt_, e.cam, e.tid, const_cast<__half*>(gallery_), e.label,
                  u_.enroll_done, det_s_);
    CK(cudaMemcpyAsync(c->enroll_h, u_.enroll_done, sizeof(int), cudaMemcpyDeviceToHost, det_s_));
    c->enroll_label = e.label;
}

static void put_line(const std::string& s) {
    std::fwrite(s.data(), 1, s.size(), stdout);
    std::fflush(stdout);
}

// One JSON line per delivered frame: {"cam","seq","ms","det","b":[[x1,y1,x2,y2,track,known,label,zone],...]}
void Pipeline::emit_live(DetCtx* c, double now) {
    std::string out;
    char buf[192];
    if (c->dbg_h) {
        static long ticks = 0;
        if (ticks++ % 10 == 0) {
            const FrameDets& fd = *c->dbg_h;
            const Letterbox& lb = c->desc_h->det.lb;
            int rows = 0;
            for (int s = 0; s < TRK; ++s) rows += c->live_h[s].id != 0;
            std::fprintf(stderr, "[dbg] cam %d fid %lld gap %d: detector %d faces (%d candidates), first %.0fx%.0f s=%.2f; "
                         "delivered %d; lb scale %.3f pad %.0f,%.0f; stats boxes %d\n", c->items[0].cam->src.id,
                         (long long)c->items[0].fid, c->items[0].gap, fd.count, fd.n_cand,
                         fd.count ? fd.box[0].z - fd.box[0].x : 0.f, fd.count ? fd.box[0].w - fd.box[0].y : 0.f,
                         fd.count ? fd.score[0] : 0.f, rows, lb.scale, lb.pad_x, lb.pad_y, c->stats_h->boxes);
        }
    }
    for (int k = 0; k < c->n; ++k) {
        const Item& it = c->items[k];
        const int rows = std::min(std::max(it.gap, 1), LIVE_ROWS);
        for (int r = rows - 1; r >= 0; --r) {
            const LiveBox* row = c->live_h + ((size_t)k * LIVE_ROWS + r) * TRK;
            const double t_cap = it.t_cap - r * it.cam->src.period;
            std::snprintf(buf, sizeof(buf), "{\"cam\":%d,\"seq\":%lld,\"ms\":%.1f,\"det\":%d,\"b\":[", it.cam->src.id,
                          (long long)(it.fid - r), (now - t_cap) * 1000.0, r == 0 ? 1 : 0);
            out += buf;
            bool first = true;
            for (int s = 0; s < TRK; ++s) {
                const LiveBox& b = row[s];
                if (!b.id) continue;
                std::snprintf(buf, sizeof(buf), "%s[%.1f,%.1f,%.1f,%.1f,%u,%d,%d,%d]", first ? "" : ",", b.box.x, b.box.y,
                              b.box.z, b.box.w, (unsigned)b.id >> 12, (b.id >> 11) & 1, (b.id & 0x7ff) - 1, b.zone);
                out += buf;
                first = false;
            }
            out += "]}\n";
        }
    }
    // enrolment: did this tick store the face, or has the request waited too long (the person left)?
    {
        std::lock_guard<std::mutex> g(enroll_m_);
        if (!enroll_q_.empty()) {
            const Enroll e = enroll_q_.front();
            if (c->enroll_label == e.label && *c->enroll_h == 1) {
                std::snprintf(buf, sizeof(buf), "{\"event\":\"enrolled\",\"cam\":%d,\"track\":%u,\"label\":%d}\n", e.cam, e.tid, e.label);
                out += buf;
                enroll_q_.pop_front();
            } else if (now - e.t_req > 8.0) {
                std::snprintf(buf, sizeof(buf), "{\"event\":\"enroll_failed\",\"cam\":%d,\"track\":%u,\"label\":%d}\n", e.cam, e.tid, e.label);
                out += buf;
                enroll_q_.pop_front();
            }
        }
    }
    put_line(out);
}

void Pipeline::set_zones(int cam, const std::vector<std::pair<int, std::vector<float>>>& zones) {
    if (cam < 0 || cam >= (int)cams_.size() || !cams_[cam]->zone_mask) return;
    Cam* c = cams_[cam].get();
    const int w = c->src.ring.w(), h = c->src.ring.h();
    std::vector<uint8_t> m((size_t)w * h, 0);
    std::vector<float> xs;
    for (auto& z : zones) {
        const std::vector<float>& p = z.second;
        const int np = (int)p.size() / 2;
        if (np < 3 || z.first < 1 || z.first > 255) continue;
        for (int y = 0; y < h; ++y) {             // even-odd scanline fill at pixel centres
            const float fy = (y + 0.5f) / h;
            xs.clear();
            for (int i = 0, j = np - 1; i < np; j = i++) {
                const float yi = p[2 * i + 1], yj = p[2 * j + 1];
                if ((yi > fy) != (yj > fy)) xs.push_back(p[2 * i] + (fy - yi) * (p[2 * j] - p[2 * i]) / (yj - yi));
            }
            std::sort(xs.begin(), xs.end());
            for (size_t k = 0; k + 1 < xs.size(); k += 2) {
                const int x0 = std::max(0, (int)std::ceil(xs[k] * w - 0.5f));
                const int x1 = std::min(w - 1, (int)std::floor(xs[k + 1] * w - 0.5f));
                for (int x = x0; x <= x1; ++x) m[(size_t)y * w + x] = (uint8_t)z.first;
            }
        }
    }
    CK(cudaMemcpy(c->zone_mask, m.data(), m.size(), cudaMemcpyHostToDevice));
}

void Pipeline::run_live(int n) {
    cams_.clear();
    std::vector<CamSource*> srcs;
    for (int i = 0; i < n; ++i) {
        auto cam = std::make_unique<Cam>();
        std::string path = cfg_.live;
        size_t at = path.find("{r}");
        if (at != std::string::npos) path.replace(at, 3, std::to_string(i));
        cam->src.id = i;
        cam->src.dec = std::make_unique<VideoDecoder>(path, DecodePool::hw_device(i), cfg_.live_w, cfg_.live_h, cfg_.live_fps);
        const int slots = (int)std::ceil(cfg_.buffer_s * cfg_.analyze_fps) + cfg_.fifo_inflight + 2;
        cam->src.ring.alloc(cfg_.live_w, cfg_.live_h, slots, true, true);
        cam->ratio = std::min(1.0, cfg_.analyze_fps / cfg_.live_fps);
        cam->src.ratio = cam->ratio;
        const size_t zb = (size_t)cfg_.live_w * cfg_.live_h;
        CK(cudaMalloc(&cam->zone_mask, zb));
        CK(cudaMemset(cam->zone_mask, 0, zb));
        srcs.push_back(&cam->src);
        cams_.push_back(std::move(cam));
    }
    det_free_.clear(); det_done_.clear(); rec_free_.clear(); rec_done_.clear();
    for (auto& c : det_ctx_) det_free_.push_back(c.get());
    for (auto& r : rec_ctx_) rec_free_.push_back(r.get());
    backlog_ = 0; det_inflight_ = 0; draining_ = false;
    DecodePool pool(srcs, n, this);       // one decoder thread per live camera
    reset_gpu_track();
    req_per_frame_ema_ = 3.0;
    denied_ema_ = 0.0;
    t0_run_ = now_s();
    run_ = true;
    pool.start(now_s());
    std::thread ts(&Pipeline::sched_loop, this), tt(&Pipeline::track_loop, this), tf(&Pipeline::finish_loop, this);
    {
        char buf[160];
        std::snprintf(buf, sizeof(buf), "{\"event\":\"ready\",\"cams\":%d,\"w\":%d,\"h\":%d,\"fps\":%.0f,\"detect_fps\":%.0f}\n", n,
                      cfg_.live_w, cfg_.live_h, cfg_.live_fps, cfg_.analyze_fps);
        put_line(buf);
    }
    // control lines on stdin:
    //   Z <cam> <zones> { <label> <points> x y x y ... }     replace the camera's zones (coordinates 0..1)
    //   E <cam> <track> <label>                              store this track's face as gallery row <label>
    //   Q                                                    stop
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream is(line);
        std::string cmd;
        is >> cmd;
        if (cmd == "Q") break;
        if (cmd == "Z") {
            int cam = -1, nz = 0;
            is >> cam >> nz;
            std::vector<std::pair<int, std::vector<float>>> zones;
            for (int z = 0; z < nz && is; ++z) {
                int label = 0, np = 0;
                is >> label >> np;
                std::vector<float> p((size_t)std::max(0, std::min(np, 256)) * 2);
                for (auto& v : p) is >> v;
                if (is) zones.emplace_back(label, std::move(p));
            }
            set_zones(cam, zones);
            char buf[96];
            std::snprintf(buf, sizeof(buf), "{\"event\":\"zones\",\"cam\":%d,\"count\":%zu}\n", cam, zones.size());
            put_line(buf);
        } else if (cmd == "E") {
            int cam = -1, label = -1;
            unsigned tid = 0;
            is >> cam >> tid >> label;
            if (is && cam >= 0 && cam < n && label >= 0 && label < G_) {
                std::lock_guard<std::mutex> g(enroll_m_);
                enroll_q_.push_back(Enroll{cam, tid, label, now_s()});
            }
        }
    }
    run_ = false;
    sched_cv_.notify_all(); q_cv_.notify_all(); det_free_cv_.notify_all(); rec_done_cv_.notify_all();
    ts.join(); tt.join(); tf.join();
    pool.stop();
    CK(cudaDeviceSynchronize());
    for (auto& c : cams_) { c->src.ring.free_all(); if (c->zone_mask) cudaFree(c->zone_mask); }
    cams_.clear();
}

void Pipeline::submit_rec_gpu(RecCtx* r) {
    Nvtx nv("rec_submit_gpu");
    const int pool = cfg_.pool;
    const int start = (int)(pool_tail_ % (unsigned long long)pool);
    int n = (int)std::min<unsigned long long>({(unsigned long long)std::min(cfg_.rec_batch, rec_.max_batch()),
                                                    pool_head_ - pool_tail_, (unsigned long long)(pool - start)});
    if (cfg_.cb && now_s() - pend_since_ < cfg_.rec_wait_ms / 1000.0) n = rec_.bucket_floor(n);   // no padding
    r->n = n;
    r->bucket = rec_.bucket_for(n);
    r->pool_start = pool_tail_;
    r->jobs.clear();
    pool_tail_ += n;
    pend_since_ = now_s();   // anything still waiting was queued no earlier than this batch was sent
    CK(cudaEventRecord(r->ev0, rec_s_));
    // the recogniser reads its input straight out of the crop pool: no gather, no copy
    rec_.run_direct(r->bucket, pool_tensor_ + (size_t)start * rec_.in_item_elems(), r->emb, rec_s_);
    launch_match(r->emb, n, gallery_, G_, r->match.d, rec_s_);
    launch_scatter_identity(r->match.d, n, start, cp_, tt_, cfg_.match_thr, nullptr, rec_s_);
    CK(cudaEventRecord(r->ev1, rec_s_));
    {
        std::lock_guard<std::mutex> g(q_m_);
        rec_done_.push_back(r);
    }
    rec_done_cv_.notify_one();
}

// ------------------------------------------------------------------------------------------------ stage A
void Pipeline::sched_loop() {
    name_thread("sched");
    const int B = std::min(cfg_.det_batch, det_.max_batch());
    while (run_) {
        DetCtx* c = nullptr;
        {
            std::unique_lock<std::mutex> lk(q_m_);
            det_free_cv_.wait_for(lk, std::chrono::milliseconds(20), [&] { return !det_free_.empty() || !run_; });
            if (det_free_.empty()) continue;
            c = det_free_.front();
            det_free_.pop_front();
        }
        int n = 0;
        {
            std::unique_lock<std::mutex> lk(m_);
            const bool fifo = cfg_.policy == "fifo";
            for (int attempt = 0; attempt < 2 && n == 0 && run_; ++attempt) {
                const int N = (int)cams_.size();
                int target = B;
                // 1-second window: let the oldest keyframe age to window_s, then drain everything that is ready in
                // back-to-back full batches (one burst per window)
                if (cfg_.window_s > 0) {
                    int ready = 0;
                    double oldest = 1e300;
                    for (int k = 0; k < N; ++k) {
                        Cam* cam = cams_[k].get();
                        if (cam->inflight >= cfg_.fifo_inflight) continue;
                        double tc;
                        if (cam->src.ring.peek(cam->last, tc) < 0) continue;
                        ++ready;
                        oldest = std::min(oldest, tc);
                    }
                    if (ready == 0) { draining_ = false; sched_cv_.wait_for(lk, std::chrono::milliseconds(2)); continue; }
                    if (!draining_ && now_s() - oldest < cfg_.window_s) { sched_cv_.wait_for(lk, std::chrono::milliseconds(2)); continue; }
                    draining_ = true;
                }
                // batching window: while a detector batch is still running, wait (up to max_wait_ms for the oldest
                // ready frame) until min_batch frames are ready, so the GPU runs fewer, larger batches
                if (cfg_.window_s > 0) {
                    // (handled above)
                } else if (cfg_.cb) {
                    // continuous batching for the detector: launch at once if the detector is idle; otherwise wait
                    // for min_batch frames or until the oldest has waited max_wait_ms; cut the batch to an exact
                    // engine bucket unless something is late (no padding work)
                    int ready = 0;
                    double oldest = 1e300;
                    for (int k = 0; k < N && ready < B; ++k) {
                        Cam* cam = cams_[k].get();
                        if (cam->inflight >= (fifo ? cfg_.fifo_inflight : cfg_.inflight) || (!fifo && cam->in_det)) continue;
                        double tc;
                        if (cam->src.ring.peek(cam->last, tc) < 0) continue;
                        ++ready;
                        oldest = std::min(oldest, tc);
                    }
                    if (ready == 0) { sched_cv_.wait_for(lk, std::chrono::microseconds(500)); continue; }
                    const bool idle = cfg_.cb_idle && det_inflight_.load() == 0;
                    const bool late = now_s() - oldest >= cfg_.max_wait_ms / 1000.0;
                    if (!idle && !late && ready < cfg_.min_batch) { sched_cv_.wait_for(lk, std::chrono::microseconds(500)); continue; }
                    // unified pipeline: always cut to an exact engine size (the leftover frames join the next tick)
                    target = ready >= B ? B : (!cfg_.unified && (late || idle)) ? ready : det_.bucket_floor(ready);
                } else if (det_inflight_.load() > 0 && cfg_.min_batch > 1) {
                    int ready = 0;
                    double oldest = 1e300;
                    for (int k = 0; k < N && ready < cfg_.min_batch; ++k) {
                        Cam* cam = cams_[k].get();
                        if (cam->inflight >= (fifo ? cfg_.fifo_inflight : cfg_.inflight) || (!fifo && cam->in_det)) continue;
                        double tc;
                        if (cam->src.ring.peek(cam->last, tc) < 0) continue;
                        ++ready;
                        oldest = std::min(oldest, tc);
                    }
                    if (ready < cfg_.min_batch && (ready == 0 || now_s() - oldest < cfg_.max_wait_ms / 1000.0)) {
                        sched_cv_.wait_for(lk, std::chrono::microseconds(500));
                        continue;
                    }
                }
                const double t_read = now_s();
                if (fifo) {
                    // every buffered frame, in order; round-robin passes keep cameras fair inside one batch
                    for (int pass = 0; pass < (cfg_.gpu_track ? 1 : B) && n < std::min(B, target); ++pass) {
                        int took = 0;
                        for (int k = 0; k < N && n < std::min(B, target); ++k) {
                            Cam* cam = cams_[(rr_ + k) % N].get();
                            if (cam->inflight >= cfg_.fifo_inflight) continue;
                            int64_t fid;
                            double t_arr, t_cap;
                            int slot = cam->src.ring.acquire_next(cam->last, fid, t_arr, t_cap);
                            if (slot < 0) continue;
                            cam->last = fid;
                            cam->in_det = true;
                            cam->inflight++;
                            c->items[n] = Item{cam, slot, fid, t_arr, t_read, 1, t_cap};
                            c->items[n].gap = cam->last_key >= 0 ? (int)(fid - cam->last_key) : 1;
                            if (cfg_.gpu_track) cam->last_key = fid;
                            ++n;
                            ++took;
                        }
                        if (!took) break;
                    }
                } else {
                    for (int k = 0; k < N && n < B; ++k) {
                        Cam* cam = cams_[(rr_ + k) % N].get();
                        if (cam->in_det || cam->inflight >= cfg_.inflight) continue;
                        double tc;
                        int64_t fid = cam->src.ring.peek(cam->last, tc);
                        if (fid < 0) continue;
                        int due = cam->last >= 0 ? (int)(std::floor(fid * cam->ratio) - std::floor(cam->last * cam->ratio)) : 1;
                        if (due <= 0) { cam->last = fid; continue; }
                        double t_arr, t_cap;
                        int slot = cam->src.ring.acquire_latest(cam->last, fid, t_arr, t_cap);
                        if (slot < 0) continue;
                        due = cam->last >= 0 ? (int)(std::floor(fid * cam->ratio) - std::floor(cam->last * cam->ratio)) : 1;
                        cam->last = fid;
                        cam->in_det = true;
                        cam->inflight++;
                        c->items[n++] = Item{cam, slot, fid, t_arr, t_read, std::max(due, 1), t_cap};
                    }
                }
                rr_ = N ? (rr_ + std::max(n, 1)) % N : 0;
                if (n == 0) sched_cv_.wait_for(lk, std::chrono::milliseconds(2));
            }
        }
        if (n == 0) {
            std::lock_guard<std::mutex> g(q_m_);
            det_free_.push_front(c);
            continue;
        }
        Nvtx r("det_submit");
        c->n = n;
        c->bucket = det_.bucket_for(n);
        DetBatch b{};
        b.n = n;
        for (int i = 0; i < n; ++i) {
            FrameRing& ring = c->items[i].cam->src.ring;
            b.f[i] = FrameRef{ring.y(c->items[i].slot), ring.uv(c->items[i].slot), ring.pitch(c->items[i].slot), ring.w(), ring.h(),
                              c->items[i].cam->src.dec->matrix()};
        }
        const int fw = b.f[0].w, fh = b.f[0].h;
        const float scale = std::min((float)DET_W / fw, (float)DET_H / fh) * cfg_.det_zoom;
        b.lb = Letterbox{scale, (DET_W - fw * scale) * 0.5f, (DET_H - fh * scale) * 0.5f, fw, fh};
        if (cfg_.unified) {
            ZoneMasks zmu{};
            zmu.pitch = fw;
            for (int i = 0; i < n; ++i) zmu.m[i] = c->items[i].cam->zone_mask;
            submit_tick(c, b, zmu);
        } else {
        CK(cudaEventRecord(c->ev0, det_s_));
        launch_prep_det(b, c->in, det_s_);
        det_.run(c->bucket, c->in, c->out, det_s_);
        ZoneMasks zm{};
        zm.pitch = fw;
        for (int i = 0; i < n; ++i) zm.m[i] = c->items[i].cam->zone_mask;
        launch_decode_nms(c->out, n, (int)(det_.out_item_elems() / 5), cfg_.det_chw, cfg_.conf, cfg_.iou, b.lb, zm,
                          c->dets.d, det_s_);
        if (cfg_.gpu_track) {
            TrackBatch tb{};
            tb.n = n;
            for (int i = 0; i < n; ++i) {
                const Item& it = c->items[i];
                tb.it[i] = TrackItem{it.cam->src.id, (int)it.fid, it.gap, (float)(it.t_cap - t0_run_),
                                     (float)it.cam->src.period, b.f[i]};
            }
            CropPool cp = cp_;
            cp.req = c->creq;
            cp.req_slot = c->creq_slot;
            cp.batch_count = c->ccount;
            CK(cudaMemsetAsync(c->ccount, 0, sizeof(int), det_s_));
            launch_track_update(c->dets.d, tb, tt_, tp_, cp, tout_, det_s_);
            launch_crop_to_pool(cp, n * MAX_FACES, pool_tensor_, det_s_);
            launch_publish_count(c->ccount, c->ccount_host, det_s_);
        } else {
            c->dets.to_host(n, det_s_);
        }
        CK(cudaEventRecord(c->ev1, det_s_));
        }
        det_inflight_++;
        {
            std::lock_guard<std::mutex> g(q_m_);
            det_done_.push_back(c);
        }
        q_cv_.notify_one();
    }
}

// ------------------------------------------------------------------------------------------------ stage T
void Pipeline::process_det(DetCtx* c) {
    {
        Nvtx r("det_wait");
        CK(cudaEventSynchronize(c->ev1));
    }
    Nvtx r("track");
    const double t_cpu0 = now_s();
    float ms = 0;
    cudaEventElapsedTime(&ms, c->ev0, c->ev1);
    if (recording_) {
        std::lock_guard<std::mutex> g(stat_m_);
        det_calls_.emplace_back(c->n, ms);
    }
    std::vector<Tracker::Res> res;
    std::vector<Item> done_now;
    std::vector<int> done_faces;
    if (cfg_.gpu_track) {
        const double now = now_s();
        // end-to-end accounting per camera and per second of video: when was the last result of that second ready
        auto chunk = [&](Cam* cam, double t_rel, bool frame) {
            if (t_rel < 0) return;
            const size_t sec = (size_t)t_rel;
            if (cam->chunk_done.size() <= sec) { cam->chunk_done.resize(sec + 1, 0.0); cam->chunk_frames.resize(sec + 1, 0); }
            cam->chunk_done[sec] = std::max(cam->chunk_done[sec], now);
            if (frame) cam->chunk_frames[sec]++;
        };
        if (cfg_.unified) {
            // the tick did everything, names included; the CPU reads its statistics to size the next ticks
            const TickStats& st = *c->stats_h;
            const int used = std::min(st.count, c->R);
            for (int k = 0; k < used; ++k)       // a name belongs to the second in which its face first asked for it
                if (st.rcam[k] >= 0 && st.rcam[k] < (int)cams_.size()) chunk(cams_[st.rcam[k]].get(), st.treq[k], false);
            req_per_frame_ema_ = 0.8 * req_per_frame_ema_.load() + 0.2 * (double)st.cands / std::max(1, c->n);
            denied_ema_ = 0.7 * denied_ema_.load() + 0.3 * st.denied;
            if (recording_) {
                std::lock_guard<std::mutex> g(stat_m_);
                rec_calls_.emplace_back(used, 0.f);
                for (int k = 0; k < used; ++k)
                    if (st.tfirst[k] >= 0.f) first_id_.push_back(now - (t0_run_ + st.tfirst[k]));
                tick_ms_sum_ += ms;
                if (cfg_.uni_split) {
                    float a = 0, gp = 0, bm = 0;
                    cudaEventElapsedTime(&a, c->ev0, c->evA);
                    cudaEventElapsedTime(&gp, c->evA, c->evB);
                    cudaEventElapsedTime(&bm, c->evB, c->ev1);
                    splitA_ms_ += a; split_gap_ms_ += gp; splitB_ms_ += bm;
                }
                ++ticks_;
                uni_slots_ += c->R;
                uni_used_ += used;
                uni_denied_ += st.denied;
                uni_boxes_ += st.boxes;
                uni_named_ += st.named;
                name_wait_max_ms_ = std::max(name_wait_max_ms_, st.max_pend_ms);
                for (int k = 0; k < used; ++k) name_delay_.push_back(now - (t0_run_ + st.treq[k]));
            }
            if (!cfg_.live.empty()) emit_live(c, now);
        } else {
            // everything happened on the GPU; the CPU only learns how many crops this batch queued
            const int queued = *c->ccount_host;
            if (queued > 0 && pool_head_ == pool_tail_) pend_since_ = now_s();
            pool_head_ += (unsigned long long)queued;
        }
        for (int k = 0; k < c->n; ++k) {
            const Item& it = c->items[k];
            if (cfg_.unified)
                for (int j = 1; j <= it.gap; ++j) chunk(it.cam, it.t_cap - (it.gap - j) * it.cam->src.period - t0_run_, true);
            if (recording_) {
                for (int j = 1; j <= it.gap; ++j) out_delay_.push_back((float)(now - (it.t_cap - (it.gap - j) * it.cam->src.period)));
                it.cam->out_frames += it.gap;
                it.cam->filled_frames += it.gap - 1;
            }
            done_now.push_back(it);
            done_faces.push_back(0);
        }
    }
    for (int k = 0; k < (cfg_.gpu_track ? 0 : c->n); ++k) {
        const Item& it = c->items[k];
        const FrameDets& fd = c->dets.h[k];
        const int nf = std::min(fd.count, cfg_.max_faces);
        Cam* cam = it.cam;
        for (int i = 0; i < nf; ++i) cam->zone_count[fd.zone[i] & 7]++;   // zone presence (headcount per zone)
        Job job{it, nf, (int)pend_req_.size(), 0, {}};
        auto add_crop = [&](int i, double t_first) {
            const FrameRing& ring = cam->src.ring;
            const float4 b = fd.box[i];
            pend_req_.push_back(CropReq{ring.y(it.slot), ring.uv(it.slot), ring.pitch(it.slot), ring.w(), ring.h(), cam->src.dec->matrix(),
                                        b.x, b.y, b.z, b.w});
            job.t_first.push_back(t_first);
            job.count++;
        };
        if (cfg_.bestshot) {
            // every frame is tracked; recognition is deferred to the end of the window, from each face's best frame
            const double W = cfg_.reverify;
            const int64_t w = (int64_t)std::floor(it.t_cap / W);
            if (cam->cur_w < 0) cam->cur_w = w;
            if (w > cam->cur_w) { flush_best(cam, (cam->cur_w + 1) * W); cam->cur_w = w; }
            // gap = 1 when every frame is detected; gap = 3 when only every 3rd frame is (Kalman fills the rest)
            const int gap = cam->last_key >= 0 ? (int)(it.fid - cam->last_key) : 1;
            fills_.clear();
            cam->ktracker.update(fd.box, nf, it.t_cap, gap, (float)cam->src.period, res, &fills_);
            cam->last_key = it.fid;
            if (recording_) {
                const double now = now_s();
                for (int j = 1; j <= gap; ++j) out_delay_.push_back((float)(now - (it.t_cap - (gap - j) * cam->src.period)));
                cam->out_frames += gap;
                cam->filled_frames += gap - 1;
            }
            for (int i = 0; i < nf; ++i) {
                const float4 b = fd.box[i];
                if (cfg_.min_rec_px > 0 && b.z - b.x < cfg_.min_rec_px) continue;
                if (!(res[i].is_new || res[i].is_due)) continue;
                const float q = (b.z - b.x) * (b.w - b.y) * fd.score[i];     // face size x detector confidence
                auto f = cam->best.find(res[i].id);
                if (f == cam->best.end() || q > f->second.q) {
                    cam->src.ring.addref(it.slot);
                    if (f != cam->best.end()) cam->src.ring.release(f->second.slot);
                    cam->best[res[i].id] = Cam::Best{q, it.slot, b, res[i].is_new ? res[i].first : -1.0, it.t_cap, res[i].prev_rec};
                }
            }
            cam->ktracker.prune(it.t_cap);
        } else if (!cfg_.track) {
            for (int i = 0; i < nf; ++i) add_crop(i, -1);
        } else {
            if (cfg_.kalman) {
                const int gap = cam->last_key >= 0 ? (int)(it.fid - cam->last_key) : 1;
                fills_.clear();
                cam->ktracker.update(fd.box, nf, it.t_cap, gap, (float)cam->src.period, res, &fills_);
                // frames (last_key, fid] now have boxes for every track: the keyframe detected, the rest smoothed
                const double now = now_s();
                if (recording_) {
                    for (int j = 1; j <= gap; ++j) out_delay_.push_back((float)(now - (it.t_cap - (gap - j) * cam->src.period)));
                    cam->out_frames += gap;
                    cam->filled_frames += gap - 1;
                }
                cam->last_key = it.fid;
            } else {
                cam->tracker.update(fd.box, nf, it.t_cap, res);   // frame time, not processing time
            }
            const int backlog = backlog_.load();
            const bool defer_new = cfg_.budget && backlog >= 2 * cfg_.rec_batch;
            const bool defer_due = cfg_.budget && backlog >= cfg_.rec_batch;
            for (int i = 0; i < nf; ++i) {
                const float4 b = fd.box[i];
                if (cfg_.min_rec_px > 0 && b.z - b.x < cfg_.min_rec_px) continue;
                if (res[i].is_new && !defer_new) {
                    if (cfg_.kalman) cam->ktracker.mark(res[i].track, it.t_cap); else cam->tracker.mark(res[i].track, it.t_cap);
                    add_crop(i, res[i].first);
                } else if (res[i].is_due) {
                    if (defer_due) { deferred_++; continue; }
                    if (recording_) { std::lock_guard<std::mutex> g(stat_m_); refresh_.push_back(it.t_cap - res[i].prev_rec); }
                    if (cfg_.kalman) cam->ktracker.mark(res[i].track, it.t_cap); else cam->tracker.mark(res[i].track, it.t_cap);
                    add_crop(i, -1);
                }
            }
            if (cfg_.kalman) cam->ktracker.prune(it.t_cap); else cam->tracker.prune(it.t_cap);
        }
        if (job.count) {
            backlog_ += job.count;
            pend_jobs_.push_back(std::move(job));
        } else {
            done_now.push_back(it);
            done_faces.push_back(nf);
        }
    }
    {
        std::lock_guard<std::mutex> g(m_);
        for (int k = 0; k < c->n; ++k) c->items[k].cam->in_det = false;
    }
    {
        std::lock_guard<std::mutex> g(q_m_);
        det_free_.push_back(c);
    }
    det_inflight_--;
    det_free_cv_.notify_one();
    const double t_done = now_s();
    for (size_t i = 0; i < done_now.size(); ++i) finish_item(done_now[i], 0, done_faces[i], t_done);
    if (recording_) track_cpu_s_ += now_s() - t_cpu0;
    sched_cv_.notify_one();
}

void Pipeline::flush_best(Cam* cam, double t_window_end) {
    if (cam->best.empty()) return;
    Job job{};
    job.frame_job = false;
    job.cam = cam;
    job.first = (int)pend_req_.size();
    job.t_window_end = t_window_end;
    const FrameRing& ring = cam->src.ring;
    for (auto& kv : cam->best) {
        const Cam::Best& b = kv.second;
        pend_req_.push_back(CropReq{ring.y(b.slot), ring.uv(b.slot), ring.pitch(b.slot), ring.w(), ring.h(),
                                    cam->src.dec->matrix(), b.box.x, b.box.y, b.box.z, b.box.w});
        job.t_first.push_back(b.t_first);
        job.pins.push_back(b.slot);
        job.count++;
        for (int j = 0; j < (int)cam->ktracker.tr.size(); ++j)
            if (cam->ktracker.tr[j].id == kv.first) {
                if (recording_ && b.prev_rec > -1e8) { std::lock_guard<std::mutex> g(stat_m_); refresh_.push_back(b.t_cap - b.prev_rec); }
                cam->ktracker.mark(j, b.t_cap);
                break;
            }
    }
    cam->best.clear();
    backlog_ += job.count;
    pend_jobs_.push_back(std::move(job));
}

// ------------------------------------------------------------------------------------------------ stage B
void Pipeline::submit_rec(RecCtx* r) {
    Nvtx nv("rec_submit");
    const int RB = std::min(cfg_.rec_batch, rec_.max_batch());
    r->jobs.clear();
    int n = 0;
    size_t used = 0;
    while (used < pend_jobs_.size() && n + pend_jobs_[used].count <= RB) {
        Job j = std::move(pend_jobs_[used]);
        std::memcpy(r->req.h + n, pend_req_.data() + j.first, sizeof(CropReq) * j.count);
        j.first = n;
        n += j.count;
        r->jobs.push_back(std::move(j));
        ++used;
    }
    // compact the pending lists
    size_t req_used = 0;
    for (auto& j : r->jobs) req_used += j.count;
    pend_jobs_.erase(pend_jobs_.begin(), pend_jobs_.begin() + used);
    pend_req_.erase(pend_req_.begin(), pend_req_.begin() + req_used);
    int off = 0;
    for (auto& j : pend_jobs_) { j.first = off; off += j.count; }
    std::atomic_thread_fence(std::memory_order_seq_cst);   // flush write-combined stores before the launch
    r->n = n;
    r->bucket = rec_.bucket_for(n);
    CK(cudaEventRecord(r->ev0, rec_s_));
    r->req.to_device(n, rec_s_);
    launch_crop_prep(r->req.d, n, r->in, rec_s_);
    rec_.run(r->bucket, r->in, r->emb, rec_s_);
    launch_match(r->emb, n, gallery_, G_, r->match.d, rec_s_);
    r->match.to_host(n, rec_s_);
    CK(cudaEventRecord(r->ev1, rec_s_));
    {
        std::lock_guard<std::mutex> g(q_m_);
        rec_done_.push_back(r);
    }
    rec_done_cv_.notify_one();
}

void Pipeline::track_loop() {
    name_thread("track");
    while (run_) {
        DetCtx* c = nullptr;
        RecCtx* r = nullptr;
        {
            std::unique_lock<std::mutex> lk(q_m_);
            const bool gpu = cfg_.gpu_track;
            auto rec_ready = [&] {
                if (!gpu) return !pend_jobs_.empty();
                const unsigned long long pend = pool_head_ - pool_tail_;
                return pend >= (unsigned long long)cfg_.rec_min ||
                       (pend > 0 && now_s() - pend_since_ >= cfg_.rec_wait_ms / 1000.0);
            };
            q_cv_.wait_for(lk, std::chrono::milliseconds(5), [&] {
                return !run_ || !det_done_.empty() || (rec_ready() && !rec_free_.empty());
            });
            if (!det_done_.empty()) {
                c = det_done_.front();
                det_done_.pop_front();
            } else if (rec_ready() && !rec_free_.empty()) {
                r = rec_free_.front();
                rec_free_.pop_front();
            }
        }
        if (c) process_det(c);
        else if (r) { if (cfg_.gpu_track) submit_rec_gpu(r); else submit_rec(r); }
    }
}

// ------------------------------------------------------------------------------------------------ stage F
void Pipeline::finish_item(const Item& it, int n_rec, int n_faces, double t_done) {
    it.cam->src.ring.release(it.slot);
    std::lock_guard<std::mutex> g(m_);
    it.cam->inflight--;
    if (recording_) {
        it.cam->eligible += it.due;
        it.cam->skipped += it.due - 1;
        it.cam->rows.push_back(Row{it.cam->src.id, it.fid, it.t_arr, it.t_read, t_done, n_rec, n_faces, it.t_cap});
    }
}

void Pipeline::finish_loop() {
    name_thread("finish");
    while (run_) {
        RecCtx* r = nullptr;
        {
            std::unique_lock<std::mutex> lk(q_m_);
            rec_done_cv_.wait_for(lk, std::chrono::milliseconds(20), [&] { return !rec_done_.empty() || !run_; });
            if (rec_done_.empty()) continue;
            r = rec_done_.front();
            rec_done_.pop_front();
        }
        {
            Nvtx nv("rec_wait");
            CK(cudaEventSynchronize(r->ev1));
        }
        const double t_done = now_s();
        float ms = 0;
        cudaEventElapsedTime(&ms, r->ev0, r->ev1);
        if (recording_) {
            std::lock_guard<std::mutex> g(stat_m_);
            rec_calls_.emplace_back(r->n, ms);
            for (auto& j : r->jobs)
                for (double tf : j.t_first) if (tf >= 0) first_id_.push_back(t_done - tf);
        }
        if (cfg_.gpu_track) {
            // pool slots are free again; the tracker kernels read this to avoid overwriting unread crops
            *(volatile unsigned long long*)released_host_ = r->pool_start + r->n;
            if (recording_) {
                std::lock_guard<std::mutex> g(stat_m_);
                for (int k = 0; k < r->n; ++k) {
                    const float tf = cp_.tfirst_host[(r->pool_start + k) % (unsigned long long)cfg_.pool];
                    if (tf >= 0.f) first_id_.push_back(t_done - (t0_run_ + tf));
                }
            }
        }
        // r->match.h[...] holds (gallery index, cosine) per face: identities / unknown-person events go here
        for (auto& j : r->jobs) {
            if (j.frame_job) { finish_item(j.item, j.count, j.n_faces, t_done); continue; }
            for (int slot : j.pins) j.cam->src.ring.release(slot);
            if (recording_) { std::lock_guard<std::mutex> g(stat_m_); id_after_window_.push_back(t_done - j.t_window_end); }
        }
        if (!cfg_.gpu_track) backlog_ -= r->n;
        {
            std::lock_guard<std::mutex> g(q_m_);
            rec_free_.push_back(r);
        }
        q_cv_.notify_one();
        sched_cv_.notify_one();
    }
}

void Pipeline::sample_gpu() {
    FILE* p = popen("nvidia-smi --query-gpu=utilization.gpu,utilization.decoder,memory.used,clocks.sm,power.draw "
                    "--format=csv,noheader,nounits", "r");
    if (!p) return;
    std::array<double, 5> v{};
    char line[256] = {};
    if (fgets(line, sizeof(line), p)) {
        std::sscanf(line, "%lf, %lf, %lf, %lf, %lf", &v[0], &v[1], &v[2], &v[3], &v[4]);
        gpu_samples_.push_back(v);
    }
    pclose(p);
}

// ------------------------------------------------------------------------------------------------ one run
struct JNum { double v; };
static std::ostream& operator<<(std::ostream& o, JNum x) {
    if (std::isfinite(x.v)) o << x.v; else o << "null";
    return o;
}
static double pct(std::vector<double> v, double p) {
    if (v.empty()) return NAN;
    std::sort(v.begin(), v.end());
    double idx = p / 100.0 * (v.size() - 1);
    size_t lo = (size_t)idx, hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (v[hi] - v[lo]) * (idx - lo);
}

std::string Pipeline::run_count(int n) {
    // cameras: each decodes its own copy of one of the 4 rotated clips
    cams_.clear();
    std::vector<CamSource*> srcs;
    for (int i = 0; i < n; ++i) {
        auto cam = std::make_unique<Cam>();
        std::string path = cfg_.video;
        size_t at = path.find("{r}");
        if (at != std::string::npos) path.replace(at, 3, std::to_string(i % 4));
        cam->src.id = i;
        cam->src.dec = std::make_unique<VideoDecoder>(path, DecodePool::hw_device(i % std::max(1, std::min(cfg_.decoders, n))));
        const bool fifo = cfg_.policy == "fifo";
        const int slots = fifo ? (int)std::ceil(cfg_.buffer_s * cfg_.analyze_fps) + cfg_.fifo_inflight + 2 : 4;
        const int fw = cam->src.dec->width(), fh = cam->src.dec->height();
        cam->src.ring.alloc(fw, fh, slots, fifo, cfg_.zero_copy);
        cam->ratio = std::min(1.0, cfg_.analyze_fps / cam->src.dec->fps());
        cam->src.ratio = cam->ratio;
        if (cfg_.zones > 0) {       // N vertical sections, ids 1..N
            std::vector<uint8_t> m((size_t)fw * fh);
            for (int y = 0; y < fh; ++y)
                for (int x = 0; x < fw; ++x) m[(size_t)y * fw + x] = (uint8_t)(1 + x * cfg_.zones / fw);
            CK(cudaMalloc(&cam->zone_mask, m.size()));
            CK(cudaMemcpy(cam->zone_mask, m.data(), m.size(), cudaMemcpyHostToDevice));
        }
        cam->tracker.reverify = cfg_.reverify;
        cam->ktracker.reverify = cfg_.reverify;
        cam->ktracker.dist_k = cfg_.dist_k;
        cam->ktracker.sigma_acc = cfg_.kf_acc;
        cam->ktracker.sigma_meas = cfg_.kf_meas;
        cam->tracker.dist_k = cfg_.dist_k;
        srcs.push_back(&cam->src);
        cams_.push_back(std::move(cam));
    }
    det_free_.clear(); det_done_.clear(); rec_free_.clear(); rec_done_.clear();
    for (auto& c : det_ctx_) det_free_.push_back(c.get());
    for (auto& r : rec_ctx_) rec_free_.push_back(r.get());
    pend_req_.clear(); pend_jobs_.clear(); backlog_ = 0; deferred_ = 0; det_inflight_ = 0;
    det_calls_.clear(); rec_calls_.clear(); first_id_.clear(); refresh_.clear(); gpu_samples_.clear(); out_delay_.clear(); draining_ = false; id_after_window_.clear();

    DecodePool pool(srcs, cfg_.decoders, this);
    if (cfg_.gpu_track) reset_gpu_track();
    req_per_frame_ema_ = 3.0;
    denied_ema_ = 0.0;
    t0_run_ = now_s();
    run_ = true;
    pool.start(now_s() + 0.5);
    std::thread ts(&Pipeline::sched_loop, this), tt(&Pipeline::track_loop, this), tf(&Pipeline::finish_loop, this);
    std::this_thread::sleep_for(std::chrono::duration<double>(cfg_.warmup));
    {
        std::lock_guard<std::mutex> g(stat_m_);
        det_calls_.clear(); rec_calls_.clear(); first_id_.clear(); refresh_.clear(); id_after_window_.clear();
    }
    out_delay_.clear();
    {
        std::lock_guard<std::mutex> g(stat_m_);
        tick_ms_sum_ = 0;
        splitA_ms_ = splitB_ms_ = split_gap_ms_ = 0;
        ticks_ = uni_slots_ = uni_used_ = uni_denied_ = uni_boxes_ = uni_named_ = 0;
        name_wait_max_ms_ = 0;
        name_delay_.clear();
    }
    for (auto& c : cams_) { c->out_frames = 0; c->filled_frames = 0; }
    deferred_ = 0;
    track_cpu_s_ = 0;
    int64_t arr0 = 0, ovf0 = 0;
    for (auto& c : cams_) { arr0 += c->src.ring.arrived.load(); ovf0 += c->src.ring.overflow.load(); }
    auto gpu_counters = [&](unsigned long long* o) {
        if (!cfg_.gpu_track) { o[4] = 0; for (auto& c : cams_) o[4] += c->ktracker.next_id - 1; return; }
        std::vector<unsigned long long> fr(n), fa(n), rq(n);
        CK(cudaMemcpy(fr.data(), tout_.frames, n * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(fa.data(), tout_.faces, n * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(rq.data(), tout_.requests, n * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        o[0] = o[1] = o[2] = 0;
        for (int i = 0; i < n; ++i) { o[0] += fr[i]; o[1] += fa[i]; o[2] += rq[i]; }
        o[3] = 0;
        for (auto& c : cams_) o[3] += (unsigned long long)std::max<int64_t>(0, c->last_key + 1);   // keyframes ~ analysed frames
        std::vector<unsigned> nt(n);
        CK(cudaMemcpy(nt.data(), tt_.next_tid, n * sizeof(unsigned), cudaMemcpyDeviceToHost));
        o[4] = 0;
        for (auto v : nt) o[4] += v;
        CK(cudaMemcpy(o + 5, tout_.reasons, 5 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(o + 10, tout_.missed, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    };
    gpu_counters(gpu_c0_);
    nvtxRangePushA("measure");
    recording_ = true;
    const double t0 = now_s();
    while (now_s() - t0 < cfg_.window) {
        sample_gpu();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    recording_ = false;
    gpu_counters(gpu_c1_);
    int64_t arr1 = 0, ovf1 = 0, queued_end = 0;
    for (auto& c : cams_) { arr1 += c->src.ring.arrived.load(); ovf1 += c->src.ring.overflow.load(); queued_end += c->src.ring.queued(); }
    nvtxRangePop();
    run_ = false;
    sched_cv_.notify_all(); q_cv_.notify_all(); det_free_cv_.notify_all(); rec_done_cv_.notify_all();
    ts.join(); tt.join(); tf.join();
    pool.stop();
    CK(cudaDeviceSynchronize());

    // ---- summary (same keys as the Python harnesses)
    std::vector<double> wait, proc, e2e, fps_cam, cap, dlag;
    double nrec = 0, nfaces = 0;
    int64_t elig = 0, skipped = 0, rows = 0, dropped = 0;
    for (auto& c : cams_) {
        for (auto& r : c->rows) {
            wait.push_back((r.t_read - r.t_arr) * 1000);
            proc.push_back((r.t_done - r.t_read) * 1000);
            e2e.push_back((r.t_done - r.t_arr) * 1000);
            cap.push_back((r.t_done - r.t_cap) * 1000);     // camera -> result, includes decode
            dlag.push_back((r.t_arr - r.t_cap) * 1000);
            nrec += r.n_rec;
            nfaces += r.n_faces;
        }
        rows += c->rows.size();
        fps_cam.push_back(c->rows.size() / cfg_.window);
        elig += c->eligible;
        skipped += c->skipped;
        dropped += c->src.ring.dropped.load();
    }
    auto mean_calls = [](const std::vector<std::pair<int, float>>& v, double& b, double& ms, double& per) {
        double sb = 0, sm = 0;
        for (auto& x : v) { sb += x.first; sm += x.second; }
        b = v.empty() ? 0 : sb / v.size();
        ms = v.empty() ? 0 : sm / v.size();
        per = sb > 0 ? sm / sb : 0;
    };
    double db, dms, dper, rb, rms, rper;
    mean_calls(det_calls_, db, dms, dper);
    mean_calls(rec_calls_, rb, rms, rper);
    auto gmean = [&](int k) {
        double s = 0;
        for (auto& g : gpu_samples_) s += g[k];
        return gpu_samples_.empty() ? NAN : s / gpu_samples_.size();
    };
    double gmax = 0;
    for (auto& g : gpu_samples_) gmax = std::max(gmax, g[2]);
    double total_fps = std::accumulate(fps_cam.begin(), fps_cam.end(), 0.0);
    // real-time check: does one second of frames take at most one second to process (throughput keeps up),
    // with every frame done inside the buffer (delay <= buffer_s) and the delay not growing?
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    size_t m = 0;
    for (auto& c : cams_)
        for (auto& r : c->rows) {
            double x = r.t_cap - t0, y = r.t_done - r.t_cap;
            sx += x; sy += y; sxx += x * x; sxy += x * y; ++m;
        }
    const double slope = m > 2 ? (m * sxy - sx * sy) / std::max(1e-9, m * sxx - sx * sx) : 0.0;
    const double arrived_ps = (arr1 - arr0) / cfg_.window;
    const double done_ratio = arr1 > arr0 ? (double)rows / (double)(arr1 - arr0) : 0.0;
    const int64_t overflow = ovf1 - ovf0;
    const double p99 = pct(cap, 99);
    const bool rt_ok = overflow == 0 && done_ratio >= 0.98 && std::isfinite(p99) && p99 <= cfg_.buffer_s * 1000 &&
                       slope < 0.02;
    auto why_rate = [&](int k) { return n ? (double)(gpu_c1_[5 + k] - gpu_c0_[5 + k]) / n / cfg_.window : 0.0; };
    double gpu_req_per_cam_s = 0, gpu_faces_per_frame = 0;
    if (cfg_.gpu_track && n) {
        const double dF = (double)(gpu_c1_[0] - gpu_c0_[0]), dA = (double)(gpu_c1_[1] - gpu_c0_[1]);
        const double dR = (double)(gpu_c1_[2] - gpu_c0_[2]), dK = (double)(gpu_c1_[3] - gpu_c0_[3]);
        (void)dF;
        gpu_req_per_cam_s = dR / n / cfg_.window;
        const double ratio = cams_.empty() ? 1.0 : std::min(1.0, cfg_.analyze_fps * cams_[0]->src.period);
        gpu_faces_per_frame = dK > 0 ? dA / (dK * ratio) : 0;
    }
    // End-to-end SLA: one second of one camera's video is done when all its frames are delivered (boxes, ids, names
    // known so far) AND every recognition first asked for by a frame of that second has been written. Delay is
    // counted from the end of that second. Seconds are evaluated only if they ended at least buffer_s before the end
    // of the measurement, so nothing that completes late is missed.
    std::vector<double> chunk_ms;
    long chunks_bad_frames = 0;
    {
        const double rel0 = t0 - t0_run_, rel1 = rel0 + cfg_.window - cfg_.buffer_s;
        for (auto& c : cams_) {
            const double fps = 1.0 / c->src.period;
            for (size_t sec = (size_t)std::ceil(rel0); sec + 1 <= (size_t)rel1; ++sec) {
                const bool have = sec < c->chunk_done.size() && c->chunk_frames[sec] >= 0.9 * fps;
                if (!have) { ++chunks_bad_frames; chunk_ms.push_back(1e9); continue; }
                chunk_ms.push_back((c->chunk_done[sec] - (t0_run_ + sec + 1.0)) * 1000.0);
            }
        }
    }
    long chunks_ok = 0;
    for (double v : chunk_ms) chunks_ok += v <= 1000.0;
    const double sla_pct = chunk_ms.empty() ? 0.0 : 100.0 * chunks_ok / chunk_ms.size();
    std::vector<double> name_ms(name_delay_);
    for (auto& v : name_ms) v *= 1000.0;
    double det_ms_sum = 0, rec_ms_sum = 0;
    int64_t of = 0, ff = 0;
    for (auto& c : cams_) { of += c->out_frames; ff += c->filled_frames; }
    const double out_fps = cams_.empty() ? 0 : of / cfg_.window / cams_.size();
    const double filled_pct = of ? 100.0 * ff / of : 0;
    auto pctf = [](const std::vector<float>& v, double p) {
        std::vector<double> d(v.begin(), v.end());
        for (auto& x : d) x *= 1000.0;
        return pct(d, p);
    };
    for (auto& x : det_calls_) det_ms_sum += x.second;
    for (auto& x : rec_calls_) rec_ms_sum += x.second;
    std::ostringstream o;
    o.setf(std::ios::fixed);
    o.precision(2);
    o << "{\"cameras\": " << n << ", \"variant\": \"" << cfg_.tag << "\", \"impl\": \"cpp\""
      << ", \"det_plan\": \"" << det_.name() << "\", \"rec_plan\": \"" << rec_.name() << "\""
      << ", \"policy\": \"" << cfg_.policy << "\", \"buffer_s\": " << cfg_.buffer_s
      << ", \"realtime_ok\": " << (rt_ok ? "true" : "false")
      << ", \"arrived_fps\": " << arrived_ps << ", \"done_ratio\": " << done_ratio
      << ", \"buffer_overflow\": " << overflow << ", \"queued_at_end\": " << queued_end
      << ", \"delay_slope_s_per_s\": " << slope
      << ", \"det_gpu_ms_per_s\": " << det_ms_sum / cfg_.window << ", \"rec_gpu_ms_per_s\": " << rec_ms_sum / cfg_.window
      << ", \"track_cpu_ms_per_s\": " << 1000 * track_cpu_s_ / cfg_.window
      << ", \"graphs\": " << cfg_.graphs << ", \"mapped\": " << cfg_.mapped << ", \"inflight\": " << cfg_.inflight
      << ", \"fps_per_cam_median\": " << JNum{pct(fps_cam, 50)} << ", \"fps_per_cam_min\": "
      << (fps_cam.empty() ? 0 : *std::min_element(fps_cam.begin(), fps_cam.end()))
      << ", \"total_fps\": " << total_fps
      << ", \"skipped_pct\": " << (elig ? 100.0 * skipped / elig : 0) << ", \"ring_dropped\": " << dropped
      << ", \"wait_p50_ms\": " << JNum{pct(wait, 50)} << ", \"wait_p95_ms\": " << JNum{pct(wait, 95)}
      << ", \"proc_p50_ms\": " << JNum{pct(proc, 50)} << ", \"proc_p95_ms\": " << JNum{pct(proc, 95)}
      << ", \"e2e_p50_ms\": " << JNum{pct(e2e, 50)} << ", \"e2e_p95_ms\": " << JNum{pct(e2e, 95)}
      << ", \"e2e_p99_ms\": " << JNum{pct(e2e, 99)}
      << ", \"cap_p50_ms\": " << JNum{pct(cap, 50)} << ", \"cap_p95_ms\": " << JNum{pct(cap, 95)}
      << ", \"cap_p99_ms\": " << JNum{pct(cap, 99)} << ", \"cap_max_ms\": " << JNum{pct(cap, 100)}
      << ", \"decode_lag_p50_ms\": " << JNum{pct(dlag, 50)} << ", \"decode_lag_p95_ms\": " << JNum{pct(dlag, 95)}
      << ", \"sla_ok\": " << (cfg_.unified && sla_pct >= 95.0 && overflow == 0 ? "true" : "false")
      << ", \"sla_seconds_within_1s_pct\": " << sla_pct << ", \"seconds_evaluated\": " << chunk_ms.size()
      << ", \"seconds_missing_frames\": " << chunks_bad_frames
      << ", \"sec_done_p50_ms\": " << JNum{pct(chunk_ms, 50)} << ", \"sec_done_p95_ms\": " << JNum{pct(chunk_ms, 95)}
      << ", \"sec_done_p99_ms\": " << JNum{pct(chunk_ms, 99)} << ", \"sec_done_max_ms\": " << JNum{pct(chunk_ms, 100)}
      << ", \"name_p50_ms\": " << JNum{pct(name_ms, 50)} << ", \"name_p95_ms\": " << JNum{pct(name_ms, 95)}
      << ", \"name_max_ms\": " << JNum{pct(name_ms, 100)} << ", \"name_wait_max_ms\": " << name_wait_max_ms_
      << ", \"boxes_per_cam_s\": " << (n ? uni_boxes_ / cfg_.window / n : 0.0)
      << ", \"named_box_pct\": " << (uni_boxes_ ? 100.0 * uni_named_ / uni_boxes_ : 0.0)
      << ", \"faces_left_unnamed_per_cam_s\": " << (n ? (double)(gpu_c1_[10] - gpu_c0_[10]) / n / cfg_.window : 0.0)
      << ", \"unified\": " << cfg_.unified << ", \"uni_split\": " << cfg_.uni_split
      << ", \"tick_ms_avg\": " << (ticks_ ? tick_ms_sum_ / ticks_ : 0.0)
      << ", \"split_A_ms_avg\": " << (ticks_ ? splitA_ms_ / ticks_ : 0.0)
      << ", \"split_gap_ms_avg\": " << (ticks_ ? split_gap_ms_ / ticks_ : 0.0)
      << ", \"split_B_ms_avg\": " << (ticks_ ? splitB_ms_ / ticks_ : 0.0)
      << ", \"ticks_per_s\": " << ticks_ / cfg_.window
      << ", \"rec_slots_used_pct\": " << (uni_slots_ ? 100.0 * uni_used_ / uni_slots_ : 0.0)
      << ", \"rec_denied_per_s\": " << uni_denied_ / cfg_.window
      << ", \"gpu_track\": " << cfg_.gpu_track
      << ", \"why_new_grow_periodic_unsure_crossed_per_cam_s\": [" << why_rate(0) << ", " << why_rate(1) << ", " << why_rate(2)
      << ", " << why_rate(3) << ", " << why_rate(4) << "]"
      << ", \"tracks_created_per_cam_s\": " << (n ? (double)(gpu_c1_[4] - gpu_c0_[4]) / n / cfg_.window : 0)
      << ", \"recheck_s\": " << cfg_.recheck_s << ", \"confirm_hits\": " << cfg_.confirm_hits
      << ", \"min_rec_px\": " << cfg_.min_rec_px << ", \"ttl\": " << cfg_.ttl << ", \"trk_iou\": " << cfg_.trk_iou
      << ", \"dist_k\": " << cfg_.dist_k << ", \"grow\": " << cfg_.grow << ", \"analyze_fps\": " << cfg_.analyze_fps
      << ", \"min_batch\": " << cfg_.min_batch << ", \"max_wait_ms\": " << cfg_.max_wait_ms
      << ", \"det_ctxs\": " << cfg_.det_ctxs << ", \"uni_headroom\": " << cfg_.uni_headroom
      << ", \"gpu_requests_per_cam_s\": " << gpu_req_per_cam_s
      << ", \"gpu_faces_per_frame\": " << gpu_faces_per_frame
      << ", \"bestshot\": " << cfg_.bestshot
      << ", \"rec_faces_per_s\": " << (rb * rec_calls_.size() / cfg_.window)
      << ", \"id_after_window_p50_ms\": " << JNum{pct(id_after_window_, 50) * 1000}
      << ", \"id_after_window_p95_ms\": " << JNum{pct(id_after_window_, 95) * 1000}
      << ", \"id_after_window_max_ms\": " << JNum{pct(id_after_window_, 100) * 1000}
      << ", \"kalman\": " << cfg_.kalman << ", \"window_s\": " << cfg_.window_s
      << ", \"stream_fps\": " << (cams_.empty() ? 0 : 1.0 / cams_[0]->src.period)
      << ", \"out_fps_per_cam\": " << out_fps << ", \"filled_pct\": " << filled_pct
      << ", \"out_p50_ms\": " << JNum{pctf(out_delay_, 50)} << ", \"out_p95_ms\": " << JNum{pctf(out_delay_, 95)}
      << ", \"out_p99_ms\": " << JNum{pctf(out_delay_, 99)} << ", \"out_max_ms\": " << JNum{pctf(out_delay_, 100)}
      << ", \"zero_copy\": " << cfg_.zero_copy << ", \"min_batch\": " << cfg_.min_batch << ", \"max_wait_ms\": " << cfg_.max_wait_ms
      << ", \"recognitions_per_frame\": " << (rows ? nrec / rows : 0)
      << ", \"faces_per_frame\": " << (rows ? nfaces / rows : 0)
      << ", \"first_id_p50_ms\": " << JNum{pct(first_id_, 50) * 1000} << ", \"first_id_p95_ms\": " << JNum{pct(first_id_, 95) * 1000}
      << ", \"refresh_p50_s\": " << JNum{pct(refresh_, 50)} << ", \"refresh_p95_s\": " << JNum{pct(refresh_, 95)}
      << ", \"rechecks_deferred\": " << deferred_
      << ", \"det_batch_avg\": " << db << ", \"det_gpu_ms_avg\": " << dms << ", \"det_ms_per_frame\": " << dper
      << ", \"rec_batch_avg\": " << rb << ", \"rec_gpu_ms_avg\": " << rms << ", \"rec_ms_per_face\": " << rper
      << ", \"gpu_util_pct\": " << JNum{gmean(0)} << ", \"decoder_util_pct\": " << JNum{gmean(1)}
      << ", \"gpu_mem_max_mib\": " << gmax << ", \"sm_clock_mhz\": " << JNum{gmean(3)} << ", \"gpu_power_w\": " << JNum{gmean(4)}
      << "}";
    for (auto& c : cams_) { c->src.ring.free_all(); if (c->zone_mask) cudaFree(c->zone_mask); }
    cams_.clear();
    return o.str();
}

// ------------------------------------------------------------------------------------------------ box accuracy
// How many boxes does the working mode skip? Run the real unified GPU pipeline on one clip, frame by frame:
//   reference = detect EVERY frame;  test = detect every K-th frame, Kalman-filled in between.
// A reference box is "skipped" if no delivered box on that frame overlaps it with IoU >= 0.5.
std::string Pipeline::eval_boxes(int N) {
    std::string path = cfg_.video;
    size_t at = path.find("{r}");
    if (at != std::string::npos) path.replace(at, 3, "0");
    struct Fr { std::vector<float4> box; std::vector<int> id; };
    const int Rcap = uni_R_.back();          // plenty of recognition slots: this measures boxes, not load
    double fps = 30;
    auto run = [&](int K, std::vector<Fr>& fr) {
        reset_gpu_track();
        VideoDecoder dec(path, DecodePool::hw_device(0));
        fps = dec.fps();
        FrameRing ring;
        ring.alloc(dec.width(), dec.height(), 1, false, false);
        const float dt = 1.f / (float)dec.fps();
        DetCtx* c = det_ctx_[0].get();
        fr.assign(N, {});
        std::vector<float4> hb(TRK);
        std::vector<int> hi(TRK);
        int last = -1;
        for (int f = 0; f < N; ++f) {
            AVFrame* af = dec.next();
            if (!af) break;
            if (f % K) continue;
            CK(cudaStreamSynchronize(DecodePool::hw_stream(0)));
            CK(cudaMemcpy2D(ring.wy(0), ring.copy_pitch(), af->data[0], af->linesize[0], ring.w(), ring.h(), cudaMemcpyDeviceToDevice));
            CK(cudaMemcpy2D(ring.wuv(0), ring.copy_pitch(), af->data[1], af->linesize[1], ring.w(), ring.h() / 2, cudaMemcpyDeviceToDevice));
            TickDesc& d = *c->desc_h;
            std::memset(&d, 0, sizeof(d));
            d.det.n = d.n = 1;
            d.det.f[0] = FrameRef{ring.y(0), ring.uv(0), ring.copy_pitch(), ring.w(), ring.h(), dec.matrix()};
            const float scale = std::min((float)DET_W / ring.w(), (float)DET_H / ring.h());
            d.det.lb = Letterbox{scale, (DET_W - ring.w() * scale) * 0.5f, (DET_H - ring.h() * scale) * 0.5f, ring.w(), ring.h()};
            d.trk[0] = TrackItem{0, f, last < 0 ? 1 : f - last, f * dt, dt, d.det.f[0]};
            CK(cudaMemcpyAsync(u_.desc, &d, sizeof(d), cudaMemcpyHostToDevice, det_s_));
            if (cfg_.uni_split) {
                CK(cudaGraphLaunch(uniA_.at(1), det_s_));
                CK(cudaGraphLaunch(uniB_.at(Rcap), det_s_));
            } else {
                CK(cudaGraphLaunch(uni_graphs_.at({1, Rcap}), det_s_));
            }
            CK(cudaStreamSynchronize(det_s_));
            for (int r = last + 1; r <= f; ++r) {
                const size_t row = (size_t)(r % OUT_RING) * TRK;     // camera 0
                CK(cudaMemcpy(hb.data(), tout_.box + row, TRK * sizeof(float4), cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(hi.data(), tout_.id + row, TRK * sizeof(int), cudaMemcpyDeviceToHost));
                for (int s = 0; s < TRK; ++s)
                    if (hi[s]) { fr[r].box.push_back(hb[s]); fr[r].id.push_back(hi[s]); }
            }
            last = f;
        }
        ring.free_all();
        return last;
    };
    auto best = [](const float4& a, const std::vector<float4>& v) {
        float b = 0.f;
        for (auto& x : v) b = std::max(b, Tracker::iou(&a.x, &x.x));
        return b;
    };
    std::vector<Fr> ref;
    run(1, ref);
    std::ostringstream o;
    o.setf(std::ios::fixed);
    o.precision(2);
    for (int K : {1, 2, 3}) {
        std::vector<Fr> test;
        const int last = K == 1 ? N - 1 : run(K, test);
        const std::vector<Fr>& t = K == 1 ? ref : test;
        long nref = 0, nout = 0, matched = 0, extra = 0, named = 0, kref = 0, kmat = 0, fref = 0, fmat = 0;
        double iou_sum = 0;
        for (int f = 0; f <= last; ++f) {
            const bool key = f % K == 0;
            for (auto& b : ref[f].box) {
                const float v = best(b, t[f].box);
                ++nref; (key ? kref : fref)++;
                if (v >= 0.5f) { ++matched; (key ? kmat : fmat)++; iou_sum += v; }
            }
            for (size_t i = 0; i < t[f].box.size(); ++i) {
                ++nout;
                if (best(t[f].box[i], ref[f].box) < 0.5f) ++extra;
                named += (t[f].id[i] >> 11) & 1;
            }
        }
        const double secs = (last + 1) / fps;
        o << "{\"detect_every\": " << K << ", \"frames\": " << last + 1
          << ", \"ref_boxes\": " << nref << ", \"ref_boxes_per_frame\": " << (double)nref / (last + 1)
          << ", \"delivered_boxes\": " << nout
          << ", \"skipped_boxes\": " << nref - matched << ", \"skipped_pct\": " << (nref ? 100.0 * (nref - matched) / nref : 0.0)
          << ", \"skipped_per_s\": " << (nref - matched) / secs
          << ", \"skipped_pct_on_detected_frames\": " << (kref ? 100.0 * (kref - kmat) / kref : 0.0)
          << ", \"skipped_pct_on_filled_frames\": " << (fref ? 100.0 * (fref - fmat) / fref : 0.0)
          << ", \"extra_boxes_pct\": " << (nout ? 100.0 * extra / nout : 0.0)
          << ", \"mean_iou_of_matched\": ";
        o.precision(3);
        o << (matched ? iou_sum / matched : 0.0);
        o.precision(2);
        o << ", \"named_box_pct\": " << (nout ? 100.0 * named / nout : 0.0) << "}\n";
    }
    return o.str();
}
