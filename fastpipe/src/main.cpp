// fastpipe: stress test (default) or validation dump (--dump N).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "common.h"
#include "pipeline.h"

extern "C" {
#include <libavutil/frame.h>
}

static std::vector<int> parse_ints(const std::string& s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back(std::stoi(t));
    return v;
}

static Config parse(int argc, char** argv) {
    Config c;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", k.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (k == "--det") c.det_plan = val();
        else if (k == "--rec") c.rec_plan = val();
        else if (k == "--video") c.video = val();
        else if (k == "--out") c.out_dir = val();
        else if (k == "--tag") c.tag = val();
        else if (k == "--counts") c.counts = parse_ints(val());
        else if (k == "--analyze-fps") c.analyze_fps = std::stod(val());
        else if (k == "--warmup") c.warmup = std::stod(val());
        else if (k == "--window") c.window = std::stod(val());
        else if (k == "--det-batch") c.det_batch = std::stoi(val());
        else if (k == "--rec-batch") c.rec_batch = std::stoi(val());
        else if (k == "--inflight") c.inflight = std::stoi(val());
        else if (k == "--decoders") c.decoders = std::stoi(val());
        else if (k == "--gallery") c.gallery = std::stoi(val());
        else if (k == "--det-ctxs") c.det_ctxs = std::stoi(val());
        else if (k == "--rec-ctxs") c.rec_ctxs = std::stoi(val());
        else if (k == "--conf") c.conf = std::stof(val());
        else if (k == "--iou") c.iou = std::stof(val());
        else if (k == "--max-faces") c.max_faces = std::stoi(val());
        else if (k == "--reverify") c.reverify = std::stod(val());
        else if (k == "--dist-k") c.dist_k = std::stof(val());
        else if (k == "--min-rec-px") c.min_rec_px = std::stof(val());
        else if (k == "--track") c.track = std::stoi(val());
        else if (k == "--budget") c.budget = std::stoi(val());
        else if (k == "--graphs") c.graphs = std::stoi(val());
        else if (k == "--mapped") c.mapped = std::stoi(val());
        else if (k == "--det-chw") c.det_chw = std::stoi(val());
        else if (k == "--dump") c.dump = std::stoi(val());
        else if (k == "--kbench") c.kbench = std::stoi(val());
        else if (k == "--kfeval") c.kfeval = std::stoi(val());
        else if (k == "--boxeval") c.boxeval = std::stoi(val());
        else if (k == "--kalman") c.kalman = std::stoi(val());
        else if (k == "--gpu-track") c.gpu_track = std::stoi(val());
        else if (k == "--recheck-s") c.recheck_s = std::stod(val());
        else if (k == "--retry-s") c.retry_s = std::stod(val());
        else if (k == "--grow") c.grow = std::stof(val());
        else if (k == "--confirm-hits") c.confirm_hits = std::stoi(val());
        else if (k == "--ttl") c.ttl = std::stof(val());
        else if (k == "--trk-iou") c.trk_iou = std::stof(val());
        else if (k == "--cross-iou") c.cross_iou = std::stof(val());
        else if (k == "--pool") c.pool = std::stoi(val());
        else if (k == "--rec-min") c.rec_min = std::stoi(val());
        else if (k == "--cb") c.cb = std::stoi(val());
        else if (k == "--unified") c.unified = std::stoi(val());
        else if (k == "--uni-rmax") c.uni_rmax = std::stoi(val());
        else if (k == "--uni-split") c.uni_split = std::stoi(val()) != 0;
        else if (k == "--uni-headroom") c.uni_headroom = std::stod(val());
        else if (k == "--cb-idle") c.cb_idle = std::stoi(val());
        else if (k == "--rec-wait-ms") c.rec_wait_ms = std::stod(val());
        else if (k == "--bestshot") { c.bestshot = std::stoi(val()); if (c.bestshot) c.kalman = true; }
        else if (k == "--kf-acc") c.kf_acc = std::stof(val());
        else if (k == "--kf-meas") c.kf_meas = std::stof(val());
        else if (k == "--window-s") c.window_s = std::stod(val());
        else if (k == "--policy") c.policy = val();
        else if (k == "--buffer-s") c.buffer_s = std::stod(val());
        else if (k == "--fifo-inflight") c.fifo_inflight = std::stoi(val());
        else if (k == "--zones") c.zones = std::stoi(val());
        else if (k == "--live") c.live = val();
        else if (k == "--det-zoom") c.det_zoom = std::min(1.0f, std::max(0.1f, std::stof(val())));
        else if (k == "--live-fps") c.live_fps = std::stod(val());
        else if (k == "--live-size") {
            const std::string v = val();
            const size_t x = v.find('x');
            c.live_w = std::stoi(v.substr(0, x));
            c.live_h = std::stoi(v.substr(x + 1));
        }
        else if (k == "--zero-copy") c.zero_copy = std::stoi(val());
        else if (k == "--min-batch") c.min_batch = std::stoi(val());
        else if (k == "--max-wait-ms") c.max_wait_ms = std::stod(val());
        else if (k == "--dump-path") c.dump_path = val();
        else if (k == "--match-thr") { c.match_thr = std::stof(val()); c.match_thr_set = true; }
        else if (k == "--unknown-thr") { c.unknown_thr = std::stof(val()); c.unknown_thr_set = true; }
        else { std::fprintf(stderr, "unknown option %s\n", k.c_str()); std::exit(2); }
    }
    if (c.unified) {      // one giant pipeline: GPU tracker, no crop pool, bucket-aligned batches
        c.gpu_track = true;
        c.cb = true;
        c.pool = 8;
        c.pool_margin = 0;
    }
    if (c.det_plan.empty() || c.rec_plan.empty()) {
        std::fprintf(stderr, "usage: fastpipe --det det.plan --rec rec.plan [--counts 1,2,16,...] [options]\n");
        std::exit(2);
    }
    return c;
}

// Match thresholds belong to the recogniser: the same cosine score means different things for different models.
// <engine without .plan>.thr holds "match_thr <x>" and "unknown_thr <y>" (written by tools/setup/build_engines.sh from the
// measured score distributions). Values given on the command line win.
static void load_thresholds(Config& c) {
    std::string p = c.rec_plan;
    const size_t dot = p.rfind('.');
    if (dot != std::string::npos && p.find('/', dot) == std::string::npos) p.erase(dot);
    p += ".thr";
    std::ifstream f(p);
    const bool have = f.good();
    std::string k;
    float v;
    while (f >> k >> v) {
        if (k == "match_thr" && !c.match_thr_set) c.match_thr = v;
        else if (k == "unknown_thr" && !c.unknown_thr_set) c.unknown_thr = v;
    }
    std::fprintf(stderr, "[rec] match at score >= %.2f, unknown below %.2f (%s%s)\n", c.match_thr, c.unknown_thr,
                 have ? p.c_str() : "no .thr file beside the engine: ArcFace R50 defaults",
                 c.match_thr_set || c.unknown_thr_set ? ", command line" : "");
}

// Random unit-norm gallery (identity content does not change the cost of matching)
static __half* make_gallery(int G) {
    std::mt19937 rng(0);
    std::normal_distribution<float> nd;
    std::vector<__half> h((size_t)G * EMB);
    for (int g = 0; g < G; ++g) {
        std::vector<float> v(EMB);
        double s = 0;
        for (auto& x : v) { x = nd(rng); s += x * x; }
        for (int k = 0; k < EMB; ++k) h[(size_t)g * EMB + k] = __float2half(v[k] / std::sqrt(s));
    }
    __half* d = nullptr;
    CK(cudaMalloc(&d, h.size() * sizeof(__half)));
    CK(cudaMemcpy(d, h.data(), h.size() * sizeof(__half), cudaMemcpyHostToDevice));
    return d;
}

// Validation: run N frames of one clip through K1 -> det -> K2 -> K3 (all faces) -> rec and write everything.
static int run_dump(const Config& cfg, TrtEngine& det, TrtEngine& rec) {
    std::string path = cfg.video;
    size_t at = path.find("{r}");
    if (at != std::string::npos) path.replace(at, 3, "0");
    VideoDecoder dec(path, DecodePool::hw_device());
    FrameRing ring;
    ring.alloc(dec.width(), dec.height(), 4, false, false);
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    __half *din, *dout, *rin, *remb;
    CK(cudaMalloc(&din, sizeof(__half) * det.in_item_elems()));
    CK(cudaMalloc(&dout, sizeof(__half) * det.out_item_elems()));
    CK(cudaMalloc(&rin, sizeof(__half) * rec.in_item_elems() * MAX_FACES));
    CK(cudaMalloc(&remb, sizeof(__half) * rec.out_item_elems() * MAX_FACES));
    HostDev<FrameDets> dets;
    dets.alloc(1, cfg.mapped, false);
    HostDev<CropReq> req;
    req.alloc(MAX_FACES, cfg.mapped, false);
    std::ofstream out(cfg.dump_path, std::ios::binary);
    std::vector<__half> emb_h(rec.out_item_elems() * MAX_FACES);
    for (int k = 0; k < cfg.dump; ++k) {
        AVFrame* f = dec.next();
        if (!f) break;
        CK(cudaStreamSynchronize(DecodePool::hw_stream(0)));
        CK(cudaMemcpy2D(ring.wy(0), ring.copy_pitch(), f->data[0], f->linesize[0], ring.w(), ring.h(), cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy2D(ring.wuv(0), ring.copy_pitch(), f->data[1], f->linesize[1], ring.w(), ring.h() / 2, cudaMemcpyDeviceToDevice));
        DetBatch b{};
        b.n = 1;
        b.f[0] = FrameRef{ring.y(0), ring.uv(0), ring.copy_pitch(), ring.w(), ring.h(), dec.matrix()};
        const float scale = std::min((float)DET_W / ring.w(), (float)DET_H / ring.h()) * cfg.det_zoom;
        b.lb = Letterbox{scale, (DET_W - ring.w() * scale) * 0.5f, (DET_H - ring.h() * scale) * 0.5f, ring.w(), ring.h()};
        launch_prep_det(b, din, s);
        det.run(1, din, dout, s);
        launch_decode_nms(dout, 1, (int)(det.out_item_elems() / 5), cfg.det_chw, cfg.conf, cfg.iou, b.lb, ZoneMasks{},
                          dets.d, s);
        dets.to_host(1, s);
        CK(cudaStreamSynchronize(s));
        if (k == 0) {   // K1 output of the first frame, for a direct check of the pre-processing
            std::vector<__half> t(det.in_item_elems());
            CK(cudaMemcpy(t.data(), din, t.size() * sizeof(__half), cudaMemcpyDeviceToHost));
            std::ofstream(cfg.dump_path + ".detin", std::ios::binary).write((const char*)t.data(), t.size() * 2);
            std::vector<uint8_t> nv12((size_t)ring.w() * ring.h() * 3 / 2);
            CK(cudaMemcpy2D(nv12.data(), ring.w(), ring.y(0), ring.copy_pitch(), ring.w(), ring.h() * 3 / 2, cudaMemcpyDeviceToHost));
            std::ofstream(cfg.dump_path + ".nv12", std::ios::binary).write((const char*)nv12.data(), nv12.size());
        }
        const int n = std::min(dets.h[0].count, MAX_FACES);
        for (int i = 0; i < n; ++i) {
            float4 bx = dets.h[0].box[i];
            req.h[i] = CropReq{ring.y(0), ring.uv(0), ring.copy_pitch(), ring.w(), ring.h(), dec.matrix(), bx.x, bx.y, bx.z, bx.w};
        }
        if (n) {
            req.to_device(n, s);
            launch_crop_prep(req.d, n, rin, s);
            rec.run(rec.bucket_for(n), rin, remb, s);
            CK(cudaMemcpyAsync(emb_h.data(), remb, sizeof(__half) * rec.out_item_elems() * n, cudaMemcpyDeviceToHost, s));
            CK(cudaStreamSynchronize(s));
        }
        int32_t hdr[2] = {k, n};
        out.write((const char*)hdr, sizeof(hdr));
        for (int i = 0; i < n; ++i) {
            float4 bx = dets.h[0].box[i];
            float v[5] = {bx.x, bx.y, bx.z, bx.w, dets.h[0].score[i]};
            out.write((const char*)v, sizeof(v));
        }
        for (int i = 0; i < n * EMB; ++i) {
            float v = __half2float(emb_h[i]);
            out.write((const char*)&v, sizeof(v));
        }
    }
    std::fprintf(stderr, "dump: %d frames -> %s\n", cfg.dump, cfg.dump_path.c_str());
    ring.free_all();
    return 0;
}


// Kernel micro-benchmark on real frames: GPU time per call (CUDA events) and achieved DRAM bandwidth.
static int run_kbench(const Config& cfg, TrtEngine& det, TrtEngine& rec) {
    std::string path = cfg.video;
    size_t at = path.find("{r}");
    if (at != std::string::npos) path.replace(at, 3, "0");
    VideoDecoder dec(path, DecodePool::hw_device(0));
    const int B = 32;
    FrameRing ring;
    ring.alloc(dec.width(), dec.height(), B, false, false);
    for (int i = 0; i < B; ++i) {        // 32 distinct frames, so K1 really reads DRAM, not one cached frame
        AVFrame* f = dec.next();
        for (int skip = 0; skip < 7; ++skip) f = dec.next();
        CK(cudaStreamSynchronize(DecodePool::hw_stream(0)));
        CK(cudaMemcpy2D(ring.wy(i), ring.copy_pitch(), f->data[0], f->linesize[0], ring.w(), ring.h(), cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy2D(ring.wuv(i), ring.copy_pitch(), f->data[1], f->linesize[1], ring.w(), ring.h() / 2, cudaMemcpyDeviceToDevice));
    }
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    __half *din, *dout, *rin, *remb, *gal;
    const int R = 128, G = cfg.gallery;
    CK(cudaMalloc(&din, sizeof(__half) * det.in_item_elems() * B));
    CK(cudaMalloc(&dout, sizeof(__half) * det.out_item_elems() * B));
    CK(cudaMalloc(&rin, sizeof(__half) * rec.in_item_elems() * R));
    CK(cudaMalloc(&remb, sizeof(__half) * rec.out_item_elems() * R));
    gal = make_gallery(G);
    FrameDets* dets;
    CK(cudaMalloc(&dets, sizeof(FrameDets) * B));
    CropReq* req_d;
    MatchOut* m_d;
    CK(cudaMalloc(&req_d, sizeof(CropReq) * R));
    CK(cudaMalloc(&m_d, sizeof(MatchOut) * R));
    DetBatch b{};
    b.n = B;
    for (int i = 0; i < B; ++i) b.f[i] = FrameRef{ring.y(i), ring.uv(i), ring.copy_pitch(), ring.w(), ring.h(), dec.matrix()};
    const float scale = std::min((float)DET_W / ring.w(), (float)DET_H / ring.h());
    b.lb = Letterbox{scale, (DET_W - ring.w() * scale) * 0.5f, (DET_H - ring.h() * scale) * 0.5f, ring.w(), ring.h()};
    const int A = (int)(det.out_item_elems() / 5);
    launch_prep_det(b, din, s);
    det.run(B, din, dout, s);
    launch_decode_nms(dout, B, A, cfg.det_chw, cfg.conf, cfg.iou, b.lb, ZoneMasks{}, dets, s);
    std::vector<FrameDets> h(B);
    CK(cudaMemcpyAsync(h.data(), dets, sizeof(FrameDets) * B, cudaMemcpyDeviceToHost, s));
    CK(cudaStreamSynchronize(s));
    std::vector<CropReq> reqs;
    int faces = 0;
    for (int i = 0; i < B && (int)reqs.size() < R; ++i) {
        faces += h[i].count;
        for (int k = 0; k < h[i].count && (int)reqs.size() < R; ++k) {
            float4 bx = h[i].box[k];
            reqs.push_back(CropReq{ring.y(i), ring.uv(i), ring.copy_pitch(), ring.w(), ring.h(), dec.matrix(), bx.x, bx.y, bx.z, bx.w});
        }
    }
    const int nr = (int)reqs.size();
    CK(cudaMemcpy(req_d, reqs.data(), sizeof(CropReq) * nr, cudaMemcpyHostToDevice));
    launch_crop_prep(req_d, nr, rin, s);
    rec.run(rec.bucket_for(nr), rin, remb, s);
    CK(cudaStreamSynchronize(s));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    auto timeit = [&](const char* name, auto&& fn, int items, const char* unit, double bytes) {
        for (int i = 0; i < 20; ++i) fn();
        const int N = 300;
        CK(cudaEventRecord(e0, s));
        for (int i = 0; i < N; ++i) fn();
        CK(cudaEventRecord(e1, s));
        CK(cudaEventSynchronize(e1));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        const double us = 1000.0 * ms / N;
        std::printf("{\"kernel\": \"%s\", \"items\": %d, \"unit\": \"%s\", \"us_per_call\": %.2f, \"us_per_item\": %.3f, "
                    "\"bytes_per_call\": %.0f, \"achieved_GBps\": %.1f}\n",
                    name, items, unit, us, us / items, bytes, bytes / (us * 1e3));
    };
    const double fbytes = (double)ring.w() * ring.h() * 1.5;          // NV12 read
    const double tbytes = 3.0 * DET_H * DET_W * 2;                     // FP16 planar write
    timeit("K1 prep_det", [&] { launch_prep_det(b, din, s); }, B, "frame", B * (fbytes + tbytes));
    timeit("K2 decode_nms", [&] { launch_decode_nms(dout, B, A, cfg.det_chw, cfg.conf, cfg.iou, b.lb, ZoneMasks{}, dets, s); },
           B, "frame", B * (5.0 * A * 2 + sizeof(FrameDets)));
    timeit("K3 crop_prep", [&] { launch_crop_prep(req_d, nr, rin, s); }, nr, "face", nr * (3.0 * REC * REC * 2));
    timeit("K4 match", [&] { launch_match(remb, nr, gal, G, m_d, s); }, nr, "face", nr * EMB * 2.0 + (double)G * EMB * 2);
    timeit("TRT detector", [&] { det.run(B, din, dout, s); }, B, "frame", 0);
    timeit("TRT recogniser", [&] { rec.run(rec.bucket_for(nr), rin, remb, s); }, nr, "face", 0);
    std::fprintf(stderr, "kbench: %d frames, %d faces (avg %.1f per frame), gallery %d\n", B, nr, faces / (double)B, G);
    ring.free_all();
    return 0;
}

// Kalman-fill accuracy: detect EVERY frame of one clip, feed only every K-th frame's detections to the Kalman tracker,
// and score the smoothed boxes for the skipped frames against the real detections on those frames.
static int run_kfeval(const Config& cfg, TrtEngine& det) {
    std::string path = cfg.video;
    size_t at = path.find("{r}");
    if (at != std::string::npos) path.replace(at, 3, "0");
    VideoDecoder dec(path, DecodePool::hw_device(0));
    FrameRing ring;
    ring.alloc(dec.width(), dec.height(), 1, false, false);
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    __half *din, *dout;
    CK(cudaMalloc(&din, sizeof(__half) * det.in_item_elems()));
    CK(cudaMalloc(&dout, sizeof(__half) * det.out_item_elems()));
    HostDev<FrameDets> dets;
    dets.alloc(1, cfg.mapped, false);
    const int N = cfg.kfeval;
    const float dt = 1.0f / (float)dec.fps();
    std::vector<std::vector<float4>> all(N);
    for (int k = 0; k < N; ++k) {
        AVFrame* f = dec.next();
        CK(cudaStreamSynchronize(DecodePool::hw_stream(0)));
        CK(cudaMemcpy2D(ring.wy(0), ring.copy_pitch(), f->data[0], f->linesize[0], ring.w(), ring.h(), cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy2D(ring.wuv(0), ring.copy_pitch(), f->data[1], f->linesize[1], ring.w(), ring.h() / 2, cudaMemcpyDeviceToDevice));
        DetBatch b{};
        b.n = 1;
        b.f[0] = FrameRef{ring.y(0), ring.uv(0), ring.copy_pitch(), ring.w(), ring.h(), dec.matrix()};
        const float scale = std::min((float)DET_W / ring.w(), (float)DET_H / ring.h());
        b.lb = Letterbox{scale, (DET_W - ring.w() * scale) * 0.5f, (DET_H - ring.h() * scale) * 0.5f, ring.w(), ring.h()};
        launch_prep_det(b, din, s);
        det.run(1, din, dout, s);
        launch_decode_nms(dout, 1, (int)(det.out_item_elems() / 5), cfg.det_chw, cfg.conf, cfg.iou, b.lb, ZoneMasks{}, dets.d, s);
        dets.to_host(1, s);
        CK(cudaStreamSynchronize(s));
        const int n = std::min(dets.h[0].count, cfg.max_faces);
        all[k].assign(dets.h[0].box, dets.h[0].box + n);
    }
    auto best_iou = [](const float4& a, const std::vector<float4>& v) {
        float b = 0.f;
        for (auto& x : v) b = std::max(b, Tracker::iou(&a.x, &x.x));
        return b;
    };
    std::vector<int> Ks = cfg.counts.size() && cfg.counts[0] > 0 && cfg.counts != std::vector<int>{1, 2, 16, 32, 64, 128} ? cfg.counts : std::vector<int>{2, 3, 5, 6, 10, 15};
    for (int K : Ks) {
        KalmanTracker kt;
        kt.dist_k = cfg.dist_k;
        kt.sigma_acc = cfg.kf_acc;
        kt.sigma_meas = cfg.kf_meas;
        std::vector<Tracker::Res> res;
        std::vector<FillBox> fills;
        std::vector<std::vector<float4>> kal_at(N);
        double sk = 0, sh = 0, sl = 0, skb = 0;
        int nk = 0, nl = 0, gk = 0, gh = 0, gl = 0, gkb = 0;
        int last = -1;
        for (int f = 0; f < N; f += K) {
            fills.clear();
            kt.update(all[f].data(), (int)all[f].size(), f * dt, last < 0 ? 1 : f - last, dt, res, &fills);
            kt.prune(f * dt);
            for (auto& fb : fills) {
                const int idx = last + fb.j;
                const float ik = best_iou(fb.kal, all[idx]), ih = best_iou(fb.hold, all[idx]);
                sk += ik; sh += ih; ++nk;
                gk += ik >= 0.5f; gh += ih >= 0.5f;
                if (fb.has_lin) {     // same faces for a like-for-like comparison with linear interpolation
                    const float il = best_iou(fb.lin, all[idx]);
                    sl += il; gl += il >= 0.5f; ++nl;
                    skb += ik; gkb += ik >= 0.5f;
                }
                kal_at[idx].push_back(fb.kal);
            }
            last = f;
        }
        int real = 0, covered = 0;     // detections on skipped frames that a filled box covers (IoU >= 0.5)
        for (int f = 0; f < N; ++f) {
            if (f % K == 0 || f > last) continue;
            for (auto& d : all[f]) { ++real; covered += best_iou(d, kal_at[f]) >= 0.5f; }
        }
        std::printf("{\"keyframe_every\": %d, \"detect_fps\": %.1f, \"filled_boxes\": %d, \"kalman_mean_iou\": %.3f, "
                    "\"kalman_iou50_pct\": %.1f, \"hold_mean_iou\": %.3f, \"hold_iou50_pct\": %.1f, \"linear_mean_iou\": %.3f, "
                    "\"linear_iou50_pct\": %.1f, \"kalman_same_faces_mean_iou\": %.3f, \"kalman_same_faces_iou50_pct\": %.1f, "
                    "\"recall_of_real_faces_pct\": %.1f}\n",
                    K, dec.fps() / K, nk, nk ? sk / nk : 0, nk ? 100.0 * gk / nk : 0, nk ? sh / nk : 0, nk ? 100.0 * gh / nk : 0,
                    nl ? sl / nl : 0, nl ? 100.0 * gl / nl : 0, nl ? skb / nl : 0, nl ? 100.0 * gkb / nl : 0,
                    real ? 100.0 * covered / real : 0);
        std::fflush(stdout);
    }
    ring.free_all();
    return 0;
}

int main(int argc, char** argv) {
    Config cfg = parse(argc, argv);
    load_thresholds(cfg);
    setenv("CUDA_MODULE_LOADING", "EAGER", 0);   // no lazy kernel loading while graphs are being captured
    // FFmpeg's NVDEC shares the primary context and requires blocking-sync scheduling; set it before the context
    // exists. Blocking sync also means waiting threads sleep instead of spinning on a CPU core.
    CK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
    CK(cudaSetDevice(0));
    DecodePool::hw_device(0);
    // continuous batching uses finer buckets so batches can be cut to an exact engine size
    TrtEngine det(cfg.det_plan, cfg.cb && !cfg.unified ? std::vector<int>{1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32}
                                       : std::vector<int>{1, 2, 4, 8, 12, 16, 24, 32}, cfg.graphs);
    TrtEngine rec(cfg.rec_plan, cfg.cb ? std::vector<int>{1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128}
                                       : std::vector<int>{1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128}, cfg.graphs);
    if (cfg.dump) return run_dump(cfg, det, rec);
    if (cfg.kbench) return run_kbench(cfg, det, rec);
    if (cfg.kfeval) return run_kfeval(cfg, det);
    __half* gallery = make_gallery(cfg.gallery);
    mkdir(cfg.out_dir.c_str(), 0755);
    Pipeline p(cfg, det, rec, gallery, cfg.gallery);
    if (cfg.boxeval) {
        std::string rows = p.eval_boxes(cfg.boxeval);
        std::printf("%s", rows.c_str());
        std::ofstream(cfg.out_dir + "/boxeval.jsonl") << rows;
        return 0;
    }
    if (!cfg.live.empty()) {
        p.run_live(cfg.counts.empty() ? 1 : cfg.counts[0]);
        cudaFree(gallery);
        return 0;
    }
    for (int n : cfg.counts) {
        std::string row = p.run_count(n);
        std::printf("%s\n", row.c_str());
        std::fflush(stdout);
        std::ofstream(cfg.out_dir + "/summary.jsonl", std::ios::app) << row << "\n";
    }
    cudaFree(gallery);
    return 0;
}
