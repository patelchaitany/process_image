#pragma once
// Per-camera tracker (CPU; about 20 boxes per frame). Port of the Python v2 tracker:
// pass 1 greedy IoU against constant-velocity predictions, pass 2 centre distance within dist_k face sizes.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <vector>

struct Tracker {
    struct Track {
        float b[4], v[4];
        double seen, rec, first;
        bool known;
        uint64_t id;
    };
    struct Res {
        int track;           // index into tracks (valid until prune)
        bool is_new, is_due;
        double first, prev_rec;
        uint64_t id = 0;     // stable track id
    };
    std::vector<Track> tr;
    double reverify = 1.0, ttl = 1.5;
    float iou_thr = 0.3f, dist_k = 1.5f;
    uint64_t next_id = 1;

    static float iou(const float* a, const float* b) {
        float iw = std::min(a[2], b[2]) - std::max(a[0], b[0]);
        float ih = std::min(a[3], b[3]) - std::max(a[1], b[1]);
        if (iw <= 0 || ih <= 0) return 0.f;
        float in = iw * ih;
        return in / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - in + 1e-6f);
    }

    void update(const float4* boxes, int n, double t, std::vector<Res>& out) {
        const int m = (int)tr.size();
        std::vector<int> match(n, -1);
        std::vector<char> used_t(m, 0);
        if (n && m) {
            std::vector<float> pred(4 * (size_t)m);
            for (int j = 0; j < m; ++j)
                for (int k = 0; k < 4; ++k) pred[4 * j + k] = tr[j].b[k] + tr[j].v[k] * (float)(t - tr[j].seen);
            std::vector<std::tuple<float, int, int>> cand;
            cand.reserve((size_t)n * m);
            for (int i = 0; i < n; ++i) {
                const float* d = &boxes[i].x;
                for (int j = 0; j < m; ++j) {
                    float v = iou(d, &pred[4 * j]);
                    if (v >= iou_thr) cand.emplace_back(-v, i, j);
                }
            }
            std::sort(cand.begin(), cand.end());
            for (auto& c : cand) {
                int i = std::get<1>(c), j = std::get<2>(c);
                if (match[i] < 0 && !used_t[j]) { match[i] = j; used_t[j] = 1; }
            }
            if (dist_k > 0) {
                cand.clear();
                for (int i = 0; i < n; ++i) {
                    if (match[i] >= 0) continue;
                    const float* d = &boxes[i].x;
                    float dcx = 0.5f * (d[0] + d[2]), dcy = 0.5f * (d[1] + d[3]);
                    for (int j = 0; j < m; ++j) {
                        if (used_t[j]) continue;
                        const float* p = &pred[4 * j];
                        float size = std::max(std::max(p[2] - p[0], p[3] - p[1]), 1.f);
                        float rel = std::hypot(dcx - 0.5f * (p[0] + p[2]), dcy - 0.5f * (p[1] + p[3])) / size;
                        if (rel <= dist_k) cand.emplace_back(rel, i, j);
                    }
                }
                std::sort(cand.begin(), cand.end());
                for (auto& c : cand) {
                    int i = std::get<1>(c), j = std::get<2>(c);
                    if (match[i] < 0 && !used_t[j]) { match[i] = j; used_t[j] = 1; }
                }
            }
        }
        out.resize(n);
        for (int i = 0; i < n; ++i) {
            const float* d = &boxes[i].x;
            int j = match[i];
            if (j >= 0) {
                Track& T = tr[j];
                float dt = std::max((float)(t - T.seen), 1e-3f);
                for (int k = 0; k < 4; ++k) {
                    T.v[k] = 0.5f * T.v[k] + 0.5f * (d[k] - T.b[k]) / dt;
                    T.b[k] = d[k];
                }
                T.seen = t;
            } else {
                Track T{};
                for (int k = 0; k < 4; ++k) { T.b[k] = d[k]; T.v[k] = 0.f; }
                T.seen = t; T.rec = -1e9; T.first = t; T.known = false; T.id = next_id++;
                tr.push_back(T);
                j = (int)tr.size() - 1;
            }
            const Track& T = tr[j];
            out[i] = Res{j, !T.known, T.known && (t - T.rec >= reverify), T.first, T.rec};
        }
    }

    void mark(int j, double t) { tr[j].rec = t; tr[j].known = true; }

    void prune(double t) {
        tr.erase(std::remove_if(tr.begin(), tr.end(), [&](const Track& T) { return t - T.seen > ttl; }), tr.end());
    }
};

// ------------------------------------------------------------------------------------------------ Kalman tracker
// Detection runs on keyframes only (every K-th frame); the frames in between are filled by a constant-velocity
// Kalman filter per box coordinate (cx, cy, w, h), smoothed with a Rauch-Tung-Striebel backward pass between two
// keyframes. Each filled box therefore uses the detections on both sides (fixed-lag smoothing, lag = one keyframe gap).
struct KF1 {
    float x, v, P00, P01, P11;
};

inline void kf_predict(KF1& s, float dt, float qa) {
    s.x += s.v * dt;
    const float P00 = s.P00 + dt * (2.f * s.P01 + dt * s.P11) + qa * dt * dt * dt / 3.f;
    const float P01 = s.P01 + dt * s.P11 + qa * dt * dt / 2.f;
    const float P11 = s.P11 + qa * dt;
    s.P00 = P00; s.P01 = P01; s.P11 = P11;
}

inline void kf_update(KF1& s, float z, float r) {
    const float S = s.P00 + r, K0 = s.P00 / S, K1 = s.P01 / S, y = z - s.x;
    s.x += K0 * y;
    s.v += K1 * y;
    const float P00 = (1.f - K0) * s.P00, P01 = (1.f - K0) * s.P01, P11 = s.P11 - K1 * s.P01;
    s.P00 = P00; s.P01 = P01; s.P11 = P11;
}

struct FillBox {        // one in-between frame's box for one track
    int j;              // 1 .. gap-1 frames after the previous keyframe
    uint64_t id;
    float4 kal;         // Kalman-smoothed box
    float4 hold;        // baseline: last keyframe box held
    float4 lin;         // baseline: linear interpolation between the two detected boxes
    bool has_lin;
};

struct KalmanTracker {
    struct Track {
        KF1 a[4];            // filtered state at the last keyframe: cx, cy, w, h
        float last_box[4];   // last detected box (x1, y1, x2, y2)
        bool matched_last;   // detected at the previous keyframe
        double seen, rec, first;
        bool known;
        uint64_t id;
    };
    std::vector<Track> tr;
    double reverify = 1.0, ttl = 1.5;
    float iou_thr = 0.3f, dist_k = 1.5f;
    float sigma_meas = 0.05f;     // detector jitter, fraction of box height
    float sigma_acc = 1.0f;       // centre acceleration noise, box heights / s^2
    float sigma_acc_size = 0.3f;  // size acceleration noise
    uint64_t next_id = 1;

    static void to_cxcywh(const float* b, float* o) { o[0] = .5f * (b[0] + b[2]); o[1] = .5f * (b[1] + b[3]); o[2] = b[2] - b[0]; o[3] = b[3] - b[1]; }
    static float4 to_box(const float* c) { return make_float4(c[0] - .5f * c[2], c[1] - .5f * c[3], c[0] + .5f * c[2], c[1] + .5f * c[3]); }

    // gap: frame intervals since the previous keyframe of this camera; dt: frame period
    void update(const float4* boxes, int n, double t, int gap, float dt, std::vector<Tracker::Res>& out,
                std::vector<FillBox>* fills) {
        const int m = (int)tr.size();
        gap = std::max(gap, 1);
        const float T = gap * dt;
        // 1. forward predictions to every frame of the gap (kept for the backward pass)
        std::vector<KF1> pred((size_t)m * 4 * (gap + 1));
        auto P = [&](int j, int k, int s) -> KF1& { return pred[((size_t)j * 4 + k) * (gap + 1) + s]; };
        std::vector<float> pb(4 * (size_t)m);
        for (int j = 0; j < m; ++j) {
            const float h = std::max(tr[j].a[3].x, 4.f);
            for (int k = 0; k < 4; ++k) {
                const float sa = (k < 2 ? sigma_acc : sigma_acc_size) * h;
                P(j, k, 0) = tr[j].a[k];
                for (int s = 1; s <= gap; ++s) { P(j, k, s) = P(j, k, s - 1); kf_predict(P(j, k, s), dt, sa * sa); }
            }
            float c[4] = {P(j, 0, gap).x, P(j, 1, gap).x, P(j, 2, gap).x, P(j, 3, gap).x};
            float4 b = to_box(c);
            pb[4 * j] = b.x; pb[4 * j + 1] = b.y; pb[4 * j + 2] = b.z; pb[4 * j + 3] = b.w;
        }
        // 2. association against the predicted boxes (IoU, then centre distance)
        std::vector<int> match(n, -1);
        std::vector<char> used(m, 0);
        if (n && m) {
            std::vector<std::tuple<float, int, int>> cand;
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < m; ++j) {
                    float v = Tracker::iou(&boxes[i].x, &pb[4 * j]);
                    if (v >= iou_thr) cand.emplace_back(-v, i, j);
                }
            std::sort(cand.begin(), cand.end());
            for (auto& c : cand) { int i = std::get<1>(c), j = std::get<2>(c); if (match[i] < 0 && !used[j]) { match[i] = j; used[j] = 1; } }
            if (dist_k > 0) {
                cand.clear();
                for (int i = 0; i < n; ++i) {
                    if (match[i] >= 0) continue;
                    const float* d = &boxes[i].x;
                    for (int j = 0; j < m; ++j) {
                        if (used[j]) continue;
                        const float* p = &pb[4 * j];
                        float size = std::max(std::max(p[2] - p[0], p[3] - p[1]), 1.f);
                        float rel = std::hypot(.5f * (d[0] + d[2] - p[0] - p[2]), .5f * (d[1] + d[3] - p[1] - p[3])) / size;
                        if (rel <= dist_k) cand.emplace_back(rel, i, j);
                    }
                }
                std::sort(cand.begin(), cand.end());
                for (auto& c : cand) { int i = std::get<1>(c), j = std::get<2>(c); if (match[i] < 0 && !used[j]) { match[i] = j; used[j] = 1; } }
            }
        }
        std::vector<int> det_of(m, -1);
        for (int i = 0; i < n; ++i) if (match[i] >= 0) det_of[match[i]] = i;
        // 3. per existing track: update at the keyframe (if detected), RTS-smooth the in-between frames, emit fills
        for (int j = 0; j < m; ++j) {
            Track& T_ = tr[j];
            KF1 fin[4];
            const int i = det_of[j];
            float z[4] = {0, 0, 0, 0};
            if (i >= 0) to_cxcywh(&boxes[i].x, z);
            const float h = std::max(P(j, 3, gap).x, 4.f);
            for (int k = 0; k < 4; ++k) {
                fin[k] = P(j, k, gap);
                if (i >= 0) kf_update(fin[k], z[k], (sigma_meas * h) * (sigma_meas * h));
            }
            if (fills && gap > 1 && i >= 0) {   // only faces seen again at this keyframe (no ghost boxes)
                // backward pass: xs[s] = xf[s] + C_s (xs[s+1] - xp[s+1]), C_s = Pf[s] F^T Pp[s+1]^-1 (no
                // measurements between keyframes, so xf[s] = xp[s] for 0 < s < gap)
                float xs[4], vs[4];
                std::vector<float> cur((size_t)4 * (gap + 1));
                const int G = gap;
                for (int k = 0; k < 4; ++k) {
                    xs[k] = fin[k].x; vs[k] = fin[k].v;
                    for (int s = G - 1; s >= 1; --s) {
                        const KF1& f = P(j, k, s);
                        const KF1& p1 = P(j, k, s + 1);
                        const float a = f.P00 + dt * f.P01, b = f.P01, c = f.P01 + dt * f.P11, d = f.P11;   // Pf F^T
                        const float det = p1.P00 * p1.P11 - p1.P01 * p1.P01;
                        const float i00 = p1.P11 / det, i01 = -p1.P01 / det, i11 = p1.P00 / det;
                        const float C00 = a * i00 + b * i01, C01 = a * i01 + b * i11;
                        const float C10 = c * i00 + d * i01, C11 = c * i01 + d * i11;
                        const float dx = xs[k] - p1.x, dv = vs[k] - p1.v;
                        const float nx = f.x + C00 * dx + C01 * dv, nv = f.v + C10 * dx + C11 * dv;
                        xs[k] = nx; vs[k] = nv;
                        cur[(size_t)k * (gap + 1) + s] = nx;
                    }
                }
                for (int s = 1; s < G; ++s) {
                    const size_t o = gap + 1;
                    float c[4] = {cur[s], cur[o + s], cur[2 * o + s], cur[3 * o + s]};
                    FillBox fb{};
                    fb.j = s; fb.id = T_.id; fb.kal = to_box(c);
                    fb.hold = make_float4(T_.last_box[0], T_.last_box[1], T_.last_box[2], T_.last_box[3]);
                    fb.has_lin = T_.matched_last && i >= 0;
                    if (fb.has_lin) {
                        const float u = (float)s / gap;
                        const float* q = &boxes[i].x;
                        fb.lin = make_float4(T_.last_box[0] + u * (q[0] - T_.last_box[0]), T_.last_box[1] + u * (q[1] - T_.last_box[1]),
                                             T_.last_box[2] + u * (q[2] - T_.last_box[2]), T_.last_box[3] + u * (q[3] - T_.last_box[3]));
                    }
                    fills->push_back(fb);
                }
            }
            for (int k = 0; k < 4; ++k) T_.a[k] = fin[k];
            T_.matched_last = i >= 0;
            if (i >= 0) {
                for (int k = 0; k < 4; ++k) T_.last_box[k] = (&boxes[i].x)[k];
                T_.seen = t;
            }
        }
        // 4. new tracks for unmatched detections
        for (int i = 0; i < n; ++i) {
            if (match[i] >= 0) continue;
            Track T_{};
            float c[4];
            to_cxcywh(&boxes[i].x, c);
            const float h = std::max(c[3], 4.f);
            for (int k = 0; k < 4; ++k) {
                const float r = sigma_meas * h;
                T_.a[k] = KF1{c[k], 0.f, r * r, 0.f, (k < 2 ? 1.0f : 0.3f) * h * h};   // velocity unknown
                T_.last_box[k] = (&boxes[i].x)[k];
            }
            T_.matched_last = true;
            T_.seen = t; T_.rec = -1e9; T_.first = t; T_.known = false; T_.id = next_id++;
            tr.push_back(T_);
            match[i] = (int)tr.size() - 1;
        }
        out.resize(n);
        for (int i = 0; i < n; ++i) {
            const Track& T_ = tr[match[i]];
            out[i] = Tracker::Res{match[i], !T_.known, T_.known && (t - T_.rec >= reverify), T_.first, T_.rec, T_.id};
        }
        (void)T;
    }

    void mark(int j, double t) { tr[j].rec = t; tr[j].known = true; }
    void prune(double t) {
        tr.erase(std::remove_if(tr.begin(), tr.end(), [&](const Track& T_) { return t - T_.seen > ttl; }), tr.end());
    }
};
