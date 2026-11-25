// K5 track_update: GPU Kalman tracker + gap fill + recognition triggers, one block per camera-frame.
// K6 scatter_identity: recogniser results back into the track table.
#include <cfloat>

#include "gputrack.cuh"

namespace {

constexpr unsigned FULL = 0xffffffffu;
constexpr int GAPMAX = 8;          // largest keyframe gap that is smoothed (longer gaps: prediction only)

struct KF {
    float x, v, P00, P01, P11;
};

__device__ __forceinline__ void kf_predict(KF& s, float dt, float qa) {
    s.x += s.v * dt;
    const float P00 = s.P00 + dt * (2.f * s.P01 + dt * s.P11) + qa * dt * dt * dt * (1.f / 3.f);
    const float P01 = s.P01 + dt * s.P11 + qa * dt * dt * 0.5f;
    const float P11 = s.P11 + qa * dt;
    s.P00 = P00; s.P01 = P01; s.P11 = P11;
}

__device__ __forceinline__ void kf_update(KF& s, float z, float r) {
    const float S = s.P00 + r, K0 = s.P00 / S, K1 = s.P01 / S, y = z - s.x;
    s.x += K0 * y;
    s.v += K1 * y;
    const float P00 = (1.f - K0) * s.P00, P01 = (1.f - K0) * s.P01, P11 = s.P11 - K1 * s.P01;
    s.P00 = P00; s.P01 = P01; s.P11 = P11;
}

__device__ __forceinline__ float iou(float4 a, float4 b) {
    const float iw = fminf(a.z, b.z) - fmaxf(a.x, b.x), ih = fminf(a.w, b.w) - fmaxf(a.y, b.y);
    if (iw <= 0.f || ih <= 0.f) return 0.f;
    const float in = iw * ih;
    return in / ((a.z - a.x) * (a.w - a.y) + (b.z - b.x) * (b.w - b.y) - in + 1e-6f);
}

__device__ __forceinline__ int ring(int f) { return ((f % OUT_RING) + OUT_RING) % OUT_RING; }

__device__ __forceinline__ float4 to_box(float cx, float cy, float w, float h) {
    return make_float4(cx - .5f * w, cy - .5f * h, cx + .5f * w, cy + .5f * h);
}

// Block-wide arg-max (value, then lowest index). All threads get the result.
__device__ __forceinline__ void block_argmax(float v, int i, float* sv, int* si, float& ov, int& oi) {
    for (int o = 16; o > 0; o >>= 1) {
        const float v2 = __shfl_down_sync(FULL, v, o);
        const int i2 = __shfl_down_sync(FULL, i, o);
        if (v2 > v || (v2 == v && i2 >= 0 && (i < 0 || i2 < i))) { v = v2; i = i2; }
    }
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = blockDim.x >> 5;
    if (l == 0) { sv[w] = v; si[w] = i; }
    __syncthreads();
    if (w == 0) {
        v = l < nw ? sv[l] : -FLT_MAX;
        i = l < nw ? si[l] : -1;
        for (int o = 16; o > 0; o >>= 1) {
            const float v2 = __shfl_down_sync(FULL, v, o);
            const int i2 = __shfl_down_sync(FULL, i, o);
            if (v2 > v || (v2 == v && i2 >= 0 && (i < 0 || i2 < i))) { v = v2; i = i2; }
        }
        if (l == 0) { sv[32] = v; si[32] = i; }
    }
    __syncthreads();
    ov = sv[32];
    oi = si[32];
    __syncthreads();
}

struct Shared {
    float4 det[MAX_FACES];
    float dscore[MAX_FACES];
    float4 pb[TRK];               // predicted track boxes
    float C[MAX_FACES * TRK];     // association scores, row = detection
    int tslot[TRK];               // compacted alive slots
    int mD[MAX_FACES], mT[TRK];   // matches
    int freeL[TRK], unL[MAX_FACES];
    float sv[33];
    int si[33];
    int m, nfree, nun, nreq;
};

// Greedy assignment on C (higher is better) for pairs above thr, until no pair is left.
__device__ void greedy(Shared& sh, int n, int m, float thr) {
    for (int it = 0; it < MAX_FACES; ++it) {
        float best = -FLT_MAX;
        int bi = -1;
        for (int e = threadIdx.x; e < n * m; e += blockDim.x) {
            const int d = e / m, j = e - d * m;
            if (sh.mD[d] >= 0 || sh.mT[j] >= 0) continue;
            const float c = sh.C[d * TRK + j];
            if (c > best) { best = c; bi = e; }
        }
        float bv;
        int be;
        block_argmax(best, bi, sh.sv, sh.si, bv, be);
        if (be < 0 || bv < thr) break;
        if (threadIdx.x == 0) {
            const int d = be / m, j = be - d * m;
            sh.mD[d] = j;
            sh.mT[j] = d;
        }
        __syncthreads();
    }
}

__device__ bool request_recognition(const CropPool& cp, const TrackItem& it, int slot, unsigned gen, float4 b,
                                    float t_first, float t_req, int limit, int* nreq) {
    if (cp.pool == 0) {
        // unified pipeline: take one of this tick's recogniser slots (exact, lock-free); over the cap -> ask again later
        int q = *(volatile int*)cp.batch_count;
        for (;;) {
            if (q >= limit) { atomicAdd(cp.denied, 1); return false; }
            const int old = atomicCAS(cp.batch_count, q, q + 1);
            if (old == q) break;
            q = old;
        }
        cp.req[q] = CropReq{it.f.y, it.f.uv, it.f.pitch, it.f.w, it.f.h, it.f.matrix, b.x, b.y, b.z, b.w};
        cp.req_slot[q] = q;
        cp.meta[q] = PoolMeta{it.cam, slot, gen, t_first};
        cp.tfirst_host[q] = t_first;
        if (cp.treq) { cp.treq[q] = t_req; cp.rcam[q] = (short)it.cam; }
        atomicAdd(nreq, 1);
        return true;
    }
    const unsigned long long h0 = *(volatile unsigned long long*)cp.head;
    if (h0 - *cp.released + (unsigned long long)cp.margin > (unsigned long long)cp.pool) return false;   // pool full
    const unsigned long long h = atomicAdd(cp.head, 1ULL);
    const int p = (int)(h % (unsigned long long)cp.pool);
    const int q = atomicAdd(cp.batch_count, 1);
    cp.req[q] = CropReq{it.f.y, it.f.uv, it.f.pitch, it.f.w, it.f.h, it.f.matrix, b.x, b.y, b.z, b.w};
    cp.req_slot[q] = p;
    cp.meta[p] = PoolMeta{it.cam, slot, gen, t_first};
    cp.tfirst_host[p] = t_first;
    atomicAdd(nreq, 1);
    return true;
}

__global__ void __launch_bounds__(256) track_update_kernel(const FrameDets* __restrict__ dets, const TrackBatch b,
                                                           TrackTable tt, TrackParams p, CropPool cp, TrackOut out,
                                                           const TickDesc* __restrict__ td) {
    __shared__ Shared sh;
    if (td && (int)blockIdx.x >= td->n) return;
    const TrackItem it = td ? td->trk[blockIdx.x] : b.it[blockIdx.x];
    const FrameDets& fd = dets[blockIdx.x];
    const int tid = threadIdx.x, N = tt.ncam * TRK, base = it.cam * TRK;
    const int n = min(fd.count, MAX_FACES);
    const int gap = max(it.gap, 1);

    // P0: detections into shared memory; compact the alive tracks; clear this camera's output rows for the gap
    if (tid == 0) { sh.m = 0; sh.nfree = 0; sh.nun = 0; sh.nreq = 0; }
    for (int d = tid; d < n; d += blockDim.x) { sh.det[d] = fd.box[d]; sh.dscore[d] = fd.score[d]; sh.mD[d] = -1; }
    for (int j = tid; j < TRK; j += blockDim.x) sh.mT[j] = -1;
    for (int e = tid; e < gap * TRK; e += blockDim.x) {
        const int r = e / TRK, s = e - r * TRK;
        out.id[((size_t)it.cam * OUT_RING + ring(it.fid - r)) * TRK + s] = 0;
    }
    __syncthreads();
    for (int s = tid; s < TRK; s += blockDim.x)
        if (tt.alive[base + s]) sh.tslot[atomicAdd(&sh.m, 1)] = s;
    __syncthreads();
    const int m = sh.m;

    // P1: thread j owns track j: load, predict to this frame (keeping each step for the smoother)
    KF st[4], pr[4];
    int slot = -1;
    float h0 = 4.f;
    if (tid < m) {
        slot = sh.tslot[tid];
        const int g = base + slot;
#pragma unroll
        for (int k = 0; k < 4; ++k) st[k] = KF{tt.x[k * N + g], tt.v[k * N + g], tt.P00[k * N + g], tt.P01[k * N + g], tt.P11[k * N + g]};
        h0 = fmaxf(st[3].x, 4.f);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float sa = (k < 2 ? p.sig_acc : p.sig_acc_size) * h0;
            pr[k] = st[k];
            for (int s = 0; s < gap; ++s) kf_predict(pr[k], it.dt, sa * sa);
        }
        sh.pb[tid] = to_box(pr[0].x, pr[1].x, pr[2].x, pr[3].x);
    }
    __syncthreads();

    // P2: IoU between every detection and every predicted track; P3: greedy match
    for (int e = tid; e < n * m; e += blockDim.x) {
        const int d = e / m, j = e - d * m;
        sh.C[d * TRK + j] = iou(sh.det[d], sh.pb[j]);
    }
    __syncthreads();
    greedy(sh, n, m, p.iou_thr);
    // P4: centre distance (in face sizes) for what IoU missed
    if (p.dist_k > 0.f) {
        for (int e = tid; e < n * m; e += blockDim.x) {
            const int d = e / m, j = e - d * m;
            const float4 a = sh.det[d], q = sh.pb[j];
            const float size = fmaxf(fmaxf(q.z - q.x, q.w - q.y), 1.f);
            const float rel = hypotf(.5f * (a.x + a.z - q.x - q.z), .5f * (a.y + a.w - q.y - q.w)) / size;
            sh.C[d * TRK + j] = rel <= p.dist_k ? -rel : -FLT_MAX;
        }
        __syncthreads();
        greedy(sh, n, m, -p.dist_k);
    }

    // P5: update matched tracks, smooth the skipped frames, age unmatched ones, decide recognition
    int rq_why = -1, rq_g = 0;
    float rq_treq = 0.f, rq_area = 0.f, rq_tfirst = -1.f;
    float4 rq_box = make_float4(0.f, 0.f, 0.f, 0.f);
    bool rq_aged = false;
    if (tid < m) {
        const int g = base + slot;
        const int d = sh.mT[tid];
        if (d >= 0) {
            const float4 db = sh.det[d];
            const float z[4] = {.5f * (db.x + db.z), .5f * (db.y + db.w), db.z - db.x, db.w - db.y};
            const float r = p.sig_meas * fmaxf(pr[3].x, 4.f);
            KF fin[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) { fin[k] = pr[k]; kf_update(fin[k], z[k], r * r); }
            const unsigned tidv = tt.tid[g];
            const int lab = tt.label[g];
            // Rauch-Tung-Striebel backward pass over the gap: each skipped frame uses the detections on both sides
            if (gap > 1 && gap <= GAPMAX) {
                float xs[4][GAPMAX];
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const float sa = (k < 2 ? p.sig_acc : p.sig_acc_size) * h0, qa = sa * sa;
                    KF f[GAPMAX + 1];
                    f[0] = st[k];
                    for (int s = 1; s <= gap; ++s) { f[s] = f[s - 1]; kf_predict(f[s], it.dt, qa); }
                    float x1 = fin[k].x, v1 = fin[k].v;
                    for (int s = gap - 1; s >= 1; --s) {
                        const KF& a = f[s];
                        const KF& p1 = f[s + 1];
                        const float A = a.P00 + it.dt * a.P01, B = a.P01, Cc = a.P01 + it.dt * a.P11, D = a.P11;
                        const float det = p1.P00 * p1.P11 - p1.P01 * p1.P01;
                        const float i00 = p1.P11 / det, i01 = -p1.P01 / det, i11 = p1.P00 / det;
                        const float dx = x1 - p1.x, dv = v1 - p1.v;
                        const float nx = a.x + (A * i00 + B * i01) * dx + (A * i01 + B * i11) * dv;
                        const float nv = a.v + (Cc * i00 + D * i01) * dx + (Cc * i01 + D * i11) * dv;
                        x1 = nx; v1 = nv;
                        xs[k][s] = nx;
                    }
                }
                for (int s = 1; s < gap; ++s) {
                    const size_t o = ((size_t)it.cam * OUT_RING + ring(it.fid - gap + s)) * TRK + slot;
                    out.box[o] = to_box(xs[0][s], xs[1][s], xs[2][s], xs[3][s]);
                    out.id[o] = (int)(tidv << 12) | (lab + 1);
                }
            }
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                tt.x[k * N + g] = fin[k].x; tt.v[k * N + g] = fin[k].v;
                tt.P00[k * N + g] = fin[k].P00; tt.P01[k * N + g] = fin[k].P01; tt.P11[k * N + g] = fin[k].P11;
            }
            tt.seen[g] = it.t;
            const int hits = tt.hits[g] + 1;
            tt.hits[g] = hits;
            const size_t o = ((size_t)it.cam * OUT_RING + ring(it.fid)) * TRK + slot;
            out.box[o] = db;
            out.id[o] = (int)(tidv << 12) | (lab + 1);

            // recognition triggers. Size from the Kalman-smoothed box (raw boxes on small faces jitter +-20%).
            const float area = fin[2].x * fin[3].x;
            // a possible ID swap only matters between two faces with different names, at least one of them known
            bool crossed = false;
            for (int j = 0; j < m && !crossed; ++j) {
                if (j == tid || iou(sh.pb[tid], sh.pb[j]) <= p.cross_iou) continue;
                const int other = tt.label[base + sh.tslot[j]];
                crossed = other != lab && (other >= 0 || lab >= 0);
            }
            const bool known = tt.known[g];
            const float rt = tt.rec_t[g], sc = tt.id_score[g];
            int why = -1;
            if (!known) { if (hits >= p.confirm_hits && (rt < 0.f || it.t - rt >= p.retry_s)) why = 0; }   // new face
            else if (area >= (1.f + p.grow) * tt.rec_area[g]) why = 1;                                      // closer view
            else if (it.t - rt >= p.recheck_s) why = 2;                                                     // periodic
            else if (sc >= p.unknown_thr && sc < p.match_thr && it.t - rt >= p.retry_s) why = 3;            // unsure
            else if (crossed && it.t - rt >= 1.0f) why = 4;                                                 // possible swap
            if (why >= 0 && db.z - db.x >= p.min_rec_px) {
                // decided here, served in P5b. t_req: capture time of the frame that first asked.
                rq_why = why; rq_g = g; rq_area = area; rq_box = db;
                rq_aged = tt.pend_t[g] >= 0.f;
                rq_treq = rq_aged ? tt.pend_t[g] : it.t;
                rq_tfirst = known ? -1.f : tt.first[g];
            } else {
                // no request this frame (no rule asks any more, or the face is below the size gate): nothing is waiting
                tt.pend_t[g] = -1.f;
            }
        } else if (it.t - tt.seen[g] > p.ttl) {
            if (tt.pend_t[g] >= 0.f) atomicAdd(out.missed, 1ULL);    // left before it could be recognised
            tt.alive[g] = 0;
        }
    }
    __syncthreads();

    // P5b: unified pipeline: list this face as a candidate (K5b hands out the slots globally, longest-waiting first).
    // Two-stream mode: take a slot in the crop pool directly.
    if (rq_why >= 0) {
        if (cp.cand) {
            const int ci = atomicAdd(cp.cand_count, 1);
            if (ci < MAX_DET_B * MAX_FACES)
                cp.cand[ci] = RecCand{rq_g, tt.gen[rq_g], (int)blockIdx.x, rq_why, rq_box, rq_tfirst, rq_treq, rq_area, it.t};
        } else if (request_recognition(cp, it, slot, tt.gen[rq_g], rq_box, rq_tfirst, rq_treq, cp.margin, &sh.nreq)) {
            tt.rec_t[rq_g] = it.t;
            tt.rec_area[rq_g] = rq_area;
            tt.pend_t[rq_g] = -1.f;
            atomicAdd(&out.reasons[rq_why], 1ULL);
        } else {
            tt.pend_t[rq_g] = rq_treq;
            if (cp.max_pend_ms) atomicMax(cp.max_pend_ms, (int)((it.t - rq_treq) * 1000.f));
        }
    }
    __syncthreads();

    // P6: new tracks for unmatched detections, in free slots
    for (int s = tid; s < TRK; s += blockDim.x)
        if (!tt.alive[base + s]) sh.freeL[atomicAdd(&sh.nfree, 1)] = s;
    for (int d = tid; d < n; d += blockDim.x)
        if (sh.mD[d] < 0) sh.unL[atomicAdd(&sh.nun, 1)] = d;
    __syncthreads();
    const int nnew = min(sh.nun, sh.nfree);
    for (int k = tid; k < nnew; k += blockDim.x) {
        const int s = sh.freeL[k], g = base + s;
        const float4 db = sh.det[sh.unL[k]];
        const float c[4] = {.5f * (db.x + db.z), .5f * (db.y + db.w), db.z - db.x, db.w - db.y};
        const float h = fmaxf(c[3], 4.f), r = p.sig_meas * h;
#pragma unroll
        for (int a = 0; a < 4; ++a) {
            tt.x[a * N + g] = c[a]; tt.v[a * N + g] = 0.f;
            tt.P00[a * N + g] = r * r; tt.P01[a * N + g] = 0.f; tt.P11[a * N + g] = (a < 2 ? 1.0f : 0.3f) * h * h;
        }
        const unsigned gen = tt.gen[g] + 1;
        tt.gen[g] = gen;
        // ids start at 1: an output id of 0 means "no box", so a track numbered 0 would never be delivered
        const unsigned tidv = atomicAdd(&tt.next_tid[it.cam], 1u) + 1u;
        tt.tid[g] = tidv;
        tt.alive[g] = 1; tt.known[g] = 0; tt.hits[g] = 1; tt.label[g] = -1;
        tt.seen[g] = it.t; tt.first[g] = it.t; tt.rec_t[g] = -1.f; tt.rec_area[g] = 0.f; tt.id_score[g] = 0.f;
        tt.pend_t[g] = -1.f;
        const size_t o = ((size_t)it.cam * OUT_RING + ring(it.fid)) * TRK + s;
        out.box[o] = db;
        out.id[o] = (int)(tidv << 12);
        if (p.confirm_hits <= 1 && db.z - db.x >= p.min_rec_px && cp.cand) {
            const int ci = atomicAdd(cp.cand_count, 1);
            if (ci < MAX_DET_B * MAX_FACES)
                cp.cand[ci] = RecCand{g, gen, (int)blockIdx.x, 0, db, it.t, it.t, (db.z - db.x) * (db.w - db.y), it.t};
        } else if (p.confirm_hits <= 1 && db.z - db.x >= p.min_rec_px &&
            request_recognition(cp, it, s, gen, db, it.t, it.t, cp.margin, &sh.nreq)) {
            atomicAdd(&out.reasons[0], 1ULL);
            tt.rec_t[g] = it.t;
            tt.rec_area[g] = (db.z - db.x) * (db.w - db.y);
        }
    }
    __syncthreads();
    if (tid == 0) {
        atomicAdd(&out.frames[it.cam], (unsigned long long)gap);
        atomicAdd(&out.faces[it.cam], (unsigned long long)n);
        atomicAdd(&out.requests[it.cam], (unsigned long long)sh.nreq);
    }
}

// K7: after this tick's recognitions, write what is now known into the rows this tick delivers.
// id = track id << 12 | recognised << 11 | (gallery index + 1); also counts delivered boxes and named boxes.
__global__ void finalize_output_kernel(const TickDesc* __restrict__ td, TrackTable tt, TrackOut out, TickStats* st) {
    if ((int)blockIdx.x >= td->n) return;
    const TrackItem it = td->trk[blockIdx.x];
    const int gap = min(max(it.gap, 1), OUT_RING), base = it.cam * TRK;
    const uint8_t* zm = td->zones.m[blockIdx.x];      // this camera's zone label image (frame-sized), may be null
    const int zpitch = td->zones.pitch;
    int boxes = 0, named = 0;
    for (int e = threadIdx.x; e < gap * TRK; e += blockDim.x) {
        const int r = e / TRK, s = e - r * TRK;
        const size_t o = ((size_t)it.cam * OUT_RING + ring(it.fid - r)) * TRK + s;
        const int id = out.id[o];
        if (!id) continue;
        ++boxes;
        // zone of the delivered box, detected or Kalman-filled alike: the label under the box centre
        unsigned char z = 0;
        if (zm) {
            const float4 b = out.box[o];
            const int cx = min(max((int)(0.5f * (b.x + b.z)), 0), it.f.w - 1);
            const int cy = min(max((int)(0.5f * (b.y + b.w)), 0), it.f.h - 1);
            z = __ldg(zm + (size_t)cy * zpitch + cx);
        }
        out.zone[o] = z;
        const unsigned tid = (unsigned)id >> 12;
        const int g = base + s;
        if (tt.alive[g] && (tt.tid[g] & 0xfffffu) == tid) {
            const int known = tt.known[g];
            named += known;
            out.id[o] = (int)(tid << 12) | (known << 11) | ((tt.label[g] + 1) & 0x7ff);
        }
    }
    if (boxes) atomicAdd(&st->boxes, boxes);
    if (named) atomicAdd(&st->named, named);
}

// K8 (live mode only): one block per frame of the tick; rows beyond the frame's gap are written empty.
__global__ void gather_rows_kernel(const TickDesc* __restrict__ td, TrackOut out, LiveBox* __restrict__ dst) {
    if ((int)blockIdx.x >= td->n) return;
    const TrackItem it = td->trk[blockIdx.x];
    const int gap = min(max(it.gap, 1), LIVE_ROWS);
    for (int e = threadIdx.x; e < LIVE_ROWS * TRK; e += blockDim.x) {
        const int r = e / TRK, s = e - r * TRK;
        LiveBox lb;
        lb.box = make_float4(0.f, 0.f, 0.f, 0.f);
        lb.id = 0;
        lb.zone = 0;
        if (r < gap) {
            const size_t o = ((size_t)it.cam * OUT_RING + ring(it.fid - r)) * TRK + s;
            lb.id = out.id[o];
            if (lb.id) { lb.box = out.box[o]; lb.zone = out.zone[o]; }
        }
        dst[((size_t)blockIdx.x * LIVE_ROWS + r) * TRK + s] = lb;
    }
}

// Enrolment: one block. Finds the accepted recognition of (cam, tid) in this tick, L2-normalises its embedding into
// gallery row `label`, and names the track straight away.
__global__ void enroll_kernel(const PoolMeta* __restrict__ meta, const int* __restrict__ count, int maxn,
                              const __half* __restrict__ emb, TrackTable tt, int cam, unsigned tid, __half* gallery,
                              int label, int* done) {
    __shared__ int s_k;
    __shared__ float s_sum[256];
    if (threadIdx.x == 0) s_k = -1;
    __syncthreads();
    const int n = min(*count, maxn);
    for (int k = threadIdx.x; k < n; k += blockDim.x) {
        const PoolMeta m = meta[k];
        if (m.cam != cam) continue;
        const int g = m.cam * TRK + m.slot;
        if (tt.alive[g] && tt.gen[g] == m.gen && (tt.tid[g] & 0xfffffu) == (tid & 0xfffffu)) atomicMax(&s_k, k);
    }
    __syncthreads();
    const int k = s_k;
    if (k < 0) return;
    float acc = 0.f;
    for (int i = threadIdx.x; i < EMB; i += blockDim.x) { const float v = __half2float(emb[(size_t)k * EMB + i]); acc += v * v; }
    s_sum[threadIdx.x] = acc;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) {
        if ((int)threadIdx.x < w) s_sum[threadIdx.x] += s_sum[threadIdx.x + w];
        __syncthreads();
    }
    const float inv = rsqrtf(fmaxf(s_sum[0], 1e-12f));
    for (int i = threadIdx.x; i < EMB; i += blockDim.x)
        gallery[(size_t)label * EMB + i] = __float2half(__half2float(emb[(size_t)k * EMB + i]) * inv);
    if (threadIdx.x == 0) {
        const int g = meta[k].cam * TRK + meta[k].slot;
        tt.known[g] = 1;
        tt.label[g] = label;
        tt.id_score[g] = 1.f;
        *done = 1;
    }
}

__global__ void __launch_bounds__(256) assign_slots_kernel(const TickDesc* __restrict__ td, TrackTable tt, CropPool cp,
                                                           TrackOut out) {
    __shared__ int cnt;
    const int tid = threadIdx.x, cap = cp.margin;
    const int nc = min(*cp.cand_count, MAX_DET_B * MAX_FACES);
    if (tid == 0) cnt = 0;
    __syncthreads();
    for (int tier = 0; tier < 3; ++tier) {
        for (int i = tid; i < nc; i += blockDim.x) {
            const RecCand c = cp.cand[i];
            const float age = c.t - c.treq;
            if ((age > 0.3f ? 0 : (age > 0.f ? 1 : 2)) != tier) continue;
            const int q = atomicAdd(&cnt, 1);
            const int cam = td->trk[c.frame].cam;
            if (q < cap) {
                const FrameRef f = td->trk[c.frame].f;
                cp.req[q] = CropReq{f.y, f.uv, f.pitch, f.w, f.h, f.matrix, c.box.x, c.box.y, c.box.z, c.box.w};
                cp.req_slot[q] = q;
                cp.meta[q] = PoolMeta{cam, c.g - cam * TRK, c.gen, c.tfirst};
                cp.tfirst_host[q] = c.tfirst;
                cp.treq[q] = c.treq;
                cp.rcam[q] = (short)cam;
                tt.rec_t[c.g] = c.t;
                tt.rec_area[c.g] = c.area;
                tt.pend_t[c.g] = -1.f;
                atomicAdd(&out.reasons[c.why], 1ULL);
                atomicAdd(&out.requests[cam], 1ULL);
            } else {
                tt.pend_t[c.g] = c.treq;                    // asks again, with priority, on its next detected frame
                atomicMax(cp.max_pend_ms, (int)(age * 1000.f));
                atomicAdd(cp.denied, 1);
            }
        }
        __syncthreads();
        if (tid == 0) cnt = min(cnt, cap);
        __syncthreads();
    }
    if (tid == 0) *cp.batch_count = cnt;
}

__global__ void publish_kernel(const int* dev_count, int* host_count) { *host_count = *dev_count; }

__global__ void scatter_identity_kernel(const MatchOut* __restrict__ m, int n, int start, CropPool cp, TrackTable tt,
                                        float match_thr, const int* valid) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n || (valid && k >= *valid)) return;
    const PoolMeta pm = cp.meta[cp.pool ? (start + k) % cp.pool : k];
    const int g = pm.cam * TRK + pm.slot;
    if (tt.gen[g] != pm.gen) return;              // slot reused by a newer track meanwhile
    const MatchOut r = m[k];
    tt.label[g] = r.score >= match_thr ? r.idx : -1;
    tt.id_score[g] = r.score;
    tt.known[g] = 1;
}

}  // namespace

void launch_track_update(const FrameDets* dets, const TrackBatch& b, TrackTable tt, TrackParams p, CropPool cp,
                         TrackOut out, cudaStream_t s) {
    if (b.n <= 0) return;
    track_update_kernel<<<b.n, 256, 0, s>>>(dets, b, tt, p, cp, out, nullptr);
}

void launch_publish_count(const int* dev_count, int* host_count, cudaStream_t s) {
    publish_kernel<<<1, 1, 0, s>>>(dev_count, host_count);
}

void launch_scatter_identity(const MatchOut* m, int n, int pool_start, CropPool cp, TrackTable tt, float match_thr,
                             const int* valid, cudaStream_t s) {
    if (n <= 0) return;
    scatter_identity_kernel<<<(n + 127) / 128, 128, 0, s>>>(m, n, pool_start, cp, tt, match_thr, valid);
}

void launch_track_update_u(const FrameDets* dets, int B, const TickDesc* d, TrackTable tt, TrackParams p, CropPool cp,
                           TrackOut out, cudaStream_t s) {
    track_update_kernel<<<B, 256, 0, s>>>(dets, TrackBatch{}, tt, p, cp, out, d);
}

void launch_finalize_output(const TickDesc* d, int B, TrackTable tt, TrackOut out, TickStats* st, cudaStream_t s) {
    finalize_output_kernel<<<B, 256, 0, s>>>(d, tt, out, st);
}

void launch_gather_rows(const TickDesc* d, int B, TrackOut out, LiveBox* dst, cudaStream_t s) {
    gather_rows_kernel<<<B, 256, 0, s>>>(d, out, dst);
}

void launch_enroll(const PoolMeta* meta, const int* count, int maxn, const __half* emb, TrackTable tt, int cam,
                   unsigned tid, __half* gallery, int label, int* done, cudaStream_t s) {
    enroll_kernel<<<1, 256, 0, s>>>(meta, count, maxn, emb, tt, cam, tid, gallery, label, done);
}

void launch_assign_slots(const TickDesc* d, TrackTable tt, CropPool cp, TrackOut out, cudaStream_t s) {
    assign_slots_kernel<<<1, 256, 0, s>>>(d, tt, cp, out);
}
