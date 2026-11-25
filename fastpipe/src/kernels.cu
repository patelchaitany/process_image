// Fused pre/post-processing kernels. Design notes: ARCHITECTURE.md section 4.
#include <cfloat>

#include "gputrack.cuh"
#include "kernels.cuh"

namespace {

constexpr unsigned FULL = 0xffffffffu;

// Limited-range YUV -> RGB in [0,255]; matrix 0 = BT.601, 1 = BT.709 (warp-uniform: one camera per frame)
__device__ __forceinline__ void yuv2rgb(float Y, float U, float V, float& r, float& g, float& b, int matrix) {
    const float c = 1.164383f * (Y - 16.f);
    const float d = U - 128.f, e = V - 128.f;
    const float rv = matrix ? 1.792741f : 1.596027f;
    const float gu = matrix ? -0.213249f : -0.391762f;
    const float gv = matrix ? -0.532909f : -0.812968f;
    const float bu = matrix ? 2.112402f : 2.017232f;
    r = fmaf(rv, e, c);
    g = fmaf(gv, e, fmaf(gu, d, c));
    b = fmaf(bu, d, c);
}

__device__ __forceinline__ float sat01(float x) { return fminf(fmaxf(x, 0.f), 1.f); }

// Bilinear sample of the Y plane and of the interleaved UV plane at full-resolution coordinates (fx, fy)
__device__ __forceinline__ void sample_nv12(const uint8_t* __restrict__ yp, const uint8_t* __restrict__ uvp,
                                            int pitch, int w, int h, float fx, float fy,
                                            float& Y, float& U, float& V) {
    fx = fminf(fmaxf(fx, 0.f), (float)(w - 1));
    fy = fminf(fmaxf(fy, 0.f), (float)(h - 1));
    int x0 = (int)fx, y0 = (int)fy;
    int x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
    float ax = fx - x0, ay = fy - y0;
    float t0 = __ldg(yp + (size_t)y0 * pitch + x0), t1 = __ldg(yp + (size_t)y0 * pitch + x1);
    float b0 = __ldg(yp + (size_t)y1 * pitch + x0), b1 = __ldg(yp + (size_t)y1 * pitch + x1);
    Y = fmaf(ay, fmaf(ax, b1 - b0, b0) - fmaf(ax, t1 - t0, t0), fmaf(ax, t1 - t0, t0));
    // chroma is subsampled 2x2; sample centres sit between luma pairs
    const int cw = w >> 1, chh = h >> 1;
    float cx = fminf(fmaxf((fx + 0.5f) * 0.5f - 0.5f, 0.f), (float)(cw - 1));
    float cy = fminf(fmaxf((fy + 0.5f) * 0.5f - 0.5f, 0.f), (float)(chh - 1));
    int u0 = (int)cx, v0 = (int)cy;
    int u1 = min(u0 + 1, cw - 1), v1 = min(v0 + 1, chh - 1);
    float bx = cx - u0, by = cy - v0;
    const uchar2 p00 = __ldg(reinterpret_cast<const uchar2*>(uvp + (size_t)v0 * pitch) + u0);
    const uchar2 p01 = __ldg(reinterpret_cast<const uchar2*>(uvp + (size_t)v0 * pitch) + u1);
    const uchar2 p10 = __ldg(reinterpret_cast<const uchar2*>(uvp + (size_t)v1 * pitch) + u0);
    const uchar2 p11 = __ldg(reinterpret_cast<const uchar2*>(uvp + (size_t)v1 * pitch) + u1);
    float ut = fmaf(bx, (float)p01.x - p00.x, p00.x), ub = fmaf(bx, (float)p11.x - p10.x, p10.x);
    float vt = fmaf(bx, (float)p01.y - p00.y, p00.y), vb = fmaf(bx, (float)p11.y - p10.y, p10.y);
    U = fmaf(by, ub - ut, ut);
    V = fmaf(by, vb - vt, vt);
}

// ------------------------------------------------------------------------------------------------ K1
// One thread = 8 horizontally adjacent output pixels. Warp = 256 contiguous pixels of one output row, so each
// plane store is 512 contiguous bytes (four full 128-byte lines).
__global__ void __launch_bounds__(128) prep_det_kernel(const DetBatch b, const TickDesc* __restrict__ td,
                                                       __half* __restrict__ out) {
    const int i = blockIdx.z;
    if (i >= (td ? td->det.n : b.n)) return;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    const int x0 = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (oy >= DET_H || x0 >= DET_W) return;
    const FrameRef f = td ? td->det.f[i] : b.f[i];     // unified pipeline: per-batch parameters come from device memory
    const Letterbox lb = td ? td->det.lb : b.lb;
    constexpr int PLANE = DET_H * DET_W;
    constexpr float K = 1.f / 255.f;
    __align__(16) __half R[8], G[8], B[8];

    const bool fast = lb.scale == 1.f && lb.pad_x == 0.f && f.w == DET_W;
    if (fast) {
        const int sy = oy - (int)lb.pad_y;
        if (sy < 0 || sy >= f.h) {
            const __half p = __float2half(114.f / 255.f);
#pragma unroll
            for (int j = 0; j < 8; ++j) R[j] = G[j] = B[j] = p;
        } else {
            const uint2 yv = __ldg(reinterpret_cast<const uint2*>(f.y + (size_t)sy * f.pitch + x0));
            const uint2 cv = __ldg(reinterpret_cast<const uint2*>(f.uv + (size_t)(sy >> 1) * f.pitch + x0));
            const uint8_t* yb = reinterpret_cast<const uint8_t*>(&yv);
            const uint8_t* cb = reinterpret_cast<const uint8_t*>(&cv);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                float r, g, bb;
                yuv2rgb(yb[j], cb[j & ~1], cb[(j & ~1) + 1], r, g, bb, f.matrix);
                R[j] = __float2half(sat01(r * K));
                G[j] = __float2half(sat01(g * K));
                B[j] = __float2half(sat01(bb * K));
            }
        }
    } else {
        // general path: letterbox + bilinear resize from any source resolution
        const float cy0 = lb.pad_y, cy1 = lb.pad_y + f.h * lb.scale;
        const float cx0 = lb.pad_x, cx1 = lb.pad_x + f.w * lb.scale;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float ox = x0 + j + 0.5f, oyc = oy + 0.5f;
            if (ox < cx0 || ox >= cx1 || oyc < cy0 || oyc >= cy1) {
                R[j] = G[j] = B[j] = __float2half(114.f / 255.f);
                continue;
            }
            float Y, U, V, r, g, bb;
            sample_nv12(f.y, f.uv, f.pitch, f.w, f.h, (ox - lb.pad_x) / lb.scale - 0.5f,
                        (oyc - lb.pad_y) / lb.scale - 0.5f, Y, U, V);
            yuv2rgb(Y, U, V, r, g, bb, f.matrix);
            R[j] = __float2half(sat01(r * K));
            G[j] = __float2half(sat01(g * K));
            B[j] = __float2half(sat01(bb * K));
        }
    }
    __half* o = out + (size_t)i * 3 * PLANE + (size_t)oy * DET_W + x0;
    *reinterpret_cast<uint4*>(o) = *reinterpret_cast<const uint4*>(R);
    *reinterpret_cast<uint4*>(o + PLANE) = *reinterpret_cast<const uint4*>(G);
    *reinterpret_cast<uint4*>(o + 2 * PLANE) = *reinterpret_cast<const uint4*>(B);
}

// ------------------------------------------------------------------------------------------------ K2
// One block per image. Candidates live in shared memory as structure-of-arrays (no bank conflicts).
constexpr int NMS_WORDS = MAX_CAND / 32;

__global__ void __launch_bounds__(256) decode_nms_kernel(const __half* __restrict__ det, int A, int chw,
                                                         float conf_thr, float iou_thr, Letterbox lb,
                                                         const ZoneMasks zones, FrameDets* __restrict__ out,
                                                         const TickDesc* __restrict__ td) {
    __shared__ float sx1[MAX_CAND], sy1[MAX_CAND], sx2[MAX_CAND], sy2[MAX_CAND];
    __shared__ float key[MAX_CAND];
    __shared__ short val[MAX_CAND];
    __shared__ uint32_t mask[MAX_CAND * NMS_WORDS];   // 32 KB: suppression bitmask, row i = boxes i suppresses
    __shared__ short keep[MAX_FACES];
    __shared__ int s_count, s_keep;

    const int img = blockIdx.x, tid = threadIdx.x, lane = tid & 31;
    if (td) {
        if (img >= td->det.n) return;
        lb = td->det.lb;
    }
    const __half* d = det + (size_t)img * 5 * A;
    if (tid == 0) s_count = 0;
    __syncthreads();

    // 1. threshold + warp-aggregated compaction: one shared atomic per warp instead of per thread
    for (int base = 0; base < A; base += blockDim.x) {
        const int a = base + tid;
        float c = 0.f;
        if (a < A) c = __half2float(chw ? d[4 * A + a] : d[a * 5 + 4]);
        const bool pass = a < A && c > conf_thr;
        const unsigned bal = __ballot_sync(FULL, pass);
        if (bal) {
            int wb = 0;
            if (lane == 0) wb = atomicAdd(&s_count, __popc(bal));
            wb = __shfl_sync(FULL, wb, 0);
            if (pass) {
                const int pos = wb + __popc(bal & ((1u << lane) - 1u));
                if (pos < MAX_CAND) {
                    float cx, cy, w, h;
                    if (chw) {
                        cx = __half2float(d[a]); cy = __half2float(d[A + a]);
                        w = __half2float(d[2 * A + a]); h = __half2float(d[3 * A + a]);
                    } else {
                        cx = __half2float(d[a * 5]); cy = __half2float(d[a * 5 + 1]);
                        w = __half2float(d[a * 5 + 2]); h = __half2float(d[a * 5 + 3]);
                    }
                    const float inv = 1.f / lb.scale;
                    sx1[pos] = (cx - 0.5f * w - lb.pad_x) * inv;
                    sy1[pos] = (cy - 0.5f * h - lb.pad_y) * inv;
                    sx2[pos] = (cx + 0.5f * w - lb.pad_x) * inv;
                    sy2[pos] = (cy + 0.5f * h - lb.pad_y) * inv;
                    key[pos] = c;
                    val[pos] = (short)pos;
                }
            }
        }
    }
    __syncthreads();
    const int n = min(s_count, MAX_CAND);
    FrameDets* o = out + img;
    if (n == 0) {
        if (tid == 0) { o->count = 0; o->n_cand = 0; }
        return;
    }

    // 2. bitonic sort, descending by score (ties by index). Strides >= 32 in shared memory, strides <= 16 in
    //    registers with warp shuffles (elements e and e^j are in the same warp because e's low 5 bits = lane).
    int P = 32;
    while (P < n) P <<= 1;
    for (int i = n + tid; i < P; i += blockDim.x) { key[i] = -FLT_MAX; val[i] = (short)i; }
    __syncthreads();
    for (int k = 2; k <= P; k <<= 1) {
        for (int j = k >> 1; j >= 32; j >>= 1) {
            for (int i = tid; i < P; i += blockDim.x) {
                const int ixj = i ^ j;
                if (ixj > i) {
                    const float ki = key[i], kj = key[ixj];
                    const short vi = val[i], vj = val[ixj];
                    const bool i_less = ki < kj || (ki == kj && vi > vj);
                    if (((i & k) == 0) == i_less) {
                        key[i] = kj; key[ixj] = ki; val[i] = vj; val[ixj] = vi;
                    }
                }
            }
            __syncthreads();
        }
        for (int e = tid; e < P; e += blockDim.x) {     // warp-uniform: P is a multiple of 32
            float kk = key[e];
            int vv = val[e];
            for (int j = min(k >> 1, 16); j > 0; j >>= 1) {
                const float pk = __shfl_xor_sync(FULL, kk, j);
                const int pv = __shfl_xor_sync(FULL, vv, j);
                const bool want_greater = ((e & j) == 0) == ((e & k) == 0);
                const bool partner_greater = pk > kk || (pk == kk && pv < vv);
                if (want_greater == partner_greater) { kk = pk; vv = pv; }
            }
            key[e] = kk;
            val[e] = (short)vv;
        }
        __syncthreads();
    }

    // 3. suppression bitmask: bit (i, j) set if j comes after i in score order and IoU(i, j) > thr
    const int nw = (n + 31) >> 5;
    for (int t = tid; t < n * nw; t += blockDim.x) {
        const int i = t / nw, w = t - i * nw;
        const int bi = val[i];
        const float ax1 = sx1[bi], ay1 = sy1[bi], ax2 = sx2[bi], ay2 = sy2[bi];
        const float aa = (ax2 - ax1) * (ay2 - ay1);
        uint32_t bits = 0;
        const int j0 = w << 5;
        for (int bb = 0; bb < 32; ++bb) {
            const int j = j0 + bb;
            if (j <= i || j >= n) continue;
            const int bj = val[j];
            const float iw = fminf(ax2, sx2[bj]) - fmaxf(ax1, sx1[bj]);
            const float ih = fminf(ay2, sy2[bj]) - fmaxf(ay1, sy1[bj]);
            if (iw <= 0.f || ih <= 0.f) continue;
            const float inter = iw * ih;
            const float ab = (sx2[bj] - sx1[bj]) * (sy2[bj] - sy1[bj]);
            if (inter > iou_thr * (aa + ab - inter)) bits |= 1u << bb;
        }
        mask[i * NMS_WORDS + w] = bits;
    }
    __syncthreads();

    // 4. one warp scans in score order; lane l holds word l of the "removed" mask
    if (tid < 32) {
        uint32_t removed = 0;
        int kept = 0;
        for (int i = 0; i < n && kept < MAX_FACES; ++i) {
            const uint32_t word = __shfl_sync(FULL, removed, i >> 5);
            if (!((word >> (i & 31)) & 1u)) {
                if (lane == 0) keep[kept] = (short)i;
                ++kept;
                if (lane < nw) removed |= mask[i * NMS_WORDS + lane];
            }
        }
        if (lane == 0) s_keep = kept;
    }
    __syncthreads();

    // 5. compact output (coalesced float4 stores into host-mapped memory) + zone lookup at the face centre
    const int kn = s_keep;
    const uint8_t* zm = td ? td->zones.m[img] : zones.m[img];
    const int zpitch = td ? td->zones.pitch : zones.pitch;
    for (int t = tid; t < kn; t += blockDim.x) {
        const int bi = val[keep[t]];
        const float4 b = make_float4(fminf(fmaxf(sx1[bi], 0.f), (float)lb.w), fminf(fmaxf(sy1[bi], 0.f), (float)lb.h),
                                     fminf(fmaxf(sx2[bi], 0.f), (float)lb.w), fminf(fmaxf(sy2[bi], 0.f), (float)lb.h));
        o->box[t] = b;
        o->score[t] = key[keep[t]];
        if (zm) {
            const int cx = min(lb.w - 1, (int)(0.5f * (b.x + b.z))), cy = min(lb.h - 1, (int)(0.5f * (b.y + b.w)));
            o->zone[t] = __ldg(zm + (size_t)cy * zpitch + cx);
        } else {
            o->zone[t] = 0;
        }
    }
    if (tid == 0) { o->count = kn; o->n_cand = s_count; }
}

// ------------------------------------------------------------------------------------------------ K3
// One block per face. Each thread makes 2 adjacent output pixels (half2 stores, 128 B per warp per plane).
__device__ __forceinline__ void crop_one(const CropReq& r, __half* __restrict__ o) {
    // integer crop window, as the reference does (truncate then clamp)
    const int cx1 = max(0, (int)r.x1), cy1 = max(0, (int)r.y1);
    const int cx2 = min(r.w, (int)r.x2), cy2 = min(r.h, (int)r.y2);
    const float cw = (float)max(1, cx2 - cx1), ch = (float)max(1, cy2 - cy1);
    const float sx = cw / REC, sy = ch / REC;
    constexpr int PL = REC * REC;
    constexpr float NK = 1.f / 127.5f;
    for (int p2 = threadIdx.x; p2 < PL / 2; p2 += blockDim.x) {
        const int p = p2 * 2, oy = p / REC, ox = p - oy * REC;
        float rr[2], gg[2], bb[2];
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            // cv2.INTER_LINEAR source coordinate, clamped to the crop window (border replicate)
            float fx = fminf(fmaxf((ox + q + 0.5f) * sx - 0.5f, 0.f), cw - 1.f);
            float fy = fminf(fmaxf((oy + 0.5f) * sy - 0.5f, 0.f), ch - 1.f);
            float Y, U, V;
            sample_nv12(r.y, r.uv, r.pitch, r.w, r.h, cx1 + fx, cy1 + fy, Y, U, V);
            yuv2rgb(Y, U, V, rr[q], gg[q], bb[q], r.matrix);
            rr[q] = (fminf(fmaxf(rr[q], 0.f), 255.f) - 127.5f) * NK;
            gg[q] = (fminf(fmaxf(gg[q], 0.f), 255.f) - 127.5f) * NK;
            bb[q] = (fminf(fmaxf(bb[q], 0.f), 255.f) - 127.5f) * NK;
        }
        *reinterpret_cast<__half2*>(o + p) = __floats2half2_rn(rr[0], rr[1]);
        *reinterpret_cast<__half2*>(o + PL + p) = __floats2half2_rn(gg[0], gg[1]);
        *reinterpret_cast<__half2*>(o + 2 * PL + p) = __floats2half2_rn(bb[0], bb[1]);
    }
}

__global__ void __launch_bounds__(256) crop_prep_kernel(const CropReq* __restrict__ reqs, __half* __restrict__ out) {
    __shared__ CropReq r;
    if (threadIdx.x == 0) r = reqs[blockIdx.x];
    __syncthreads();
    crop_one(r, out + (size_t)blockIdx.x * 3 * REC * REC);
}

// Crops queued by the GPU tracker, written straight into their crop-pool slots (grid sized for the worst case;
// blocks beyond this batch's device-side count exit at once)
__global__ void __launch_bounds__(256) crop_to_pool_kernel(CropPool cp, __half* __restrict__ pool_tensor) {
    if ((int)blockIdx.x >= *cp.batch_count) return;
    __shared__ CropReq r;
    __shared__ int slot;
    if (threadIdx.x == 0) { r = cp.req[blockIdx.x]; slot = cp.req_slot[blockIdx.x]; }
    __syncthreads();
    crop_one(r, pool_tensor + (size_t)slot * 3 * REC * REC);
}

// ------------------------------------------------------------------------------------------------ K4
// One warp per embedding; lane holds 16 dims in registers. Gallery rows are read as 1 KB coalesced warp loads
// (the whole gallery stays in L2). Running arg-max in registers: no similarity matrix is written.
__device__ __forceinline__ void unpack16(const uint4& a, const uint4& b, float2 (&v)[8]) {
    const __half2* ha = reinterpret_cast<const __half2*>(&a);
    const __half2* hb = reinterpret_cast<const __half2*>(&b);
#pragma unroll
    for (int k = 0; k < 4; ++k) { v[k] = __half22float2(ha[k]); v[k + 4] = __half22float2(hb[k]); }
}

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(FULL, x, o);
    return x;
}

__global__ void __launch_bounds__(256) match_kernel(const __half* __restrict__ emb, int n,
                                                    const __half* __restrict__ gal, int G,
                                                    MatchOut* __restrict__ out) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (warp >= n) return;
    const uint4* e4 = reinterpret_cast<const uint4*>(emb + (size_t)warp * EMB) + lane * 2;
    float2 v[8];
    unpack16(__ldg(e4), __ldg(e4 + 1), v);
    float ss = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) ss = fmaf(v[k].x, v[k].x, fmaf(v[k].y, v[k].y, ss));
    const float inv = rsqrtf(fmaxf(warp_sum(ss), 1e-12f));
    float best = -FLT_MAX;
    int bi = -1;
    for (int g = 0; g < G; g += 2) {      // two rows per iteration for memory-level parallelism
        const uint4* g0 = reinterpret_cast<const uint4*>(gal + (size_t)g * EMB) + lane * 2;
        const bool two = g + 1 < G;
        const uint4* g1 = two ? g0 + EMB / 8 : g0;
        float2 a[8], b[8];
        unpack16(__ldg(g0), __ldg(g0 + 1), a);
        unpack16(__ldg(g1), __ldg(g1 + 1), b);
        float d0 = 0.f, d1 = 0.f;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            d0 = fmaf(v[k].x, a[k].x, fmaf(v[k].y, a[k].y, d0));
            d1 = fmaf(v[k].x, b[k].x, fmaf(v[k].y, b[k].y, d1));
        }
        d0 = warp_sum(d0);
        d1 = warp_sum(d1);
        if (d0 > best) { best = d0; bi = g; }
        if (two && d1 > best) { best = d1; bi = g + 1; }
    }
    if (lane == 0) out[warp] = MatchOut{bi, best * inv};
}

}  // namespace

void launch_prep_det(const DetBatch& b, __half* out, cudaStream_t s) {
    if (b.n <= 0) return;
    dim3 block(32, 4);
    dim3 grid((DET_W / 8 + 31) / 32, (DET_H + 3) / 4, b.n);
    prep_det_kernel<<<grid, block, 0, s>>>(b, nullptr, out);
}

void launch_decode_nms(const __half* det_out, int n, int A, int layout_chw, float conf_thr, float iou_thr,
                       Letterbox lb, const ZoneMasks& zones, FrameDets* out, cudaStream_t s) {
    if (n <= 0) return;
    decode_nms_kernel<<<n, 256, 0, s>>>(det_out, A, layout_chw, conf_thr, iou_thr, lb, zones, out, nullptr);
}

void launch_crop_prep(const CropReq* reqs, int n, __half* out, cudaStream_t s) {
    if (n <= 0) return;
    crop_prep_kernel<<<n, 256, 0, s>>>(reqs, out);
}

void launch_match(const __half* emb, int n, const __half* gallery, int G, MatchOut* out, cudaStream_t s) {
    if (n <= 0) return;
    match_kernel<<<(n * 32 + 255) / 256, 256, 0, s>>>(emb, n, gallery, G, out);
}

void launch_crop_to_pool(CropPool cp, int max_req, __half* pool_tensor, cudaStream_t s) {
    if (max_req <= 0) return;
    crop_to_pool_kernel<<<max_req, 256, 0, s>>>(cp, pool_tensor);
}

// Unified pipeline: grid sized for the batch bucket B; blocks beyond the tick's frame count (device-side) exit at once
void launch_prep_det_u(const TickDesc* d, int B, __half* out, cudaStream_t s) {
    dim3 block(32, 4);
    dim3 grid((DET_W / 8 + 31) / 32, (DET_H + 3) / 4, B);
    prep_det_kernel<<<grid, block, 0, s>>>(DetBatch{}, d, out);
}

void launch_decode_nms_u(const __half* det_out, int B, int A, int layout_chw, float conf_thr, float iou_thr,
                         const TickDesc* d, FrameDets* out, cudaStream_t s) {
    decode_nms_kernel<<<B, 256, 0, s>>>(det_out, A, layout_chw, conf_thr, iou_thr, Letterbox{}, ZoneMasks{}, out, d);
}
