#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

// Model geometry
constexpr int DET_W = 640;        // detector input width
constexpr int DET_H = 384;        // detector input height (16:9 frame letterboxed to a multiple of 32)
constexpr int MAX_DET_B = 32;     // max frames per detector batch
constexpr int MAX_FACES = 64;     // max faces kept per frame after NMS
constexpr int MAX_CAND = 512;     // max candidates per frame entering NMS
constexpr int REC = 112;          // recogniser input size
constexpr int EMB = 512;          // embedding size

// A decoded NV12 frame in device memory (pitch-linear, Y plane then interleaved UV plane)
struct FrameRef {
    const uint8_t* y;
    const uint8_t* uv;
    int pitch;
    int w, h;
    int matrix;      // YUV->RGB matrix from the stream: 0 = BT.601, 1 = BT.709 (both limited range)
};

// Letterbox geometry shared by K1 (forward) and K2 (inverse)
struct Letterbox {
    float scale;     // frame px -> detector px
    float pad_x;     // detector px of padding on the left
    float pad_y;     // detector px of padding on the top
    int w, h;        // frame size
};

// Passed by value as a kernel parameter (about 1 KB): no host->device copy per batch
struct DetBatch {
    FrameRef f[MAX_DET_B];
    Letterbox lb;
    int n;
};

// Per-frame detection results, written by K2 straight into host-mapped pinned memory
struct FrameDets {
    int count;
    int n_cand;          // candidates above threshold (before NMS), for diagnostics
    int _pad[2];
    float4 box[MAX_FACES];   // x1, y1, x2, y2 in frame pixels
    float score[MAX_FACES];
    uint8_t zone[MAX_FACES]; // zone id from the camera's zone mask at the face centre (0 = no zone)
};

// Per-frame zone masks (one uint8 label image per camera, frame-sized): "camera frame split into sections"
struct ZoneMasks {
    const uint8_t* m[MAX_DET_B];
    int pitch;
};

// One face crop request (written by the CPU tracker into host-mapped memory, read once per block by K3)
struct CropReq {
    const uint8_t* y;
    const uint8_t* uv;
    int pitch;
    int w, h;
    int matrix;
    float x1, y1, x2, y2;
};

struct MatchOut {
    int idx;
    float score;
};

// K1: NV12 -> RGB -> letterbox (and resize if needed) -> /255 -> FP16 planar [B,3,DET_H,DET_W]
void launch_prep_det(const DetBatch& b, __half* out, cudaStream_t s);

// K2: score filter -> box decode -> sort -> NMS -> compact per-frame results
// det_out: FP16 [B,5,A] (channel-first, layout_chw=1) or [B,A,5] (layout_chw=0); cx,cy,w,h,conf in detector px
void launch_decode_nms(const __half* det_out, int n, int A, int layout_chw, float conf_thr, float iou_thr,
                       Letterbox lb, const ZoneMasks& zones, FrameDets* out, cudaStream_t s);

// K3: crop -> bilinear resize to 112x112 -> NV12->RGB -> (x-127.5)/127.5 -> FP16 planar [n,3,112,112]
void launch_crop_prep(const CropReq* reqs, int n, __half* out, cudaStream_t s);

// K4: L2-normalise embeddings, cosine vs L2-normalised gallery, arg-max per embedding
void launch_match(const __half* emb, int n, const __half* gallery, int G, MatchOut* out, cudaStream_t s);
