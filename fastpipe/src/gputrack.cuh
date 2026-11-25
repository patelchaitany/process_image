#pragma once
// GPU-resident tracking and recognition scheduling (ARCHITECTURE.md section 10).
// After detection, nothing about a frame goes back to the CPU: Kalman tracking, gap filling, the decision which faces
// to recognise, the face crops and the identity write-back all run on the GPU. The CPU only learns how many crops a
// batch queued (4 bytes, host-mapped) to size the recogniser call.
#include "kernels.cuh"

constexpr int TRK = 128;           // track slots per camera
constexpr int OUT_RING = 64;       // output frames kept per camera (boxes + ids + names), ring

// Structure-of-arrays track table, index = cam * TRK + slot. Kalman arrays are [4 axes][ncam * TRK]: cx, cy, w, h.
struct TrackTable {
    float *x, *v, *P00, *P01, *P11;
    float *seen, *first, *rec_t, *rec_area, *id_score;
    float* pend_t;                 // capture time of the frame that first asked for a recognition still not done (-1: none)
    int *hits, *label;
    unsigned *gen, *tid;
    unsigned char *alive, *known;
    unsigned *next_tid;            // per camera
    int ncam;
};

struct TrackItem {
    int cam, fid, gap;
    float t, dt;                   // frame capture time (s since run start), frame period
    FrameRef f;
};
struct TrackBatch {
    TrackItem it[MAX_DET_B];
    int n;
};

struct TrackParams {
    float iou_thr, dist_k;
    float sig_meas, sig_acc, sig_acc_size, ttl;
    float recheck_s, retry_s, grow, min_rec_px;
    float unknown_thr, match_thr;  // score < unknown_thr: clearly unknown; >= match_thr: matched; between: retry
    float cross_iou;               // two tracks overlapping more than this: possible ID swap, re-check
    int confirm_hits;              // detections before a new track is recognised
};

// A face that asks for recognition in this tick (unified pipeline). The tracker kernel lists them; K5b hands out slots.
struct RecCand {
    int g;                         // cam * TRK + slot
    unsigned gen;
    int frame;                     // index of its frame in the tick
    int why;
    float4 box;
    float tfirst, treq, area, t;
};

struct PoolMeta {
    int cam, slot;
    unsigned gen;
    float t_first;                 // >= 0 for a track's first recognition (metrics)
};

// Crop pool: a ring of FP16 [3,112,112] crops in device memory. Producers (tracker kernels) append; the recogniser
// reads ranges of it in place.
struct CropPool {
    CropReq* req;                  // this batch's crop requests [MAX_DET_B * MAX_FACES]
    int* req_slot;                 // pool slot per request
    int* batch_count;              // requests this batch (device, reset per batch)
    int* denied;                   // unified pipeline: requests refused because the tick's cap was reached
    float* treq;                   // unified pipeline: per accepted request, when it was first asked (may be null)
    short* rcam;
    int* max_pend_ms;
    RecCand* cand;                 // unified pipeline: this tick's candidates and their count
    int* cand_count;
    unsigned long long* head;      // total crops ever requested (device)
    const volatile unsigned long long* released;   // crops the recogniser has finished with (host-mapped)
    PoolMeta* meta;                // [pool]
    float* tfirst_host;            // [pool] host-mapped, metrics only
    int pool;                      // 0 = unified pipeline: crops go straight into this tick's recogniser input
    int margin;                    // pool: free-slot margin; unified: this tick's recognition cap
};

struct TrackOut {
    float4* box;                   // [ncam][OUT_RING][TRK]
    int* id;                       // [ncam][OUT_RING][TRK]: track id << 12 | (label + 1), 0 = empty
    unsigned long long* frames;    // per camera: output frames written
    unsigned long long* faces;     // per camera: detected faces
    unsigned long long* requests;  // per camera: recognitions queued
    unsigned long long* reasons;   // [5]: new, grow, periodic, unsure, crossed
    unsigned long long* missed;    // faces that left the scene while still waiting for a recognition slot
    unsigned char* zone;           // [ncam][OUT_RING][TRK]: zone label under the centre of each delivered box (0 = none)
};

// One delivered box as the application receives it (live mode): K8 gathers a tick's rows into one contiguous block.
constexpr int LIVE_ROWS = 8;       // most frames per camera reported by one tick (the detected frame + filled ones)
struct LiveBox {
    float4 box;                    // x1, y1, x2, y2 in frame pixels
    int id;                        // track id << 12 | known << 11 | (label + 1); 0 = empty
    int zone;
};

// ---- unified pipeline ("one giant pipeline"): everything that changes from batch to batch is read from device
// memory, so decode-output -> names can be captured once as a single CUDA graph and replayed with one launch.
struct TickDesc {
    DetBatch det;                  // frames, letterbox, count
    ZoneMasks zones;
    TrackItem trk[MAX_DET_B];
    int n;
    int reserve;                   // recognition slots kept for faces that were refused a slot before
};
struct TickStats {
    int count;                     // recognitions accepted this tick (<= the tick's cap)
    int denied;                    // requests over the cap; those faces ask again on their next detected frame
    int boxes;                     // boxes delivered by this tick (detected + Kalman-filled)
    int named;                     // of those, boxes whose face has been through recognition
    int max_pend_ms;               // longest wait so far of a face still waiting for a recognition slot
    int cands;                     // faces that asked for recognition this tick
    float tfirst[128];             // first-seen time per accepted request (< 0 for re-checks), metrics only
    float treq[128];               // capture time of the frame that first asked for this recognition
    short rcam[128];               // camera of each accepted request
};

void launch_track_update(const FrameDets* dets, const TrackBatch& b, TrackTable tt, TrackParams p, CropPool cp,
                         TrackOut out, cudaStream_t s);
void launch_prep_det_u(const TickDesc* d, int B, __half* out, cudaStream_t s);
void launch_decode_nms_u(const __half* det_out, int B, int A, int layout_chw, float conf_thr, float iou_thr,
                         const TickDesc* d, FrameDets* out, cudaStream_t s);
// Last step of a tick: write the names known after this tick's recognitions into the delivered rows, count boxes
void launch_finalize_output(const TickDesc* d, int B, TrackTable tt, TrackOut out, TickStats* st, cudaStream_t s);
// K5b: give this tick's recogniser slots to the candidates, longest-waiting first (exact, no slot wasted)
void launch_assign_slots(const TickDesc* d, TrackTable tt, CropPool cp, TrackOut out, cudaStream_t s);
void launch_track_update_u(const FrameDets* dets, int B, const TickDesc* d, TrackTable tt, TrackParams p, CropPool cp,
                           TrackOut out, cudaStream_t s);
// K8 (live mode): copy this tick's delivered rows to dst[frame in tick][row, 0 = newest][track slot]
void launch_gather_rows(const TickDesc* d, int B, TrackOut out, LiveBox* dst, cudaStream_t s);
// Enrolment (live mode): if this tick recognised track `tid` of camera `cam`, store its embedding as gallery row
// `label` and name the track at once. *done is set to 1 when it happened.
void launch_enroll(const PoolMeta* meta, const int* count, int maxn, const __half* emb, TrackTable tt, int cam,
                   unsigned tid, __half* gallery, int label, int* done, cudaStream_t s);
void launch_crop_to_pool(CropPool cp, int max_req, __half* pool_tensor, cudaStream_t s);
void launch_publish_count(const int* dev_count, int* host_count, cudaStream_t s);
// valid: optional device-side count; entries >= *valid are ignored (unified pipeline runs a fixed-size batch)
void launch_scatter_identity(const MatchOut* m, int n, int pool_start, CropPool cp, TrackTable tt, float match_thr,
                             const int* valid, cudaStream_t s);
