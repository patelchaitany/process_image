# fastpipe: GPU-resident multi-camera face pipeline (C++ / CUDA / TensorRT)

Goal: as many cameras as possible on one GPU at a bounded delay. Every pixel and per-face operation stays on the
GPU. Only a few kilobytes per frame (boxes, IDs) cross PCIe, and the CPU only schedules and tracks.
Target GPU: RTX 3060 Laptop (GA106, sm_86):
- 30 SMs, 48 warps per SM
- 100 KB shared memory per SM
- 3 MB L2
- about 336 GB/s DRAM
- one NVDEC

## 1. Where the time went before (measured)

| Version | What dominated | Evidence |
|---|---|---|
| Repo (`process_image`) | Per-camera, batch-1, CPU round trips | NV12 → full BGR frame (kernel 1) → FP32 640×640 (kernel 2, nearest-neighbour, 3 strided byte loads per pixel); the whole YOLO output copied to the CPU for decode and NMS; one crop kernel launch **per face**; Triton gRPC hop per inference; per-session `cudaStreamSynchronize` |
| Python v1 | Python global lock, multi-process time-slicing | One CPU core busy, GPU 40–50% idle at 16–24 cameras; 4–8 CUDA contexts at 64+ cameras cut throughput from 130 to 71 frames/s |
| Python v2 | GPU compute plus host copies | 452 frames/s at 84% GPU. Each frame is uploaded as RGB (691 KB H2D), crops are made on the CPU (37 KB H2D per face), and ORT adds cast and transpose kernels |

The models set the floor. Each analysed frame costs:
- **Detector:** YOLOv8n-face at 384×640, about 4.9 GFLOP.
- **Recogniser:** ArcFace R50 at 112×112, about 6.3 GFLOP per face, run on about 2.5 faces per frame under the
  tracking policy (about 16 GFLOP).
  (The default recogniser is now EdgeFace-S, about a fifth of that GPU time per face: section 13.)

That is about 21 GFLOP per frame. At a realistic 15–20 effective FP16 TFLOPS on this GPU, the ceiling is about
**700–950 frames/s**, roughly 1.5–2× the Python v2 result. INT8 for R50 and a lighter recogniser move the ceiling;
everything else in this design removes overhead, so the pipeline gets close to that ceiling.

## 2. Dataflow

```
 per camera (×N)               shared, batched                                         CPU
┌──────────────┐  NV12 (dev)  ┌──────────────────────────────┐   ≤1.3 KB/frame   ┌───────────────────┐
│ demux (CPU)  │─────────────▶│ K1 prep_det  (fused)         │   host-mapped     │ tracker + policy  │
│ NVDEC decode │  frame ring  │ TRT detector (graph)         │──────────────────▶│ (per camera)      │
│ → ring slot  │  (4 slots)   │ K2 decode_nms (fused)        │                   │ new? re-check?    │
└──────────────┘              └──────────────────────────────┘                   │ budget            │
        ▲                          det stream                                    └─────────┬─────────┘
        │ slot released after crops                                              crop table│ (host-mapped)
        │                     ┌──────────────────────────────┐   8 B/face        ┌─────────▼─────────┐
        └─────────────────────│ K3 crop_prep_rec (fused)     │──────────────────▶│ identities, events │
                              │ TRT recogniser (graph)       │   host-mapped     │ metrics            │
                              │ K4 l2norm_match (fused)      │                   └───────────────────┘
                              └──────────────────────────────┘
                                   rec stream
```

- **Stage D (decode):** a pool of decoder threads. Each owns a set of cameras, paces them in real time, and
  decodes with NVDEC (libavcodec `cuda` hwaccel on the primary CUDA context). The decoded NV12 surface is copied
  device-to-device into the camera's frame ring on the thread's own stream, and the slot is published once that copy
  completes.
- **Stage A (detect):** one scheduler thread takes the newest due frame of up to 32 cameras, then launches
  `K1 → TRT det → K2` on the det stream. K2 writes compact detections straight into host-mapped pinned memory (zero
  copy, no `cudaMemcpy` call), and an event is recorded.
- **Stage T (track):** a CPU thread waits on the event and updates each camera's tracker. It decides which faces to
  recognise, then writes a crop table into host-mapped memory and appends it to the pending recognition batch.
- **Stage B (recognise):** `K3 → TRT rec → K4` on the rec stream, batching crops from several detection batches (up
  to 128). K4 writes (gallery index, score) per face into host-mapped memory.
- **Stage F (finish):** a completion thread waits on rec events, attaches identities to tracks, releases frame-ring
  slots and records metrics.

Stages A and B are on separate streams with separate buffers, so the detector of batch k+1 runs while the
recogniser of batch k does. With up to 3 detection batch contexts and 2 recognition contexts in flight, CPU work
(tracking) is hidden behind GPU work.

## 3. Memory layout

| Buffer | Location | Layout | Size | Lifetime |
|---|---|---|---|---|
| Frame ring | device, `cudaMallocPitch` | NV12: Y `[H][pitch]`, then UV `[H/2][pitch]` interleaved. Pitch is 256 B aligned, so every row starts on a 128 B line and vector loads stay aligned | 4 × 360 KB per camera (128 cameras: 180 MB) | Slot pinned by refcount from schedule until its crops are done |
| Det input | device | FP16 `[B,3,384,640]` planar (NCHW). The layout TensorRT asks for is checked with the engine inspector; if a reformat layer appears, switch to `kHWC8` and write that directly | 1.47 MB per frame, 47 MB at B=32 | Per det context (×3) |
| Det output | device | FP16 `[B,5,A]` channel-first (A=5040). Thread *t* reads anchor *t* of each channel, so loads are coalesced; the repo's `[B,A,5]` transpose is removed | 50 KB per frame | Per det context |
| Detections | host-mapped pinned | per frame: `int count; float4 box[64]; float score[64]` (SoA) | 1.3 KB per frame | Per det context |
| Crop table | host-mapped pinned | per face: `{slot ptrs, pitch, box, out index}` (32 B) | 4 KB per 128 faces | Per rec context |
| Rec input | device | FP16 `[R,3,112,112]` planar | 75 KB per face, 9.6 MB at R=128 | Per rec context (×2) |
| Embeddings | device | FP16 `[R,512]` | 1 KB per face | Per rec context |
| Gallery | device | FP16 `[G,512]`, L2-normalised at enrolment | 300 KB for G=300; fits in L2 | Resident |
| Matches | host-mapped pinned | `int idx; float score` per face | 8 B per face | Per rec context |

Per analysed frame, PCIe traffic falls from about 790 KB (Python v2) to under 2 KB. Device traffic in the custom
kernels is about 1.9 MB per frame, which is about 6 µs at the DRAM peak.

## 4. Fused kernels

All kernels use `__restrict__`, read-only loads (`__ldg`), vector types (`uchar4`, `half2`, `float4`) and no
dynamic allocation. Kernel parameters carry small per-batch tables by value (CUDA 12.1+ allows 32 KB of
parameters), so there is no H2D copy for them.

### K1 `prep_det`: NV12 → RGB → letterbox → normalise → FP16 planar, one launch per batch

- **Grid** `(ceil(W_out/8/32), H_out, B)`; block of 32×4. Each thread produces **8 horizontally adjacent output
  pixels**.
  - Loads: one 8-byte Y row segment (`uint2`) and four UV pairs (one `uint2`).
  - Stores: one 16-byte `int4` (8 × FP16) into each of the R, G and B planes.
- **Coalescing:** a warp covers 256 contiguous pixels of one row, so each plane store is 512 B, i.e. 4 full 128 B
  lines with no partial sectors.
- **Letterbox rows** are written as constants without loading anything. A general bilinear-resize path (for
  1080p → 384×640) uses the same thread mapping. Neighbouring warps reuse source rows through L1.
- **Math:** BT.601 limited-range YUV→RGB in FP32 registers, then `×1/255` folded into the colour matrix
  coefficients, so there is one FMA chain per channel.
- **Expected:** memory-bound at about 1.8 MB per frame, about 6 µs per frame. Check in Nsight Compute: DRAM
  throughput above 70% of peak and about 4 sectors per request.

### K2 `decode_nms`: score filter → box decode → sort → NMS → compact output, one block per image

1. **Compaction without per-thread atomics:** each thread tests the confidences of anchors *t*, *t*+256, … Each warp
   then uses `__ballot_sync` and `__popc` with a single `atomicAdd` to a shared counter, so a warp reserves its
   output range in one operation.
2. Candidates go to shared memory as **SoA arrays** (`x1[], y1[], x2[], y2[], s[]`, 4 B each), up to 512 (observed
   maximum 122). Consecutive threads hit consecutive banks, so there are no bank conflicts.
3. **Bitonic sort** of the scores, descending, in shared memory. Index payloads move with the keys. Strides ≤ 16
   use `__shfl_xor_sync` within a warp, with no shared-memory round trip.
4. **Bitmask NMS:**
   - The IoU of every candidate pair is computed in parallel; row *i* gets a bit for each lower-scored box *j* that
     *i* suppresses. With n ≤ 512 that is 16 words per row, at most 32 KB of shared memory.
   - One warp then scans in score order. Lanes 0–15 each hold one 32-bit word of the "removed" mask. Box *i* is
     kept if its bit is clear, in which case every lane ORs in its word of row *i*. Each step is one instruction per
     lane, with no serial O(n²) loop.
5. Box coordinates are mapped to frame pixels by subtracting the letterbox pad and dividing by the scale. Up to 64
   faces, largest kept first if over the limit, are written to host-mapped memory with coalesced `float4` stores.

The 50 KB per frame read is coalesced because of the channel-first output. Shared memory is about 44 KB per block,
so 2 blocks fit per SM, enough to cover 32 images.

### K3 `crop_prep_rec`: face crop → bilinear resize → NV12→RGB → normalise → FP16 planar, one block per face

- **Mapping:** block of 256 threads; output pixel `p = tid + k·256` row-major over 112×112. Stores to each plane are
  therefore contiguous per warp (64 B of FP16 per warp per plane; `half2` pairs would double that).
- **Source reads** are gathers, but from a small region (one face is typically under 64×64 source pixels), so they
  hit L1 after the first touch. The 4 Y taps and 4 UV taps are fetched with `__ldg`.
- **Normalisation** `(x − 127.5)/127.5` is folded into the colour matrix.
- **One launch for every face in the recognition batch**, instead of one per face as in the repo.
- **Reading the crop table:** each block reads its 32 B entry from host-mapped memory once, broadcast through
  shared memory.
- **Later:** with a landmark detector (SCRFD-KPS), the same kernel takes a 2×3 similarity transform instead of a box
  (five-point alignment), at the same cost.

### K4 `l2norm_match`: normalise → cosine against the gallery → arg-max, no intermediate matrix

- **One warp per embedding.** The lane holds 16 dims (512/32) as 8 `half2` in registers, and the squared norm is
  reduced with `__shfl_xor_sync`.
- **Gallery loop:** each warp walks the gallery rows. Each lane reads 32 B of a row with coalesced `uint4` loads; the
  whole gallery (300 KB) stays in L2. It takes a `half2` dot product with FP32 accumulation, reduces across the
  warp, and keeps a running maximum (score and index) in registers.
- **Output** is written once per embedding. There is no R×G similarity matrix in memory.
- **Large galleries:** for more than about 10k people, switch to a tensor-core (WMMA 16×16×16) tile variant that
  stages gallery tiles in shared memory and reduces the max per tile.

## 5. Inference engines (TensorRT C++ API, no ORT, no Triton hop)

- Engines are built with `trtexec`:
  - FP16, plus INT8 for the explicit-QDQ models.
  - Explicit I/O formats (`fp16:chw`), so our kernels write exactly what the first layer reads.
  - One optimisation profile: detector B = 1..32, recogniser R = 1..128.
- Per engine, one `IExecutionContext` per **batch bucket**:
  - Buckets: detector 1, 2, 4, 8, 12, 16, 24, 32; recogniser 8, 16, 32, 48, 64, 96, 128. Input is padded up to the
    bucket size.
  - Every context shares one user-managed activation buffer; this is safe because they serialise on one stream.
  - Shapes and tensor addresses are fixed per bucket, so each bucket's `enqueueV3` is **captured once into a CUDA
    graph** and replayed with a single `cudaGraphLaunch`. This removes per-layer launch overhead, which is large
    under WDDM/WSL.
- Per-layer timing (`trtexec --dumpProfile --dumpLayerInfo --profilingVerbosity=detailed`) is checked for reformat or
  copy layers at the I/O boundaries. Each one found is removed by matching our kernel's output format to it.

## 6. Threads, streams and synchronisation

| Thread | Waits on | Issues |
|---|---|---|
| Decoders (pool of 8, real-time paced) | Packet due time | NVDEC decode, D2D copy into the ring, slot publish (atomic) |
| Scheduler | New frames, free det context, per-camera in-flight < 2 | K1, TRT det graph, K2, event (det stream) |
| Tracker | Det event | CPU tracking; crop table; triggers a rec batch when it has ≥ R_min crops or the det queue is empty |
| Recogniser submit | Pending crops, free rec context | K3, TRT rec graph, K4, event (rec stream) |
| Finisher | Rec event | Identities, slot release, metrics |

- **No device-wide syncs.** Only events, and `cudaEventSynchronize` on worker threads.
- **The CPU never waits for the GPU on the submit path.** Back-pressure works by counting contexts and in-flight
  frames, not by syncing.
- **Overload policy, carried over from v2:**
  - Each camera always works on its newest frame.
  - New faces are recognised before re-checks, and re-checks are deferred while the recogniser has a backlog of
    more than one batch.
  - A camera never has more than 2 frames in flight.

The tracker stays on the CPU: about 20 boxes per camera per frame, branchy greedy matching, about 10 µs per frame.
Moving it to the GPU would add a kernel launch per batch and gain nothing, since its latency is hidden by the
pipelining.

## 7. Profiling method and acceptance checks

1. **Nsight Systems** (`nsys profile -t cuda,nvtx,osrt --cuda-graph-trace=node`) at 16 and 64 cameras, with NVTX
   ranges per stage and per batch. It answers:
   - Are the det and rec streams overlapping?
   - Where are the GPU idle gaps, and what is the CPU doing during them?
   - What do launch and API overhead cost?
   - How much time goes to each kernel and each TRT layer group?
2. **Nsight Compute** on K1–K4 (`--set full`):
   - achieved DRAM throughput as a % of peak
   - sectors per request, for coalescing
   - shared-memory bank conflicts
   - achieved occupancy
   - top warp-stall reasons
   - The kernels together should take under 5% of GPU time, and K1/K3 should run at 60% or more of DRAM peak.
3. **TensorRT layer profiles** for both engines: top layers, reformat layers, INT8 vs FP16 per layer.
4. **End to end**, with the same metrics and the same crowd clip as v1/v2, at 1–128 cameras, but each camera now
   decodes its own stream (no fan-out):
   - fps per camera
   - delay at the 50th and 95th percentile
   - time until a new face's identity is known
   - re-check interval
   - total frames per second
   - GPU, NVDEC and CPU use
5. **Correctness:**
   - detections vs the ONNX Runtime reference on the same frames: ≥ 99% matched at IoU > 0.5
   - embeddings vs the reference: cosine ≥ 0.98, with the differences due to bilinear vs OpenCV resize
   - INT8 vs FP32 embedding cosine, from `quantize.py`

## 8. Platform choices

- **C++ rather than Rust.** TensorRT, NVDEC/libavcodec, CUDA graphs and Nsight NVTX are all first-class C/C++
  APIs. Rust would need FFI bindings around each of them and gain nothing on the GPU side.
- **WSL2 (Ubuntu 24.04) on the laptop**, not native Windows:
  - The Linux CUDA 12.6 toolchain, NVDEC, Nsight Systems 2024.5 and Nsight Compute 2024.3 are already there, on the
    E: drive.
  - Windows' C: drive has no room for Visual Studio and the CUDA toolkit.
  - The single-process design does not need CUDA IPC, which is what failed under WSL2.
  - Production would run the same code on native Linux.

## 10. GPU-resident tracking and smart recognition (`--gpu-track 1`)

- **Track table:** 128 slots per camera in device memory, stored field by field (SoA). It holds the Kalman state
  (cx, cy, w, h, each with velocity and a 2×2 covariance), track ID, generation, name, match score and the time and
  size of the last recognition.
- **K5 `track_update`**, one 256-thread block per camera-frame (a camera appears at most once per detector batch, so
  no two blocks touch the same camera):
  1. Load detections into shared memory and compact the live tracks.
  2. Thread *j* owns track *j*: it predicts the track over the keyframe gap and keeps each step for the smoother.
  3. Build the detection × track IoU matrix in shared memory (64 × 128). Assign greedily with repeated block-wide
     argmax (warp-shuffle reduction), then a centre-distance pass for what IoU missed.
  4. Kalman-update matched tracks, RTS-smooth the skipped frames, write boxes, IDs and names to the per-camera output
     ring.
  5. Create tracks for unmatched detections (free slots found by ballot compaction), expire lost ones.
  6. Recognition rules: new after 3 hits; smoothed area +50%; every 3 s; borderline score; crossing a face with a
     different name. Selected faces get a crop-pool slot (`atomicAdd` on a ring head, bounded by the released tail the
     host publishes).
- **Crop-to-pool:** the grid is sized for the worst case and blocks beyond this batch's device-side count exit at once.
  Each face is written straight into its pool slot as FP16 [3,112,112].
- **Recogniser:** reads `pool[tail : tail+R]` in place (`run_direct`; no CUDA graph, because the address changes).
- **K4:** matches embeddings against the gallery.
- **K6 `scatter_identity`:** writes the name and score into the track slot if its generation still matches.
- **CPU:** reads a 4-byte count per batch (host-mapped) and publishes the released pool tail. Frames, boxes, crops and
  identities never leave the GPU; only the final per-frame output does.

## 11. Unified pipeline: one CUDA graph per batch (`--unified 1`)

- **Tick:** K1 → detector → K2 → K5 → crop → recogniser → K4 → K6 on one stream, captured once per
  (frames bucket B, recognitions bucket R) and replayed with a single `cudaGraphLaunch`.
- **Per-tick parameters in device memory.** K1, K2 and K5 take an optional `const TickDesc*` (frame references,
  letterbox, zone masks, camera id, frame id, gap, timestamps). The scheduler copies the 3 KB descriptor from pinned
  host memory before the launch. Grids are sized for the bucket; blocks beyond the tick's frame count exit at once.
- **Recognition slots:** K5 claims slots in the tick's recogniser input with a compare-and-swap loop on a device
  counter, bounded by the tick's cap R (baked into that graph). Requests over the cap are counted as denied and are
  not marked on the track, so the face asks again on its next detected frame.
- **Fixed-size recogniser call:** the recogniser, K4 and K6 run at batch R; K6 ignores entries beyond the device-side
  count. The scheduler picks R from an exponential average of requests per frame, rounded down to a captured size.
- **Buffers:** one set of device buffers serves every tick, because ticks execute in order on one stream. Only the
  descriptor and the 520-byte statistics block are per-tick host memory.
- **K5b `assign_slots`:** K5 only appends the faces that ask for recognition to a candidate list. One 256-thread
  block then hands out the tick's recogniser slots in three rounds (waited more than 0.3 s, waited at all, new), using a
  block-local counter, so every slot up to the cap is used and nobody is served out of turn. A refused face keeps the
  capture time of the frame that first asked and returns with priority on its next detected frame.
- **K7 `finalize_output`:** after K6, rewrites the rows this tick delivers as
  `track id << 12 | recognised << 11 | (gallery index + 1)` and counts delivered and named boxes.
- **Per-second accounting:** each accepted request carries its camera and first-ask time back in the 1.3 KB statistics
  block; the host records, per camera and per second of video, when the last frame or name of that second was ready.
- **Exact-size recognition (`--uni-split 1`, default):** the tick is two graph launches on the same stream. Graph A(B)
  runs K1 → detector → K2 → K5 → K5b with the largest cap; the accepted count (4 bytes) is copied to pinned host
  memory and the scheduler thread spins on an event (the GPU idles about 0.09 ms). Graph B(R) then runs crop →
  recogniser → K4 → K6 → K7 at the smallest captured size ≥ count (sizes in steps of 4). Recogniser slots used rise
  from about 86% to 97%. `--uni-split 0` keeps the single graph per tick.
- **Default mode:** `--unified 1`, detection on every 2nd frame (`--analyze-fps 15`), FIFO with a 3 s buffer, slots
  rounded up (`--uni-headroom 1.0`). `--boxeval N` measures skipped boxes against detecting every frame.
- **Trade-off, measured:** about 16% more GPU time than the two-stream scheduler below saturation (no overlap between
  the two networks' kernels) and higher frame delay, but even degradation under overload. See REPORT.md.

## 12. Zones and live cameras (`--live`)

**Zones.** Each camera has a label image in device memory, one byte per frame pixel (0 = no zone, 1..255 = zone).
Zones arrive as polygons in 0..1 frame coordinates (`Pipeline::set_zones`); the CPU fills them into the label image
(even-odd scanline, later zones on top) and uploads it, only when the zones change. K7 (`finalize_output_kernel`)
then looks up the label under the centre of every delivered box, detected or Kalman-filled, and writes it to
`TrackOut.zone`. The per-box cost is one byte read on the GPU. `TickDesc.zones` carries the pointers, so zones can
change between ticks without re-capturing a graph.

**Live input.** `--live <pipe pattern>` replaces the test clips by one named pipe per camera. Each H.264 access unit
arrives behind a 16-byte header `{'FPAU', bytes, frame number}` and goes straight to the NVDEC decoder
(`VideoDecoder::next_live`, low-delay, no demuxer). The frame number rides through the decoder as the packet
timestamp and becomes the frame id, so results can be matched to the sender's frames. One decoder thread per camera;
capture time is arrival time.

**Live output.** After a tick's graphs, K8 (`gather_rows_kernel`) packs the rows the tick delivered (up to 8 frames
per camera: box, track id, name, zone) into one block and a single copy brings it to pinned host memory. The track
thread prints one JSON line per frame on stdout. This is the only per-box data that leaves the GPU.

**Control** (stdin lines): `Z <cam> <zones> {<label> <points> x y ...}` replaces a camera's zones, `E <cam> <track>
<label>` stores that track's face as gallery row `<label>` (`enroll_kernel`: the embedding of the track's next
recognition is L2-normalised into the gallery on the GPU and the track is named at once), `Q` stops.

**Test page.** `live/` holds a browser page, a bridge (WebSocket to pipes) and a launcher; see `live/README.md`.

## 13. Recogniser and match thresholds

The recogniser is an engine file, nothing else: every supported model takes a 112×112 RGB face scaled to −1..1 (K3)
and returns 512 numbers (K4 normalises them). `setup/build_engines.sh` builds four:

| Engine | Model | Size | GPU time per face | 40 dense cameras |
|---|---|---|---|---|
| `rec_edgeface_s_fp16` (default) | EdgeFace-S | 14.8 MB | 0.09 ms | 100% of seconds within 1 s, GPU 35% |
| `rec_edgeface_base_fp16` | EdgeFace-Base | 73 MB | 0.19 ms | 100%, GPU 47% |
| `rec_r50_fp16` | ArcFace R50 (w600k) | 174 MB | 0.51 ms | 95.2%, GPU 94% |
| `rec_mbf_fp16` | MobileFaceNet (w600k) | 13.6 MB | 0.26 ms | not run |

**Thresholds travel with the engine.** A cosine score does not mean the same thing for two models, so each engine has
a `<engine>.thr` file (`match_thr`, `unknown_thr`), written by `build_engines.sh` and read in `main.cpp`
(`load_thresholds`). A score at or above `match_thr` names the face, one below `unknown_thr` is a stranger, and one
between is retried after `--retry-s`. `--match-thr` and `--unknown-thr` override the file. The values give every model
the false-match rate ArcFace R50 has at 0.45 / 0.25.

**The trade.** EdgeFace-S is the default because it frees about 60% of the GPU at 40 dense cameras and still passes
at 64. At the same false-match rate it re-matches the same face less often than ArcFace R50 (28% against 42% one
second apart on the test clips' small faces). Use `--rec engines/rec_r50_fp16.plan` when matching
quality matters more than camera count. Measurements, the model comparison and licences:
[../fastpipe-dev/benchmarks/models/RESULTS.md](../fastpipe-dev/benchmarks/models/RESULTS.md).

## 9. Build, run and results

```bash
# WSL2 / Linux with CUDA 12.x, TensorRT 10.x, FFmpeg 6 dev packages (setup/wsl_trt_install.sh)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
setup/build_engines.sh                      # FP16 engines + per-layer profiles (ONNX from setup/prep_onnx.py)
build/fastpipe --det engines/det_r384_fp16.plan --rec engines/rec_edgeface_s_fp16.plan --counts 16,32,64,128
build/fastpipe ... --policy fifo --buffer-s 3 --budget 0 --counts 40,48,56   # real-time rule: every frame, 3 s buffer
build/fastpipe ... --dump 60      # validation dump (../fastpipe-dev/tools/check/compare_dump.py)
build/fastpipe ... --kbench 1     # per-kernel GPU time and achieved bandwidth
../fastpipe-dev/tools/bench/prof.sh 64 n64   # Nsight Systems timeline with NVTX stage ranges
```

Model comparison, the recogniser choice and its measurements: [../fastpipe-dev/benchmarks/models/RESULTS.md](../fastpipe-dev/benchmarks/models/RESULTS.md).
