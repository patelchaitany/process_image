# Real-Time Face Detection & Recognition Pipeline

## Overview

A C++ pipeline that detects and recognizes faces from live RTSP streams and MP4 files within a 20ms per-frame budget. Uses NVIDIA Triton Inference Server with CUDA shared memory to serve YOLO (face detection) and ArcFace (face embedding) models via TensorRT FP16. Face matching runs on FAISS-GPU with a SQLite-backed face database. Designed for one camera today, architected for 16+ cameras later.

**Target hardware**: NVIDIA T4 (Turing datacenter, 16GB GDDR6, 65 TOPS INT8 / 130 TFLOPS FP16 Tensor Cores).

## Constraints

- Total pipeline latency: < 20ms per frame (target: ~13ms)
- Language: C++17
- GPU inference: TensorRT FP16 via Triton Inference Server
- CPU fallback: ONNX Runtime via Triton (no GPU required)
- Input sources: MP4 files, RTSP streams
- Face database: SQLite on disk, FAISS-GPU index in memory
- Monitoring: Per-frame CSV metrics with < 0.04ms overhead

## Architecture Decision: Triton + CUDA Shared Memory

Three approaches were evaluated:

| Approach | Pros | Cons |
|---|---|---|
| **Triton + CUDA Shared Memory** | Dynamic batching for multi-camera; zero-copy inference; model management | Extra deployment dependency |
| Embedded TensorRT (no server) | Zero network overhead; simpler | No batching across cameras; manual model management |
| NVIDIA DeepStream | Zero PCIe transfers (GPU decode) | Heavy SDK; no CPU fallback; steep learning curve |

**Selected: Triton + CUDA Shared Memory.** The 0.3ms PCIe transfer for the frame is not the bottleneck (inference is), and Triton's dynamic batching is critical for multi-camera scaling. CUDA shared memory eliminates tensor data from gRPC calls -- only control metadata travels over the wire.

## System Components

Nine components, each with one responsibility:

| Component | Language | Responsibility | Runs On |
|---|---|---|---|
| FrameSource | C++ (FFmpeg) | Decode MP4/RTSP streams into raw BGR frames | CPU |
| GPUMemoryPool | C++/CUDA | Pre-allocate CUDA buffers, register as Triton CUDA shared memory | GPU |
| Preprocessor | CUDA kernels | Resize, normalize, color convert, layout transform | GPU |
| TritonClient | C++ (gRPC) | Send inference requests referencing CUDA shared memory handles | CPU (control) |
| PostProcessor | C++/CUDA | NMS on YOLO output, face cropping + ArcFace preprocessing | CPU + GPU |
| FaceMatcher | C++ (FAISS-GPU) | Cosine similarity search against known face database | GPU |
| FaceDatabase | C++ (SQLite) | Persist face embeddings + metadata; load into FAISS at startup | CPU (disk) |
| ResultHandler | C++ | Package per-frame results (face IDs, bboxes, scores) for the caller | CPU |
| MetricsLogger | C++ | CUDA event timing + lock-free async CSV writer | CPU (async thread) |

**Triton-side models** (served, not part of the C++ client):

| Model | Input | Output | Backend |
|---|---|---|---|
| yolo26_face | [1,3,640,640] FP32 | [N,5] detections (cxcywh + conf) | ONNX Runtime + TensorRT FP16 |
| arcface | [M,3,112,112] FP32 | [M,512] embeddings | ONNX Runtime + TensorRT FP16 |

Both models use the existing Triton configs at `/Users/chpatel/projects/motivation/models/model_repository/` with dynamic batching enabled (preferred batch sizes: 1, 4, 8, max queue delay: 100us).

## Data Flow: The Frame Journey

Every byte is tracked through the pipeline. The frame enters GPU once, all processing stays on GPU, and only tiny result metadata returns to CPU.

### PCIe Transfers Per Frame

| Transfer | Direction | Size | Time | Why |
|---|---|---|---|---|
| #1 Raw frame | CPU → GPU | ~6 MB (1080p) | 0.3ms | FFmpeg decodes to CPU; unavoidable without NVDEC |
| #2 YOLO output | GPU → CPU | ~1 KB | 0.01ms | NMS runs on CPU; output tensor is small |
| #3 Final results | GPU → CPU | ~100 B | 0.01ms | Application needs match results on CPU |

Total PCIe transfer time: ~0.32ms (2.5% of pipeline).

### Atomic Operation Timing (T4, FP16)

#### Stage 1: Frame Decode (CPU)

| Operation | What | Time |
|---|---|---|
| av_read_frame | Read compressed packet from MP4/RTSP | ~0.1ms |
| avcodec_send_packet | Send packet to H.264 decoder | ~0.01ms |
| avcodec_receive_frame | Decode one H.264 frame (1080p) to YUV420 | ~0.8-1.2ms |
| sws_scale | Convert YUV420 to BGR24 | ~0.3-0.5ms |
| **Subtotal** | | **~1.2-1.8ms** |

#### Stage 2: CPU to GPU Transfer

| Operation | What | Time |
|---|---|---|
| cudaMemcpyAsync (H2D) | Copy 1920x1080x3 BGR to CUDA shared memory buffer | ~0.3ms |
| **Subtotal** | | **~0.3ms** |

#### Stage 3: YOLO Preprocessing (GPU CUDA Kernels)

| Operation | What | Time |
|---|---|---|
| Kernel launch | CUDA driver dispatch | ~0.005ms |
| Bilinear resize | 1920x1080 to 640x640 | ~0.05-0.08ms |
| Letterbox padding | Pad to maintain aspect ratio | ~0.02ms |
| BGR to RGB | Channel swap | ~0.02ms |
| HWC to CHW | Layout transpose for model input | ~0.03ms |
| Normalize /255.0 | uint8 to float32, scale to [0,1] | ~0.01ms |
| **Subtotal** | | **~0.14-0.17ms** |

These operations can be fused into a single CUDA kernel: ~0.05-0.08ms total (one memory read, one write).

#### Stage 4: gRPC Call #1 to Triton

| Operation | What | Time |
|---|---|---|
| Protobuf serialize | Request metadata (model name, shm handle) | ~0.01ms |
| TCP loopback send | localhost:8001 | ~0.02ms |
| Triton request parse | Deserialize gRPC request | ~0.01ms |
| Triton queue scheduling | Dynamic batching decision | ~0.02ms |
| Triton shm resolve | Look up registered CUDA shared memory pointer | ~0.005ms |
| **Subtotal** | | **~0.065ms** |

#### Stage 5: YOLO Inference (TensorRT FP16, GPU)

| Layer Group | Time | % |
|---|---|---|
| TRT engine enqueue | ~0.01ms | <1% |
| Stem convolution | ~0.3ms | 5% |
| Backbone (CSPDarknet, 4 stages) | ~2.5-3.5ms | 60% |
| Neck (PAN-FPN) | ~1.0-1.5ms | 25% |
| Detection head | ~0.5-0.8ms | 10% |
| **Subtotal** | **~4.3-6.1ms** | |

#### Stage 6: gRPC Response #1

| Operation | What | Time |
|---|---|---|
| Triton output to shm | Write output tensor to CUDA shared memory | ~0.005ms |
| Protobuf serialize response | Timing stats + output shm handle | ~0.01ms |
| TCP loopback return | localhost | ~0.02ms |
| Client deserialize | Parse response metadata | ~0.01ms |
| **Subtotal** | | **~0.045ms** |

#### Stage 7: YOLO Output Copy + Parsing

| Operation | What | Time |
|---|---|---|
| cudaMemcpyAsync (D2H) | Copy YOLO output tensor (~168KB) to CPU | ~0.01ms |
| Confidence threshold | Filter 8400 detections, keep conf > 0.5 | ~0.02ms |
| Box coord decode | Convert cxcywh to xyxy on ~50 candidates | ~0.005ms |
| **Subtotal** | | **~0.035ms** |

#### Stage 8: NMS (CPU)

| Operation | What | Time |
|---|---|---|
| Sort by confidence | std::sort on ~20-50 candidates | ~0.002ms |
| IoU computation | Pairwise intersection-over-union | ~0.01ms |
| Greedy suppression | Remove boxes with IoU > 0.45 | ~0.005ms |
| Scale to original | Map 640x640 coords back to 1920x1080 | ~0.001ms |
| **Subtotal** | | **~0.02ms** |

#### Stage 9: Face Crop + ArcFace Preprocessing (GPU)

For a batch of 5 detected faces:

| Operation | What | Time |
|---|---|---|
| Kernel launch | Single dispatch for all faces | ~0.005ms |
| Crop from original | Read face regions from GPU frame buffer | ~0.02ms |
| Bilinear resize to 112x112 | Per-face resize | ~0.03ms |
| Normalize (x-127.5)/127.5 | ArcFace normalization | ~0.01ms |
| HWC to CHW | Layout transpose | ~0.01ms |
| Batch packing | Pack into [5,3,112,112] contiguous tensor | ~0.01ms |
| **Subtotal** | | **~0.085ms** |

#### Stage 10: gRPC Call #2 + ArcFace Inference

| Operation | What | Time |
|---|---|---|
| gRPC request overhead | Same as Stage 4 | ~0.065ms |
| ArcFace TensorRT FP16 | ResNet-100 backbone (batch of 5 faces) | ~2.5-4.0ms |
| gRPC response overhead | Same as Stage 6 | ~0.045ms |
| **Subtotal** | | **~2.6-4.1ms** |

ArcFace layer breakdown:

| Layer Group | Time | % |
|---|---|---|
| Initial conv (3 to 64, 112x112) | ~0.2ms | 7% |
| ResNet Stage 1 (64ch, 56x56) | ~0.3ms | 10% |
| ResNet Stage 2 (128ch, 28x28) | ~0.5ms | 16% |
| ResNet Stage 3 (256ch, 14x14) | ~0.8ms | 26% |
| ResNet Stage 4 (512ch, 7x7) | ~0.6ms | 19% |
| Global avg pool + FC to 512 | ~0.1ms | 3% |
| BatchNorm + output | ~0.05ms | 2% |

#### Stage 11: FAISS-GPU Matching

| Operation | What | Time |
|---|---|---|
| L2 normalize | Normalize 512-dim embedding to unit length | ~0.005ms |
| FAISS search | Top-K nearest neighbors (cosine similarity) | ~0.05ms |
| Threshold check | Filter below confidence on CPU | ~0.001ms |
| **Subtotal** | | **~0.056ms** |

FAISS search scales with database size:

| DB Size | Time | Index Type |
|---|---|---|
| 100 faces | ~0.02ms | Flat |
| 1,000 faces | ~0.05ms | Flat |
| 10,000 faces | ~0.1ms | IVF |
| 100,000 faces | ~0.5ms | IVF (nprobe=16) |

#### Stage 12: Result Copy

| Operation | What | Time |
|---|---|---|
| cudaMemcpyAsync (D2H) | Copy match IDs + scores (~100 bytes) | ~0.001ms |
| SQLite lookup (optional) | Fetch face name by ID (cached after first read) | ~0.05ms |
| Result struct assembly | Pack into result object | ~0.001ms |
| **Subtotal** | | **~0.05ms** |

### Total Pipeline Budget

| Stage | Time | % |
|---|---|---|
| Frame decode (FFmpeg) | ~1.5ms | 12% |
| CPU to GPU transfer | ~0.3ms | 2% |
| YOLO preprocess | ~0.15ms | 1% |
| gRPC #1 overhead | ~0.11ms | 1% |
| YOLO inference | ~5.2ms | **41%** |
| YOLO parse + NMS | ~0.055ms | <1% |
| Face crop + preprocess | ~0.085ms | 1% |
| gRPC #2 overhead | ~0.11ms | 1% |
| ArcFace inference | ~3.3ms | **26%** |
| FAISS matching | ~0.056ms | <1% |
| Result handling | ~0.05ms | <1% |
| Monitoring overhead | ~0.035ms | <1% |
| **Total** | **~11-14ms** | |

Bottleneck: inference (YOLO 41% + ArcFace 26% = 67% of pipeline time). Memory transfers are 2.5%.

## Monitoring & Metrics

### Instrumentation Strategy

All GPU operations are timed using CUDA Events (recorded into the stream, no synchronization during the frame). CPU operations use `std::chrono::high_resolution_clock`. Triton gRPC responses include built-in timing statistics for free.

| Technique | Overhead per call | Count per frame | Safe for production |
|---|---|---|---|
| cudaEventRecord | ~0.002ms | 12 | Yes |
| chrono::now() | ~0.00002ms | 10 | Yes |
| Triton response stats | 0ms (already computed) | 2 | Yes |
| CSV row format + buffer push | ~0.011ms | 1 | Yes |
| **Total per frame** | | | **~0.035ms** |

CUDA event results are read asynchronously after the frame completes, on a separate monitoring thread. The pipeline thread never blocks on timing reads.

### CSV Metrics Output

One row per frame with every atomic timing:

```
frame_id,timestamp_utc,source_id,
decode_ms,yuv_to_bgr_ms,cpu_to_gpu_ms,
preprocess_resize_ms,preprocess_color_ms,preprocess_transpose_ms,preprocess_normalize_ms,preprocess_total_ms,
grpc1_overhead_ms,yolo_inference_ms,triton_yolo_queue_ms,triton_yolo_compute_ms,
yolo_output_copy_ms,confidence_filter_ms,nms_sort_ms,nms_iou_ms,nms_total_ms,faces_detected,
face_crop_ms,face_arcface_preprocess_ms,
grpc2_overhead_ms,arcface_inference_ms,triton_arcface_queue_ms,triton_arcface_compute_ms,
faiss_normalize_ms,faiss_search_ms,
result_copy_ms,total_pipeline_ms,gpu_memory_used_mb
```

### Async CSV Writer

The pipeline thread formats each row and pushes it into a lock-free ring buffer (~0.011ms). A separate writer thread drains the buffer and flushes to disk. The pipeline thread never blocks on file I/O.

Configuration (`pipeline.yaml`):

```yaml
metrics:
  enabled: true
  output_dir: "./metrics/"
  format: "csv"
  flush_interval_ms: 1000
  buffer_size: 100
  rotate_size_mb: 100
```

File rotation: new file every 100MB, named `metrics_YYYY-MM-DD_HHMMSS.csv`.

## Face Database

### SQLite Schema

```sql
CREATE TABLE faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    embedding BLOB NOT NULL,  -- 512 × float32 = 2048 bytes
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_faces_name ON faces(name);
```

### Lifecycle

- **Startup**: Load all rows from SQLite, upload embeddings to GPU, build FAISS-GPU index (Flat for < 1000 faces, IVF for larger).
- **Enrollment**: New face runs through the pipeline to extract an embedding, which is written to SQLite and added to the FAISS index in-place.
- **Lookup flow**: FAISS returns a vector index, which maps to the SQLite row ID.

## Error Handling

| Failure | Behavior |
|---|---|
| Triton not reachable at startup | Retry 3 times with 1s backoff, then exit with clear error message |
| Triton gRPC call fails mid-pipeline | Skip frame, log error, increment `frames_dropped` counter in metrics CSV |
| FFmpeg decode error (corrupt frame) | Skip frame, continue to next; log to metrics as `decode_ms = -1` |
| FAISS returns no match (unknown face) | Return result with `face_id = -1` and `confidence = 0.0`; not an error |
| CUDA OOM during buffer allocation | Reduce buffer pool size and retry; if still fails, fall back to CPU mode |
| SQLite write failure on enrollment | Return error to caller; FAISS index is not updated (remains consistent with DB) |

## CPU Fallback

When `cudaGetDeviceCount()` returns 0, the system auto-switches to CPU mode:

| Aspect | GPU Mode | CPU Mode |
|---|---|---|
| Triton model config | `KIND_GPU` + TensorRT accelerator | `KIND_CPU`, ONNX Runtime only |
| Shared memory | CUDA shared memory (GPU pointer) | System shared memory (CPU pointer) |
| Preprocessing | CUDA kernels | OpenCV (cv::resize, cv::dnn::blobFromImage) |
| FAISS | FAISS-GPU | FAISS-CPU |
| Expected latency | ~13ms | ~100-200ms (functional, not real-time) |

The C++ client uses runtime polymorphism (abstract interfaces) so the same pipeline orchestration code works for both modes. Backend selection happens once at startup.

## Multi-Camera Scaling

Each camera gets its own decode thread and pre-allocated CUDA shared memory region. Triton's dynamic batching (already configured with `preferred_batch_size: [1, 4, 8]` and `max_queue_delay_microseconds: 100`) automatically groups frames from different cameras into single inference calls.

For 16 cameras at 30fps (480 frames/sec), adjust Triton configs:
- `preferred_batch_size: [4, 8, 16]`
- `instance_group: count: 2` (two model instances on the same GPU, if memory allows)

## Project Structure

```
process_image/
├── CMakeLists.txt
├── src/
│   ├── main.cpp
│   ├── frame_source/
│   │   ├── frame_source.h
│   │   ├── ffmpeg_source.h
│   │   └── ffmpeg_source.cpp
│   ├── gpu/
│   │   ├── memory_pool.h
│   │   ├── memory_pool.cu
│   │   ├── preprocessor.h
│   │   └── preprocessor.cu
│   ├── inference/
│   │   ├── triton_client.h
│   │   ├── triton_client.cpp
│   │   ├── face_detector.h
│   │   ├── face_detector.cpp
│   │   ├── face_recognizer.h
│   │   └── face_recognizer.cpp
│   ├── matching/
│   │   ├── face_database.h
│   │   ├── face_database.cpp
│   │   ├── face_matcher.h
│   │   └── face_matcher.cpp
│   ├── postprocess/
│   │   ├── nms.h
│   │   ├── nms.cpp
│   │   ├── face_cropper.h
│   │   └── face_cropper.cu
│   ├── pipeline/
│   │   ├── pipeline.h
│   │   └── pipeline.cpp
│   ├── metrics/
│   │   ├── metrics_logger.h
│   │   ├── metrics_logger.cpp
│   │   ├── frame_metrics.h
│   │   └── ring_buffer.h
│   └── utils/
│       ├── config.h
│       └── timer.h
├── config/
│   └── pipeline.yaml
├── models/ -> /Users/chpatel/projects/motivation/models
├── metrics/
├── tests/
│   ├── test_pipeline.cpp
│   └── test_nms.cpp
└── docs/
```

## Dependencies

| Library | Purpose | Version Constraint |
|---|---|---|
| FFmpeg (libavcodec, libavformat, libswscale) | Video decode (MP4, RTSP) | >= 5.0 |
| Triton Client C++ (grpc_client) | Inference requests + CUDA shared memory | Matches Triton server version |
| CUDA Toolkit | GPU kernels, memory management, events | >= 11.8 |
| FAISS (faiss-gpu) | Face embedding similarity search | >= 1.7 |
| SQLite3 | Face database persistence | >= 3.36 |
| yaml-cpp | Configuration loading | >= 0.7 |
| OpenCV (optional) | CPU fallback preprocessing | >= 4.5 |
| spdlog (optional) | Structured logging | >= 1.10 |

## Success Criteria

1. Single-camera 1080p pipeline completes in < 20ms per frame (target: ~13ms)
2. CUDA shared memory verified: Triton `compute_input_time_ns` near zero
3. CPU fallback functional (not necessarily real-time)
4. Per-frame CSV metrics capture all atomic timings with < 0.04ms overhead
5. SQLite face database loads into FAISS-GPU index at startup
6. Clean build with CMake, no external build system dependencies beyond listed libraries
