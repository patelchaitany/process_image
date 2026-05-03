# Project Architecture: Real-Time Face Detection & Recognition Pipeline

## Overview

A high-performance C++17/CUDA pipeline that processes live video (RTSP streams or MP4 files), detects faces using YOLO, and identifies them using ArcFace embeddings with FAISS similarity search. Designed for NVIDIA T4 GPUs with a target latency of <20ms per frame (~13ms typical). Includes a CPU fallback mode.

---

## Directory Structure

```
process_image/
├── src/                          # C++ source code (the actual pipeline)
│   ├── main.cpp                  # ENTRY POINT - CLI parsing, signal handling, launches Pipeline
│   ├── pipeline/
│   │   ├── pipeline.h/cpp        # Main orchestrator - init, run loop, shutdown
│   │   └── result_handler.h/cpp  # Packages per-frame detection results
│   ├── frame_source/
│   │   ├── frame_source.h        # Abstract interface for video sources
│   │   ├── ffmpeg_source.h/cpp   # Decodes MP4/RTSP via FFmpeg to BGR24 frames
│   ├── inference/
│   │   ├── triton_client.h/cpp   # gRPC client to Triton (CUDA shared memory support)
│   │   ├── face_detector.h/cpp   # Wraps YOLO inference via TritonClient
│   │   └── face_recognizer.h/cpp # Wraps ArcFace inference via TritonClient
│   ├── gpu/
│   │   ├── memory_pool.h/cu      # CUDA buffer allocation + Triton SHM registration
│   │   ├── cpu_memory_pool.h/cpp # CPU RAM buffers (fallback mode)
│   │   ├── preprocessor.h/cu     # CUDA kernels: resize, color convert, normalize
│   │   ├── cpu_preprocessor.h/cpp# OpenCV-based preprocessing (CPU fallback)
│   ├── postprocess/
│   │   ├── nms.h/cpp             # Non-maximum suppression (confidence + IoU filtering)
│   │   ├── face_cropper.h/cu     # CUDA kernels: crop face regions + ArcFace prep
│   ├── matching/
│   │   ├── face_database.h/cpp   # SQLite wrapper for persistent face storage
│   │   ├── face_matcher.h/cpp    # FAISS-GPU cosine similarity search
│   │   └── cpu_face_matcher.h/cpp# Brute-force cosine search (CPU fallback)
│   ├── metrics/
│   │   ├── frame_metrics.h       # Per-frame timing struct (26 fields)
│   │   ├── metrics_logger.h/cpp  # Async CSV writer with background thread
│   │   └── ring_buffer.h         # Lock-free SPSC ring buffer for metrics
│   └── utils/
│       ├── config.h              # PipelineConfig struct, parse_args(), YAML loader
│       └── timer.h               # CpuTimer, CudaEventTimer, ScopedCpuTimer
├── tools/                        # Python utilities
│   ├── enroll_faces.py           # Populate face DB from photo folders
│   ├── download_models.py        # Create Triton model repo (dummy or real ONNX)
│   └── requirements.txt          # tritonclient[grpc], numpy, opencv-python
├── tests/                        # GoogleTest unit tests
│   ├── test_nms.cpp
│   ├── test_frame_source.cpp
│   ├── test_face_database.cpp
│   ├── test_face_matcher.cpp
│   ├── test_triton_client.cpp
│   └── test_cpu_preprocessor.cpp
├── config/
│   └── pipeline.yaml             # Runtime configuration (thresholds, sizes, metrics)
├── docs/                         # Design specs & implementation guides
│   ├── superpowers/specs/        # Architecture design document
│   ├── implementation-details/   # Phase-by-phase implementation notes (phases 0-11)
│   ├── setup.md                  # Build & setup instructions
│   └── examples.md               # Usage examples
└── CMakeLists.txt                # Build system (CMake)
```

---

## Entry Point

### `src/main.cpp` - THE STARTING POINT

This is where everything begins. When you run `./process_image`, execution starts here.

**What it does:**
1. Parses command-line arguments (`--input`, `--triton`, `--db`, `--config`, etc.)
2. Sets up a signal handler (SIGINT/SIGTERM) so Ctrl+C triggers graceful shutdown
3. Creates a `Pipeline` object and calls `pipeline.init(config)`
4. Calls `pipeline.run(running_flag)` which enters the main frame-processing loop
5. On shutdown signal, `running_flag` becomes false, loop exits, resources are cleaned up

```
main(argc, argv)
  │
  ├─> parse_args(argc, argv)          [config.h]    → fills PipelineConfig struct
  ├─> setup signal handler                           → sets `running = false` on Ctrl+C
  ├─> Pipeline pipeline;
  ├─> pipeline.init(config);                         → opens video, connects Triton, allocates GPU
  ├─> pipeline.run(running);                         → main loop: decode → detect → recognize → match
  └─> pipeline.shutdown();                           → frees all resources
```

---

## Complete Call Flow (Who Calls What)

### Initialization Phase (`Pipeline::init`)

```
Pipeline::init(config)
  │
  ├─> detect_device()                                → checks if CUDA GPU is available
  │
  ├─> FFmpegSource::open(input_path)                 → opens MP4 file or RTSP stream
  │
  ├─> TritonClient::connect(triton_url)              → gRPC connection (3 retries, 1s backoff)
  │
  ├─> [IF GPU MODE]
  │   ├─> GPUMemoryPool::init(frame_size, batch)     → pre-allocates CUDA device buffers
  │   ├─> GPUMemoryPool::register_triton_shm()       → registers buffers with Triton for zero-copy
  │   ├─> Preprocessor::init()                       → readies CUDA kernels
  │   └─> FaceMatcher::init(database)                → builds FAISS-GPU index from DB embeddings
  │
  ├─> [IF CPU MODE]
  │   ├─> CPUMemoryPool::init(frame_size, batch)     → allocates RAM buffers
  │   ├─> CPUPreprocessor::init()                    → readies OpenCV operations
  │   └─> CPUFaceMatcher::init(database)             → builds FAISS-CPU index
  │
  ├─> FaceDatabase::open(db_path)                    → opens SQLite, loads all stored embeddings
  │
  └─> MetricsLogger::init(output_dir, buffer_size)   → starts async writer background thread
```

### Frame Processing Loop (`Pipeline::run`)

This is the hot path. Every frame goes through this sequence:

```
Pipeline::run(running)
  │
  └─> while (running) {
      │
      ├─> FFmpegSource::read(frame)                  → decode next frame to BGR24
      │       │
      │       └─ Returns: raw frame (1920x1080 BGR)
      │
      ├─> [GPU PATH: process_frame_gpu()]
      │   │
      │   ├─> GPUMemoryPool::upload_frame(frame)     → CPU→GPU transfer (PCIe)
      │   │
      │   ├─> Preprocessor::preprocess_yolo()        → CUDA: resize 1920x1080→640x640,
      │   │                                             BGR→RGB, HWC→CHW, normalize to [0,1]
      │   │
      │   ├─> FaceDetector::detect()                 → calls TritonClient for YOLO inference
      │   │       │                                     (uses CUDA shared memory - zero copy)
      │   │       └─> TritonClient::infer("yolo26_face", shm_input, shm_output)
      │   │               │
      │   │               └─ Returns: raw YOLO output [N x 8400 x 5]
      │   │
      │   ├─> apply_nms(detections)                  → confidence filter (>0.5) + IoU suppression
      │   │       │
      │   │       └─ Returns: filtered bounding boxes in original image coordinates
      │   │
      │   ├─> FaceCropper::crop_and_preprocess()     → CUDA: crop face regions from original frame,
      │   │                                             resize to 112x112, normalize for ArcFace
      │   │
      │   ├─> FaceRecognizer::recognize()            → calls TritonClient for ArcFace inference
      │   │       │                                     (uses CUDA shared memory)
      │   │       └─> TritonClient::infer("arcface", shm_input, shm_output)
      │   │               │
      │   │               └─ Returns: 512-dim L2-normalized embeddings per face
      │   │
      │   ├─> FaceMatcher::match(embeddings)         → FAISS-GPU cosine similarity search
      │   │       │
      │   │       └─ Returns: (person_id, similarity_score) per face
      │   │
      │   └─> ResultHandler::handle(results)         → packages detection + recognition results
      │
      ├─> [CPU PATH: process_frame_cpu()]
      │   │
      │   ├─> CPUMemoryPool::upload_frame(frame)     → RAM copy
      │   ├─> CPUPreprocessor::preprocess_yolo()     → OpenCV: resize, color convert, normalize
      │   ├─> FaceDetector::detectDirect()           → TritonClient with inline data (no SHM)
      │   ├─> apply_nms(detections)                  → same NMS algorithm
      │   ├─> CPUPreprocessor::preprocess_arcface()  → OpenCV: crop, resize to 112x112
      │   ├─> FaceRecognizer::recognizeDirect()      → TritonClient with inline data
      │   ├─> CPUFaceMatcher::match(embeddings)      → brute-force cosine search
      │   └─> ResultHandler::handle(results)
      │
      └─> MetricsLogger::log_frame(metrics)          → push timing data to lock-free ring buffer
                                                        (background thread flushes to CSV)
      }
```

---

## Module Dependency Map

Shows which module depends on (calls into) which other modules:

```
main.cpp
  └─> Pipeline
        ├─> FFmpegSource         (reads frames)
        ├─> TritonClient         (sends inference requests)
        │     ├─> GPUMemoryPool  (shared memory references)
        │     └─> CPUMemoryPool  (inline data)
        ├─> Preprocessor         (GPU preprocessing)
        │     └─> GPUMemoryPool  (reads/writes GPU buffers)
        ├─> CPUPreprocessor      (CPU preprocessing)
        │     └─> CPUMemoryPool  (reads/writes CPU buffers)
        ├─> FaceDetector         (YOLO detection)
        │     └─> TritonClient   (inference call)
        ├─> NMS                  (post-process detections)
        ├─> FaceCropper          (crop detected faces)
        │     └─> GPUMemoryPool  (GPU buffers)
        ├─> FaceRecognizer       (ArcFace embeddings)
        │     └─> TritonClient   (inference call)
        ├─> FaceMatcher          (GPU similarity search)
        │     └─> FaceDatabase   (loads known faces)
        ├─> CPUFaceMatcher       (CPU similarity search)
        │     └─> FaceDatabase   (loads known faces)
        ├─> ResultHandler        (output packaging)
        └─> MetricsLogger        (async CSV logging)
              └─> RingBuffer     (lock-free data passing)
```

---

## Python Tools (Separate from the C++ Pipeline)

### `tools/enroll_faces.py` - Database Population Tool

Run this **before** the pipeline to register known faces.

```
enroll_faces.py
  │
  ├─> Connect to Triton server
  ├─> init_database(db_path)              → creates SQLite schema (faces table)
  │
  └─> For each person folder in --faces-dir:
      ├─> For each image:
      │   ├─> preprocess_yolo(image)      → letterbox resize to 640x640
      │   ├─> run_yolo(triton, input)     → detect faces
      │   ├─> decode_yolo_output()        → parse bounding boxes
      │   ├─> nms(detections)             → filter overlaps
      │   ├─> [if exactly 1 face found]
      │   │   ├─> preprocess_arcface()    → crop & resize to 112x112
      │   │   └─> run_arcface()           → extract 512-dim embedding
      │   └─> [if 0 or >1 faces → skip]
      │
      ├─> Average all embeddings for this person
      ├─> L2-normalize the average
      └─> Insert (name, embedding) into SQLite
```

### `tools/download_models.py` - Model Setup Tool

Run this to create the Triton model repository.

```
download_models.py
  │
  ├─> [Default: Dummy mode]
  │   ├─> Create yolo26_face/1/model.onnx   → random-weight ONNX (correct I/O shapes)
  │   ├─> Create arcface/1/model.onnx       → random-weight ONNX (correct I/O shapes)
  │   └─> Write config.pbtxt for each       → Triton serving config
  │
  └─> [With --real flag]
      ├─> Download real YOLO face model
      ├─> Download real ArcFace model
      └─> Write config.pbtxt for each
```

---

## Configuration

### `config/pipeline.yaml`

```yaml
metrics:
  enabled: true
  output_dir: "./metrics/"
  flush_interval_ms: 1000     # how often background thread writes CSV
  buffer_size: 100            # ring buffer capacity
  rotate_size_mb: 100         # CSV file rotation threshold

pipeline:
  frame_width: 1920
  frame_height: 1080
  yolo_input_size: 640        # YOLO expects 640x640
  arcface_input_size: 112     # ArcFace expects 112x112
  max_faces_per_frame: 32

detection:
  confidence_threshold: 0.5   # minimum YOLO confidence
  nms_iou_threshold: 0.45     # IoU threshold for NMS

matching:
  similarity_threshold: 0.6   # minimum cosine similarity to declare a match
  faiss_ivf_threshold: 1000   # switch from flat to IVF index above this many faces
  faiss_nprobe: 16            # IVF search depth
```

### CLI Arguments (override YAML)

```
./process_image --input <video.mp4|rtsp://...> --triton <host:port> [options]

Required:
  --input <path|url>           Video source
  --triton <url>               Triton server (e.g., localhost:8001)

Optional:
  --config <path>              YAML config (default: config/pipeline.yaml)
  --db <path>                  Face database (default: ./faces.db)
  --confidence <float>         YOLO threshold (default: 0.5)
  --match-threshold <float>    Face similarity threshold (default: 0.6)
  --device <gpu|cpu|auto>      Force device (default: auto-detect)
  --output-csv <path>          Metrics output path
  --yolo-model <name>          Triton model name (default: yolo26_face)
  --arcface-model <name>       Triton model name (default: arcface)
```

---

## Build & Run Sequence

```
1. cmake -B build -DCMAKE_BUILD_TYPE=Release     # Configure
2. cmake --build build -j$(nproc)                 # Compile → build/process_image
3. python3 tools/download_models.py               # Create model repo
4. docker run ... nvcr.io/nvidia/tritonserver     # Start Triton with models
5. python3 tools/enroll_faces.py --faces-dir ./faces/ --db ./faces.db --triton localhost:8001
6. ./build/process_image --input video.mp4 --triton localhost:8001 --db ./faces.db
```

---

## External Dependencies

| Library | Purpose | Used By |
|---------|---------|---------|
| FFmpeg (libavcodec, libavformat, libswscale) | Video decoding | `ffmpeg_source.cpp` |
| NVIDIA Triton Client (gRPC) | Inference requests | `triton_client.cpp` |
| CUDA Toolkit | GPU kernels | `preprocessor.cu`, `memory_pool.cu`, `face_cropper.cu` |
| FAISS (GPU/CPU) | Similarity search | `face_matcher.cpp`, `cpu_face_matcher.cpp` |
| SQLite3 | Face database | `face_database.cpp` |
| yaml-cpp | Config loading | `config.h` |
| OpenCV | CPU image processing | `cpu_preprocessor.cpp` |
| GoogleTest | Unit testing | `tests/` |

---

## Inference Models (Running on Triton Server)

| Model | Input Shape | Output Shape | Purpose |
|-------|-------------|--------------|---------|
| `yolo26_face` | 1x3x640x640 FP32 | Nx8400x5 | Face detection (bbox + confidence) |
| `arcface` | Nx3x112x112 FP32 | Nx512 | Face embedding (512-dim vector) |

---

## Error Handling Summary

| Failure | What Happens |
|---------|--------------|
| Triton unreachable at startup | 3 retries with 1s backoff, then exit |
| gRPC call fails during processing | Skip that frame, log `total_pipeline_ms = -1` |
| FFmpeg can't decode a frame | Skip frame, log `decode_ms = -1` |
| No face match found (FAISS) | Return `face_id = -1, confidence = 0.0` (normal) |
| CUDA out of memory | Reduce pool size and retry; fall back to CPU mode |
| SQLite write failure | Return error; FAISS index stays consistent |
