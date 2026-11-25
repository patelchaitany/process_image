# process_image

Real-time face detection and recognition pipeline for MP4 files and RTSP streams. A C++17/CUDA application detects faces with **YOLO**, extracts **ArcFace** embeddings via **NVIDIA Triton Inference Server**, and matches identities against a **SQLite + FAISS** face gallery. Target latency is **< 20 ms per frame** (~13 ms typical on NVIDIA T4).

The project ships two runtimes:

| Binary | Purpose |
|--------|---------|
| `process_image` | Single-stream CLI for batch processing and benchmarking |
| `video_analytics_service` | Multi-camera gRPC service with webhook result delivery |

A **Streamlit client** (`client/app.py`) provides a UI for face registration and session management against the gRPC service.

**Many cameras on one GPU:** [`fastpipe/`](fastpipe/README.md) is a second implementation that keeps every frame on the GPU (C++/CUDA/TensorRT, no Triton): 40 dense 720p cameras on a 6 GB laptop GPU at 35% load, with zones per camera. That folder holds only the pipeline; the browser camera page, benchmarks, evaluators and test scripts are in [`fastpipe-dev/`](fastpipe-dev/README.md).

**GPU cluster runbook:** For a full copy-paste setup on Ubuntu 24.04 with NVIDIA T4 (GPU verify → `setup.sh` → Docker/Triton → model download → build → enroll → run), use [`commands.md`](commands.md).

---

## Architecture

### High-level system

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           process_image (this repo)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────────┐  │
│  │ FrameSource  │───▶│ Preprocessor │───▶│ TritonClient (gRPC + CUDA    │  │
│  │ NVDEC/FFmpeg │    │ CUDA / OpenCV│    │ shared memory)               │  │
│  └──────────────┘    └──────────────┘    └──────────────┬───────────────┘  │
│                                                          │                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────▼───────────────┐  │
│  │ ResultWriter │◀───│ FaceMatcher  │◀───│ PostProcess (NMS + crop)     │  │
│  │ console/csv/ │    │ FAISS-GPU/CPU│    │ FaceDetector + FaceRecognizer│  │
│  │ webhook      │    │ + SQLite DB  │    └──────────────────────────────┘  │
│  └──────────────┘    └──────────────┘                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ gRPC (inference only)
                    ┌───────────────────────────────┐
                    │  NVIDIA Triton Inference Server │
                    │  yolo26_face  │  arcface      │
                    │  (640×640)    │  (112×112)    │
                    └───────────────────────────────┘
```

**Design choice:** Triton + CUDA shared memory. Tensor data stays on GPU; gRPC carries only control metadata. Triton's dynamic batching enables multi-camera scaling through the service binary.

For the full call graph (init, per-frame loop, module dependencies), see [`project_architect.md`](project_architect.md).

### Per-frame data flow

```mermaid
flowchart LR
    A[MP4 / RTSP] --> B[FrameSource]
    B --> C[GPU Memory Pool]
    C --> D[YOLO Preprocess]
    D --> E[Triton: yolo26_face]
    E --> F[NMS + Face Crop]
    F --> G[Triton: arcface]
    G --> H[FAISS Match]
    H --> I[ResultHandler]
    I --> J[Console / CSV / Webhook]
```

| Stage | Component | Runs on |
|-------|-----------|---------|
| Decode | `NvdecSource` (GPU) or `FFmpegSource` + `PrefetchSource` (CPU) | GPU / CPU |
| Preprocess | `Preprocessor` (CUDA) or `CPUPreprocessor` (OpenCV) | GPU / CPU |
| Detection | `FaceDetector` → Triton `yolo26_face` | GPU (via Triton) |
| Post-process | `apply_nms`, `FaceCropper` | CPU + GPU |
| Recognition | `FaceRecognizer` → Triton `arcface` | GPU (via Triton) |
| Matching | `FaceMatcher` / `CPUFaceMatcher` + `FaceDatabase` | GPU / CPU |
| Output | `ResultHandler` → `ResultWriter` plugins | CPU |

### Multi-camera service architecture

```
Clients (gRPC)              video_analytics_service              Triton
─────────────               ─────────────────────────            ──────
StartSession  ──────────▶  SessionManager (≤30 sessions)
StopSession   ──────────▶       │
ListSessions  ──────────▶       ├── Session A (RTSP, own thread + GPU buffers)
                                ├── Session B (RTSP, own thread + GPU buffers)
                                └── Session N (MP4, own thread + GPU buffers)
                                         │
                                         ▼
                                SharedResources
                                (TritonClient, FaceDatabase, FaceMatcher)
                                         │
                                         ▼ webhook POST (JSON)
                                   Client callback URL
```

Details: [`docs/video-analytics-service.md`](docs/video-analytics-service.md) · API: [`docs/api-reference.md`](docs/api-reference.md)

---

## Repository layout

```
process_image/
├── src/                    # C++ pipeline core
│   ├── main.cpp            # CLI entry point → process_image
│   ├── service_main.cpp    # Service entry point → video_analytics_service
│   ├── pipeline/           # Orchestrator and result handling
│   ├── frame_source/       # NVDEC, FFmpeg, prefetch
│   ├── gpu/                # CUDA memory pool and preprocessors
│   ├── inference/          # Triton client, YOLO, ArcFace wrappers
│   ├── postprocess/        # NMS, face cropper
│   ├── matching/           # SQLite gallery + FAISS matcher
│   ├── output/             # ResultWriter plugins (console, CSV, webhook)
│   ├── service/            # gRPC server, session manager
│   └── metrics/            # Per-frame timing + async CSV logger
├── tools/                  # Python utilities (enroll, models, viz)
├── client/                 # Streamlit UI for the gRPC service
├── proto/                  # analytics.proto (gRPC API)
├── config/pipeline.yaml    # Runtime thresholds and geometry
├── commands.md             # GPU cluster setup & run commands (Ubuntu + T4)
├── faces/                  # Enrollment photo gallery (person subfolders)
├── tests/                  # GoogleTest unit tests
└── docs/                   # Setup, examples, design spec, phase notes
```

---

## Quick start

> **Running on a GPU server (Ubuntu 24.04 + T4)?** Follow the numbered runbook in [`commands.md`](commands.md) — it covers `setup.sh`, NVIDIA Container Toolkit, FAISS-GPU, real ONNX model download, Triton in Docker, build, enrollment, pipeline run, CLI flags, cleanup, and an optional MediaMTX RTSP server.

### Prerequisites

- CMake ≥ 3.18, C++17, FFmpeg, SQLite3, yaml-cpp
- **GPU mode (Linux):** CUDA ≥ 11.8, FAISS-GPU, Triton C++ client SDK
- **CPU mode (macOS / no GPU):** OpenCV, FAISS-CPU
- Docker (for Triton Inference Server)

| Guide | When to use |
|-------|-------------|
| [`commands.md`](commands.md) | **Primary** — end-to-end GPU cluster setup and run (Ubuntu 24.04, T4/A100/L4/H100 CUDA arch notes) |
| [`docs/setup.md`](docs/setup.md) | Cross-platform install (macOS CPU, Linux GPU, Docker, model repo, troubleshooting) |
| [`docs/examples.md`](docs/examples.md) | Usage patterns after the stack is running |

### Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

This produces `build/process_image`. When gRPC, Protobuf, and libcurl are available, `build/video_analytics_service` is built as well.

See [`commands.md` §8](commands.md#8-build-the-project) for GPU-specific `CMAKE_CUDA_ARCHITECTURES` overrides (A100, A10, L4, H100).

### Run (CLI)

```bash
# 1. Start Triton with model repository
docker run --gpus all --rm -p 8001:8001 \
  -v $(pwd)/models/model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models

# 2. Enroll known faces
python3 tools/enroll_faces.py \
  --faces-dir ./faces/ \
  --db ./faces.db \
  --triton localhost:8001

# 3. Process a video
./build/process_image \
  --input input/input_3.mp4 \
  --triton localhost:8001 \
  --db ./faces.db \
  --bbox-csv ./output/detections.csv
```

Full CLI flags and cleanup: [`commands.md` §10](commands.md#10-run-the-pipeline) · More examples (RTSP, CPU mode, metrics, visualization): [`docs/examples.md`](docs/examples.md)

### Run (gRPC service + Streamlit client)

```bash
# Start the service
./build/video_analytics_service \
  --triton localhost:8001 \
  --db ./faces.db \
  --port 50051

# Start the UI (separate terminal)
pip install -r client/requirements.txt
streamlit run client/app.py
```

---

## Documentation

### Getting started

| Document | Description |
|----------|-------------|
| [`commands.md`](commands.md) | **Operational runbook** — GPU verify, `setup.sh`, Container Toolkit, FAISS-GPU, `download_models.py --real`, Triton Docker, build, enroll, `process_image` run, all CLI flags, cleanup, MediaMTX |
| [`docs/setup.md`](docs/setup.md) | Platform support, dependencies, build, Triton model repo, troubleshooting |
| [`docs/examples.md`](docs/examples.md) | Enrollment, CLI usage, RTSP, metrics, bbox visualization |

### Architecture and design

| Document | Description |
|----------|-------------|
| [`project_architect.md`](project_architect.md) | Detailed architecture: call flow, module map, config, error handling |
| [`docs/superpowers/specs/2026-05-02-face-detection-recognition-pipeline-design.md`](docs/superpowers/specs/2026-05-02-face-detection-recognition-pipeline-design.md) | Original design spec: constraints, timing budget, component decisions |
| [`docs/video-analytics-service.md`](docs/video-analytics-service.md) | Multi-camera gRPC service design and scaling characteristics |
| [`docs/api-reference.md`](docs/api-reference.md) | gRPC API reference, webhook payload format, client examples |

### Implementation phases

Phase-by-phase notes live under [`docs/implementation-details/`](docs/implementation-details/):

| Phase | Topic |
|-------|-------|
| [0](docs/implementation-details/phase-0-scaffolding.md) | Project scaffolding, CMake, config |
| [1](docs/implementation-details/phase-1-frame-source.md) | FFmpeg frame source |
| [2](docs/implementation-details/phase-2-gpu-memory-preprocessing.md) | GPU memory pool and CUDA preprocessors |
| [3](docs/implementation-details/phase-3-triton-yolo.md) | Triton client and YOLO detector |
| [4](docs/implementation-details/phase-4-postprocess.md) | NMS and face cropper |
| [5](docs/implementation-details/phase-5-arcface.md) | ArcFace recognizer |
| [6](docs/implementation-details/phase-6-db-matcher.md) | SQLite gallery and FAISS matcher |
| [7](docs/implementation-details/phase-7-pipeline-cli.md) | Pipeline orchestrator and CLI |
| [8](docs/implementation-details/phase-8-metrics.md) | Per-frame metrics and async CSV |
| [9](docs/implementation-details/phase-9-enrollment.md) | Face enrollment tool |
| [10](docs/implementation-details/phase-10-cpu-fallback.md) | CPU fallback path |
| [11](docs/implementation-details/phase-11-error-handling.md) | Error handling and graceful shutdown |
| [13](docs/implementation-details/phase-13-bbox-logging.md) | Bbox CSV logging and ResultWriter plugins |

---

## Inference models (Triton)

| Model | Input | Output | Purpose |
|-------|-------|--------|---------|
| `yolo26_face` | `[N, 3, 640, 640]` FP32 | `[N, 8400, 5]` | Face detection (bbox + confidence) |
| `arcface` | `[N, 3, 112, 112]` FP32 | `[N, 512]` | 512-dim L2-normalized face embedding |

Create the model repository with `tools/download_models.py` (dummy or real weights). See [`docs/setup.md`](docs/setup.md#model-repository-setup).

---

## CLI reference

### `process_image`

```
Required:
  --input <path|url>       MP4 file or RTSP URL
  --triton <url>           Triton gRPC URL (default: localhost:8001)

Common options:
  --db <path>              Face database (default: ./faces.db)
  --config <path>          YAML config (default: config/pipeline.yaml)
  --bbox-csv <path>        Per-detection CSV (default: ./output/detections.csv)
  --output-csv <path>      Metrics CSV (default: ./metrics/metrics.csv)
  --device <gpu|cpu|auto>  Device mode (default: auto)
  --confidence <float>     YOLO threshold (default: 0.5)
  --match-threshold <float> Similarity threshold (default: 0.6)
  --no-console             Suppress per-frame stdout
```

### `video_analytics_service`

```
  --triton <url>            Triton URL (default: localhost:8001)
  --db <path>               Face database (default: ./faces.db)
  --port <int>              gRPC port (default: 50051)
  --max-sessions <int>      Max concurrent cameras (default: 30)
  --device <gpu|cpu|auto>   Device mode (default: auto)
```

Full gRPC API: [`docs/api-reference.md`](docs/api-reference.md)

---

## Python tools

| Script | Purpose |
|--------|---------|
| `tools/enroll_faces.py` | Build SQLite gallery from `faces/<person>/` photo folders |
| `tools/download_models.py` | Generate Triton model repository (dummy or real ONNX) |
| `tools/draw_bboxes.py` | Overlay detection CSV onto video for visualization |

Dependencies: `tools/requirements.txt`

---

## Tests

```bash
cd build
ctest --output-on-failure

# Or run individual tests
./test_nms
./test_face_database
./test_face_matcher
./test_triton_client
./test_cpu_preprocessor
```

---

## Future roadmap

Checklist of planned improvements and scaling work. Items marked **done** reflect the current `main` branch.

### Core pipeline

- [x] End-to-end CLI pipeline (`process_image`)
- [x] Triton gRPC client with CUDA shared memory
- [x] FAISS-GPU / FAISS-CPU face matching
- [x] NVDEC hardware decode with FFmpeg CPU fallback
- [x] CPU fallback mode (OpenCV preprocessing, inline Triton inference)
- [x] Per-frame metrics CSV with async writer
- [x] Modular `ResultWriter` plugins (console, CSV, webhook)
- [ ] FAISS IVF index for large galleries (> 1 000 faces) — config hooks exist in `pipeline.yaml`, flat index only today
- [ ] Reuse pooled device buffer in `FaceCropper` (avoid per-frame `cudaMalloc`)
- [ ] Additional result writers: JSON-lines, WebSocket push, database sink

### Multi-camera and service

- [x] gRPC `video_analytics_service` with session manager
- [x] Webhook-based result delivery per session
- [x] Streamlit client for registration and session control
- [ ] Load testing and tuning for 16+ concurrent cameras
- [ ] Session-level metrics aggregation and health dashboards
- [ ] gRPC streaming alternative to webhooks (optional transport)

### Operations and tooling

- [ ] Docker Compose stack (Triton + service + client) for one-command deploy
- [ ] CI pipeline (build matrix: GPU + CPU, unit tests)
- [ ] Automated enrollment validation (quality checks, duplicate detection)
- [ ] Model versioning and hot-swap via Triton model repository

### Performance

- [ ] Fused YOLO preprocess kernel benchmark and auto-tuning
- [ ] Zero-copy NVDEC → GPU memory pool path (eliminate remaining CPU staging)
- [ ] Triton dynamic batching tuning for multi-session workloads

---

## License

See repository license file (if present) or contact the project maintainers.
