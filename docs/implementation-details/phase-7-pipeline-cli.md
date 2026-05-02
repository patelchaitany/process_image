# Phase 7: Pipeline orchestration & CLI

This document describes how the runtime pipeline is owned, initialized, and driven from the command line, how graceful shutdown works, and how per-frame results are surfaced to the operator.

## What was implemented

### Files, classes, and functions

| File | Role |
|------|------|
| `src/pipeline/pipeline.h` / `pipeline.cpp` | `Pipeline` class: `init`, `run`, `shutdown`, `process_frame`, `detect_device` |
| `src/pipeline/result_handler.h` / `result_handler.cpp` | `ResultHandler::handle`, `FrameResult` struct |
| `src/main.cpp` | `main`, global `g_running`, POSIX signal handlers |
| `src/utils/config.h` | `PipelineConfig`, `parse_args` |

**`Pipeline`** holds every stage as a private `std::unique_ptr` member and is the single owner of those objects. Public API:

- `bool init(const PipelineConfig& config)` — builds the graph in dependency order.
- `void run(std::atomic<bool>& running)` — decode loop; processes frames until `running` is false or input EOF.
- `void shutdown()` — stops metrics I/O, releases GPU/Triton/DB resources, resets components.
- `void process_frame(Frame& frame, FrameMetrics& metrics)` — one frame through upload → YOLO → NMS → optional ArcFace → FAISS → results.

**`ResultHandler`** formats human-readable stdout for each `FrameResult` (frame id, per-face bbox, detection confidence, match name and score).

**`parse_args`** walks `argc`/`argv` with a simple `for` loop and string comparisons; no third-party CLI library.

**`main`** installs signal handlers, parses config, constructs a stack-allocated `Pipeline`, calls `init`, `run`, and `shutdown`.

## Architecture decisions

### Single owner with `unique_ptr` for all components

Every subsystem (`FFmpegSource`, `GPUMemoryPool`, `Preprocessor`, `TritonClient`, detector/recognizer, `FaceDatabase`, `FaceMatcher`, `FaceCropper`, `ResultHandler`, `MetricsLogger`) is stored as `std::unique_ptr` on `Pipeline`. That enforces a single ownership tree, deterministic teardown order in `shutdown()`, and no accidental shared ownership or leaks across threads.

**Reasoning:** The pipeline is a linear composition of stages. Exclusive ownership matches that model, keeps headers small, and makes resource lifetime obvious when extending the graph.

### Initialization order (actual code)

`Pipeline::init` wires dependencies in this order:

1. **Device policy** — `detect_device()` sets `use_gpu_` from `--device` and `cudaGetDeviceCount()`.
2. **Frame source** — `FFmpegSource::open(config_.input_source)` (must know geometry before GPU buffers).
3. **Triton client** — `TritonClient::connect(..., 3 retries, 1000 ms backoff)` per design spec.
4. **GPU memory pool** (if GPU) — dimensions from frame source + `max_faces_per_frame`; on failure, retry with half the face budget; on second failure, CPU fallback. If pool exists, register CUDA shared memory with Triton.
5. **Preprocessor** — constructed before inference so YOLO preprocessing can run on the pool’s buffers.
6. **FaceDetector / FaceRecognizer** — both reference `*triton_client_` and model names from config.
7. **FaceDatabase** — `open(config_.db_path)` and row count log.
8. **FaceMatcher** — `init(*face_database_, match_threshold, use_gpu_)`.
9. **FaceCropper** — used later in `process_frame` for ArcFace input.
10. **ResultHandler** — stateless formatter.
11. **MetricsLogger** — `init(metrics_dir)` where `metrics_dir` is derived from `--output-csv` (parent path, or `./metrics` if no `/`).

**Reasoning:** Triton must exist before detectors and before registering shared memory. The frame source must be open before allocating GPU buffers sized to the stream. The database must load before FAISS/GPU matcher setup. Cropper and result handler have no cross-dependencies but are created before the run loop.

### Frame loop design

`Pipeline::run` allocates a `Frame` and `FrameMetrics`, then:

```cpp
while (running.load(std::memory_order_relaxed)) {
    total_timer.start();
    metrics = FrameMetrics{};
    // decode ...
    if (!got_frame) {
        if (frame_source_->is_open()) {
            metrics.decode_ms = -1.0f;  // decode error, stream still open
            metrics_logger_->log_frame(metrics);
            continue;
        }
        break;  // EOF
    }
    metrics.decode_ms = ...
    try {
        process_frame(frame, metrics);
        success = true;
    } catch (...) {
        success = false;
    }
    if (!success) {
        metrics.total_pipeline_ms = -1.0f;
        metrics_logger_->log_frame(metrics);
        continue;
    }
    total_timer.stop();
    metrics.total_pipeline_ms = total_timer.elapsed_ms();
    metrics_logger_->log_frame(metrics);
}
```

**Reasoning:** The atomic `running` flag decouples the hot loop from signal delivery. Relaxed ordering is sufficient for a boolean “should I exit?” flag. Decode errors while the source remains open are recorded (`decode_ms = -1`) and counted as dropped frames. `process_frame` returns early on Triton failure; outer `try/catch` handles unexpected exceptions without crashing the process.

### Manual CLI parsing

`parse_args` implements flags exactly as listed in the product spec (`--input`, `--triton`, `--output-csv`, `--db`, `--confidence`, `--match-threshold`, `--device`, `--yolo-model`, `--arcface-model`, `-h`/`--help`). Values are stored in `PipelineConfig`; `--help` prints to stderr and `exit(0)`.

**Note:** Only `--input` is validated as required. `--triton` defaults to `localhost:8001` in `PipelineConfig` if omitted, which matches ergonomic local development; callers can still pass an explicit URL.

**Reasoning:** Avoids a dependency on CLI libraries, keeps the binary self-contained, and mirrors the spec’s flag names one-to-one.

### SIGINT / SIGTERM handling

A file-scope `std::atomic<bool> g_running{true}` is flipped to `false` in `signal_handler`, registered for `SIGINT` and `SIGTERM` before parsing arguments.

**Reasoning:** RTSP and long MP4 runs need a clean stop path: the next loop iteration sees `running == false`, the loop exits, and `main` calls `pipeline.shutdown()` so metrics flush and handles unregister.

### ResultHandler output format

For each `FrameResult`, stdout shows:

- One line per frame: `Frame <id>: <n> faces detected`
- Per detection: `bbox=(x1,y1,x2,y2)`, `det_conf`, `match=<name|unknown>`, match confidence

Unknown or non-matching faces use `"unknown"` with confidence `0.0` when there is no valid `MatchResult` at that index or `face_id < 0`.

**Reasoning:** Stdout is the simplest integration surface for demos, systemd/journal, or piping without adding a logging framework dependency to this phase.

## Key code snippets

**Ownership model (header):**

```32:46:src/pipeline/pipeline.h
    PipelineConfig config_;
    bool initialized_ = false;
    bool use_gpu_ = true;

    std::unique_ptr<FFmpegSource> frame_source_;
    std::unique_ptr<GPUMemoryPool> memory_pool_;
    std::unique_ptr<Preprocessor> preprocessor_;
    std::unique_ptr<TritonClient> triton_client_;
    std::unique_ptr<FaceDetector> face_detector_;
    std::unique_ptr<FaceRecognizer> face_recognizer_;
    std::unique_ptr<FaceDatabase> face_database_;
    std::unique_ptr<FaceMatcher> face_matcher_;
    std::unique_ptr<FaceCropper> face_cropper_;
    std::unique_ptr<ResultHandler> result_handler_;
    std::unique_ptr<MetricsLogger> metrics_logger_;
```

**Result handler output:**

```4:20:src/pipeline/result_handler.cpp
void ResultHandler::handle(const FrameResult& result) {
    printf("Frame %lu: %zu faces detected\n",
           result.frame_id, result.detections.size());

    for (size_t i = 0; i < result.detections.size(); ++i) {
        const auto& det = result.detections[i];
        const char* name = "unknown";
        float conf = 0.0f;

        if (i < result.matches.size() && result.matches[i].face_id >= 0) {
            name = result.matches[i].name.c_str();
            conf = result.matches[i].confidence;
        }

        printf("  [%zu] bbox=(%.0f,%.0f,%.0f,%.0f) det_conf=%.3f match=%s (%.3f)\n",
               i, det.x1, det.y1, det.x2, det.y2, det.confidence, name, conf);
    }
}
```

**Signal handling and driver:**

```8:44:src/main.cpp
static std::atomic<bool> g_running{true};

static void signal_handler(int sig) {
    (void)sig;
    g_running.store(false, std::memory_order_relaxed);
    fprintf(stderr, "\nInterrupted. Shutting down...\n");
}

int main(int argc, char* argv[]) {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    PipelineConfig config = parse_args(argc, argv);
    // ...
    Pipeline pipeline;
    if (!pipeline.init(config)) {
        fprintf(stderr, "Pipeline initialization failed\n");
        return 1;
    }

    pipeline.run(g_running);
    pipeline.shutdown();
```

## Verification approach

1. **Build** the `process_image` target with CMake and run with `--help`; confirm usage text matches the spec.
2. **Init failure:** wrong path for `--input` → expect non-zero exit and clear stderr from `FFmpegSource`.
3. **Triton retry:** with no server listening, expect three retries then fatal message and failed `init`.
4. **Run/stop:** start against a short MP4; send Ctrl+C during RTSP or long file and confirm “Interrupted” plus `shutdown` path (no crash); confirm “Processed / Dropped” counts look sane.
5. **Results:** with a populated DB and Triton up, confirm stdout lines include bbox integers, `det_conf`, and known names when `FaceMatcher` returns positive IDs.
