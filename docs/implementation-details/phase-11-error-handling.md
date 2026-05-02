# Phase 11: Error Handling — Implementation Details

This document describes **Phase 11**: how startup failures, per-frame faults, resource teardown, and **signal** handling are implemented in **`TritonClient`**, **`Pipeline`**, and **`main`**. It maps to the **Error Handling** table in: [Face Detection & Recognition Pipeline Design](../superpowers/specs/2026-05-02-face-detection-recognition-pipeline-design.md).

---

## What Was Implemented

### Files, classes, and functions

| File | Responsibility |
|------|----------------|
| `src/inference/triton_client.h` | **`InferResult`**: **`success`**, **`error_msg`**, **`output_data`**, **`output_shape`**, Triton timing fields (**`queue_time_ns`**, **`compute_time_ns`**, **`compute_input_time_ns`**). **`TritonClient`**: **`connect(url, max_retries, retry_delay_ms)`**, **`disconnect()`**, CUDA SHM **register/unregister**, **`infer()`**. |
| `src/inference/triton_client.cpp` | **Retry loop** on connect; **`disconnect`** calls **`unregister_all_shm`**; **`infer`** returns **`InferResult`** with **`success=false`** when not connected. (gRPC body is **stubbed**; structure matches production error paths.) |
| `src/pipeline/pipeline.h` | **`Pipeline`**: **`init`**, **`run(std::atomic<bool>&)`**, **`shutdown()`**, **`detect_device()`**, **`process_frame`**. |
| `src/pipeline/pipeline.cpp` | **Device detection**, **Triton connect** with retries, **GPU OOM** retry / CPU fallback, **decode** error handling, **`try`/`catch`** around **`process_frame`**, early returns on **YOLO/ArcFace** failure, **`shutdown`** ordering. |
| `src/main.cpp` | **`SIGINT` / `SIGTERM`** handler setting **`std::atomic<bool> g_running`**; **`pipeline.run(g_running)`** then **`pipeline.shutdown()`**. |
| `src/matching/face_database.cpp` | **`FaceDatabase::add_face`** returns **`bool`** (**`SQLITE_DONE`** only on success)—enrollment callers must **not** update **FAISS** / in-memory index when this is **false**. |

---

## Architecture Decisions

### 1. Error taxonomy (six spec failure modes)

| Failure mode | What happens in code / intent |
|--------------|-------------------------------|
| **Triton unreachable at startup** | **`TritonClient::connect`** loops up to **`max_retries`** with **`retry_delay_ms`** sleep between attempts; on exhaustion returns **`false`**. **`Pipeline::init`** logs **FATAL** and aborts init. |
| **Triton gRPC fails mid-pipeline** | Callers (**`FaceDetector::detect`**, **`FaceRecognizer::recognize`**) should surface failure; **`process_frame`** treats **`!yolo_ok`** as early **return**; **`!arcface_ok`** returns **detections without matches**. Outer **`run`** wraps **`process_frame`** in **`try`/`catch`** for exceptions. |
| **FFmpeg decode error** | **`run`**: if **`read`** fails but source **`is_open`**, treat as corrupt/skip: **`decode_ms = -1`**, increment **`frames_dropped`**, **`continue`**. |
| **FAISS no match** | Not an error: **`MatchResult`** with **`face_id == -1`**, **`confidence == 0.0`** (**`CPUFaceMatcher`** threshold gate; **`FaceMatcher`** follows same contract when fully implemented). |
| **CUDA OOM** | **`GPUMemoryPool::init`** failure → retry with **half** **`max_faces_per_frame`**; second failure → **`use_gpu_ = false`**, **`memory_pool_.reset()`**, proceed toward **CPU path** (full CPU pool wiring is per Phase 10). |
| **SQLite write failure on enrollment** | **`FaceDatabase::add_face`** returns **`false`**; spec: **do not** add to **FAISS** / **`FaceMatcher::add_face`** if persistence failed—**transactional consistency**: on-disk DB is **source of truth**. |

**Reasoning:** Separates **fatal startup** (no stream, no Triton) from **soft per-frame** (bad packet, flaky RPC) from **domain negative** (unknown face).

---

### 2. Recovery strategies per failure mode

| Mode | Strategy |
|------|----------|
| **Triton startup** | **Retry with backoff** (`connect` parameters **`3`**, **`1000`** ms in **`Pipeline::init`**). |
| **gRPC mid-pipeline** | **Skip or degrade**: spec **skip frame** + metrics counter; code path uses **early return** / partial **`FrameResult`** for ArcFace failure; **exception** path sets **`total_pipeline_ms = -1`**, **`frames_dropped++`**. |
| **Decode** | **Skip frame**, log via **`decode_ms = -1`** in metrics. |
| **No match** | Return **unknown** identity; pipeline continues. |
| **CUDA OOM** | **Reduce pool** (half faces), then **CPU fallback** flag. |
| **SQLite** | **Return error to caller**; no index update. |

**Reasoning:** Avoid stopping the stream on single-frame glitches; reserve **hard exit** for **configuration** and **connectivity** errors at init.

---

### 3. Graceful shutdown sequence

**Decision:** **`Pipeline::shutdown()`** (`src/pipeline/pipeline.cpp`) order:

1. **`MetricsLogger::shutdown()`** — flush async CSV / drain buffers (**metrics first** so late frames still recorded if possible).
2. **`GPUMemoryPool::unregister_triton_shm()`** then **`release()`** — release **CUDA** buffers after unregistering from Triton.
3. **`TritonClient::disconnect()`** — **`unregister_all_shm`** inside client, clear connection.
4. **`FaceDatabase::close()`** — finalize SQLite.
5. **`FFmpegSource::close()`** — release demuxer/decoder.
6. Reset remaining subsystems (**detector**, **recognizer**, **matcher**, etc.).

**Reasoning:**

- **Flush metrics** before tearing down I/O consumers.
- **Unregister shared memory** before destroying the **device pointers** Triton might still resolve.
- **DB** closed before process exit; **frame source** closed last among external resources after inference is done.

This matches the spec intent: **metrics flush → Triton SHM unregister → GPU pool release → Triton disconnect → DB close → frame source close** (with **`metrics_logger`** explicitly handling the metrics leg).

---

### 4. Resource cleanup vs initialization order

**Initialization** (`Pipeline::init`): **frame source** → **Triton** → **GPU memory pool** (+ register) → **preprocessor** → **detector/recognizer** → **database** → **matcher** → **cropper** / **result** → **metrics**.

**Shutdown** is **not** a perfect mirror of every middle step but respects **dependencies**: release **GPU-backed Triton regions** before **disconnecting**; close **DB** and **source** after processing stops; drop **unique_ptrs** to detectors last to avoid use-after-free if any background work referenced them (defensive ordering).

---

### 5. Consistency: SQLite + FAISS / matcher

**Decision:** **`FaceDatabase::add_face`** is **`INSERT`** with **`sqlite3_step` == `SQLITE_DONE`** check. **`FaceMatcher::add_face`** / **`CPUFaceMatcher::add_face`** must only run **after** a successful DB insert (or on **startup reload** from DB).

**Reasoning:** If SQLite fails (disk full, locked DB), the **in-memory index** must **not** gain a row the DB does not have—otherwise **IDs** and **names** diverge after restart.

---

### 6. SIGINT / SIGTERM handling

**Decision:** **`main`** installs **`signal_handler`** for **`SIGINT`** and **`SIGTERM`** that sets **`g_running`** to **`false`** with **`memory_order_relaxed`** and prints a short message. **`Pipeline::run`** evaluates **`running.load`** **each loop iteration**.

**Reasoning:** Cooperative shutdown: current frame may finish **`process_frame`** before the loop notices the flag; destructor **`~Pipeline`** also calls **`shutdown()`** for **RAII** safety if **`run`** returns.

---

## Key Code Snippets

**Triton connect retries:**

```10:41:src/inference/triton_client.cpp
bool TritonClient::connect(const std::string& url, int max_retries, int retry_delay_ms) {
    url_ = url;

    for (int attempt = 1; attempt <= max_retries; ++attempt) {
        fprintf(stderr, "TritonClient: connecting to %s (attempt %d/%d)...\n",
                url.c_str(), attempt, max_retries);
        // ...
        if (attempt < max_retries) {
            fprintf(stderr, "TritonClient: retrying in %dms...\n", retry_delay_ms);
            std::this_thread::sleep_for(std::chrono::milliseconds(retry_delay_ms));
        }
    }

    fprintf(stderr, "TritonClient: FAILED to connect after %d attempts. Exiting.\n", max_retries);
    return false;
}
```

**Init: Triton failure fatal; GPU OOM degrade:**

```48:69:src/pipeline/pipeline.cpp
    triton_client_ = std::make_unique<TritonClient>();
    if (!triton_client_->connect(config_.triton_url, 3, 1000)) {
        fprintf(stderr, "FATAL: Cannot reach Triton at %s after 3 retries. Exiting.\n",
                config_.triton_url.c_str());
        return false;
    }

    // Initialize GPU memory pool (with CUDA OOM recovery)
    if (use_gpu_) {
        memory_pool_ = std::make_unique<GPUMemoryPool>();
        if (!memory_pool_->init(frame_source_->width(), frame_source_->height(),
                                 config_.max_faces_per_frame)) {
            fprintf(stderr, "Warning: GPU memory allocation failed, attempting with reduced pool\n");
            // Retry with half the max faces
            if (!memory_pool_->init(frame_source_->width(), frame_source_->height(),
                                     config_.max_faces_per_frame / 2)) {
                fprintf(stderr, "Warning: GPU still OOM, falling back to CPU mode\n");
                use_gpu_ = false;
                memory_pool_.reset();
            }
        }
```

**Decode error vs EOF; gRPC exception → drop:**

```132:172:src/pipeline/pipeline.cpp
    while (running.load(std::memory_order_relaxed)) {
        // ...
        bool got_frame = frame_source_->read(frame);
        // ...
        if (!got_frame) {
            if (frame_source_->is_open()) {
                metrics.decode_ms = -1.0f;
                frames_dropped++;
                metrics_logger_->log_frame(metrics);
                continue;
            }
            break;  // EOF
        }
        // ...
        bool success = false;
        try {
            process_frame(frame, metrics);
            success = true;
        } catch (...) {
            success = false;
        }

        if (!success) {
            frames_dropped++;
            metrics.total_pipeline_ms = -1.0f;
            metrics_logger_->log_frame(metrics);
            continue;
        }
```

**YOLO / ArcFace soft failure inside `process_frame`:**

```213:276:src/pipeline/pipeline.cpp
    bool yolo_ok = face_detector_->detect("yolo_input_shm", "yolo_output_shm",
                                           raw_detections, config_.confidence_threshold);
    // ...
    if (!yolo_ok) {
        // Triton gRPC call failed mid-pipeline: skip frame
        return;
    }
    // ...
    bool arcface_ok = face_recognizer_->recognize(
        "arcface_input_shm", "arcface_output_shm",
        static_cast<int>(detections.size()), embeddings);
    // ...
    if (!arcface_ok) {
        // ArcFace gRPC failed - report detections without matches
        FrameResult result;
        result.frame_id = frame.frame_index;
        result.detections = detections;
        result_handler_->handle(result);
        return;
    }
```

**Shutdown ordering:**

```293:321:src/pipeline/pipeline.cpp
void Pipeline::shutdown() {
    if (metrics_logger_) {
        metrics_logger_->shutdown();
        metrics_logger_.reset();
    }
    if (memory_pool_) {
        memory_pool_->unregister_triton_shm();
        memory_pool_->release();
        memory_pool_.reset();
    }
    if (triton_client_) {
        triton_client_->disconnect();
        triton_client_.reset();
    }
    if (face_database_) {
        face_database_->close();
        face_database_.reset();
    }
    if (frame_source_) {
        frame_source_->close();
        frame_source_.reset();
    }
    face_detector_.reset();
    face_recognizer_.reset();
    face_matcher_.reset();
    // ...
}
```

**Signal handling:**

```8:14:src/main.cpp
static std::atomic<bool> g_running{true};

static void signal_handler(int sig) {
    (void)sig;
    g_running.store(false, std::memory_order_relaxed);
    fprintf(stderr, "\nInterrupted. Shutting down...\n");
}
```

```16:18:src/main.cpp
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
```

**SQLite insert success gate:**

```62:64:src/matching/face_database.cpp
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    return rc == SQLITE_DONE;
```

---

## Verification Approach

1. **Triton down:** Start pipeline with invalid URL; confirm **3** attempts and **failed `init`** (non-zero exit from **`main`**).
2. **Decode:** Mock **`FFmpegSource::read`** to return **`false`** with **`is_open() == true`**; expect **`decode_ms == -1`** row and **`frames_dropped`** increment in logs.
3. **Exception path:** Inject a **throw** inside **`process_frame`** in a test build; expect **`total_pipeline_ms == -1`** and **drop** increment (**note:** purely **`bool` false`** returns from **`detect`** without exception are handled inside **`process_frame`**; align detector implementation with spec **skip + drop** if counters must match).
4. **Shutdown:** Run under **strace**/Instruments or CUDA **`compute-sanitizer`**; verify **no use-after-free** and that **unregister** precedes **pool release** (order in **`shutdown`**).
5. **Enrollment:** Unit-test **`FaceDatabase::add_face`** failure (read-only DB) and assert **matcher `add_face` not invoked** in the enrollment façade (when present).
6. **SIGINT:** Run live stream, **Ctrl+C**, confirm **"Interrupted"** message and clean exit via **`shutdown()`** after loop exit.
