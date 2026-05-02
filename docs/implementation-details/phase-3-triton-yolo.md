# Phase 3: Triton Client & YOLO Face Detector — Implementation Details

This document describes **Phase 3** (client side): **`TritonClient`** for gRPC-based inference with **CUDA shared memory** registration, **`InferResult`** for outputs and Triton timing stats, and **`FaceDetector`** wrapping YOLO **`infer`** plus confidence filtering. It aligns with the design spec: [Face Detection & Recognition Pipeline Design](../superpowers/specs/2026-05-02-face-detection-recognition-pipeline-design.md).

---

## What Was Implemented

### Files, classes, and functions

| File | Role |
|------|------|
| `src/inference/triton_client.h` | **`InferResult`** struct; **`TritonClient`** with **`connect`**, **`disconnect`**, **`register_cuda_shm`**, **`unregister_cuda_shm`**, **`unregister_all_shm`**, **`infer`**. |
| `src/inference/triton_client.cpp` | Retry loop for **`connect`** (structure for 3 attempts / delay); placeholder “connected” path; stub **`register`** / **`infer`** with TODOs pointing to **`triton::client::InferenceServerGrpcClient`**. |
| `src/inference/face_detector.h` | **`RawDetection`** (`cx`, `cy`, `w`, `h`, **`confidence`**); **`FaceDetector`** using a **`TritonClient&`** and model name. |
| `src/inference/face_detector.cpp` | Builds **`[1,3,640,640]`** input and **`[1,8400,5]`** output shapes; calls **`infer`**; parses rows **`[cx, cy, w, h, conf]`** above **`confidence_threshold`**. |
| `tests/test_triton_client.cpp` | Tests **infer** without connection (failure), **register** without connection (**false**), placeholder **connect** behavior, disabled live-server test. |

**Important implementation note:** The Triton C++ gRPC client calls are **not wired yet** (comments/TODOs). Behavior today is **stubbed** for compilation and interface freeze; the sections below document **intended architecture** as reflected in headers, call shapes, and the design spec, and call out gaps for verification.

---

## Architecture Decisions

### 1. gRPC vs HTTP for Triton

**Decision:** The code and CMake/tooling target the **gRPC** Triton client (`InferenceServerGrpcClient`, default port **8001** in the spec). HTTP/REST is not implemented in this module.

**Reasoning:**

- **Lower latency** for small control messages: protobuf over HTTP/2 vs. REST framing.
- **Streaming and advanced APIs** (where needed) align with production Triton deployments.
- The Python enrollment tool in the spec explicitly uses **`tritonclient[grpc]`** — parity between languages.

### 2. CUDA shared memory lifecycle: register once, reference by name

**Decision (target):**

- At init (or pool setup), call **`RegisterCudaSharedMemory`** (or equivalent) **once per region** with a stable **`name`**, **`gpu_ptr`**, and **byte size**.
- Each **`infer`** request **references** input/output tensor data by **that name** — **no tensor payload** in the gRPC body.

**Reasoning:**

- Matches the design doc’s **zero-copy** pattern: **PCIe** is paid on **`cudaMemcpy`** into the client-owned buffer (Stage 2), not on every inference RPC.
- **Unregister** on shutdown (`disconnect` → **`unregister_all_shm`**) avoids server-side stale handles after the GPU allocations are freed.

**Current code:** **`register_cuda_shm`** logs and returns **`true`**; production must replace this with the real **`triton::client`** registration API and track names for cleanup.

### 3. Retry / backoff on connect

**Decision:** **`connect(url, max_retries = 3, retry_delay_ms = 1000)`** loops **1…max_retries**, logging each attempt.

**Reasoning:** Matches the spec error table: **“Retry 3 times with 1s backoff”** if Triton is not reachable at startup.

**Current code caveat:** The placeholder implementation **returns success on the first attempt** before the retry **`sleep`** path runs; integrating the real client should **only** set **`connected_ = true`** after **`IsServerLive`** (or equivalent) succeeds, and **`sleep_for`** between failures until attempts are exhausted.

### 4. `InferResult` abstraction

**Decision:** **`InferResult`** carries:

- **`output_data`**: `const float*` into **shared memory** (or mapped CPU buffer in CPU fallback mode — future).
- **`output_shape`**: `std::vector<int64_t>` e.g. **`{1, 8400, 5}`**.
- **`success`**, **`error_msg`**.
- **`queue_time_ns`**, **`compute_time_ns`**, **`compute_input_time_ns`** — populated from Triton response statistics.

**Reasoning:**

- **Single object** returned from **`infer`** avoids out-parameters scattered across the codebase.
- **Timing** mirrors the spec’s CSV metrics (**triton_yolo_queue_ms**, **triton_yolo_compute_ms**) and validates **shared memory** ( **`compute_input_time_ns`** near zero when input is already on GPU).

**Current code:** Placeholder **`infer`** sets **`success = true`** but **`output_data = nullptr`** — **`FaceDetector`** parsing will require a real pointer in production.

### 5. `FaceDetector`: thin adapter over shapes and thresholds

**Decision:** Hard-coded **`input_shape`** **`{1,3,640,640}`** and **`output_shape`** **`{1,8400,5}`** for **`yolo26_face`**; filters **`conf`** ≥ threshold.

**Reasoning:**

- Keeps detection policy **decoupled** from transport; swapping models means changing **`model_name_`** and shapes in one place.
- Output layout **`cx, cy, w, h, conf`** matches the spec’s **“[N,5] detections (cxcywh + conf)”** description (here **`N=8400`** anchors).

### 6. Zero-copy pattern (target end state)

**Decision:** gRPC carries **model name**, **tensor names**, **shapes**, **IO tensor shared-memory identifiers**, and receives **metadata + timing**; **float tensors** live in **pre-registered** CUDA buffers.

**Reasoning:** Minimizes RPC bandwidth and CPU overhead — consistent with **Stage 4–6** in the design doc.

---

## Reasoning Summary

| Topic | Choice | Why |
|--------|--------|-----|
| Protocol | gRPC | Latency, parity with `tritonclient[grpc]`, spec default `8001` |
| SHM | Register once, infer by name | Zero-copy tensors, server resolves GPU pointer |
| Connect | 3 × 1s backoff (intended) | Spec’d startup resilience |
| Result type | `InferResult` | Output pointer + shape + Triton timings |
| Detector | `FaceDetector` wrapper | Centralizes YOLO tensor contract |

---

## Key Code Snippets

**`InferResult` and `TritonClient` API:**

```7:40:src/inference/triton_client.h
struct InferResult {
    const float* output_data = nullptr;
    std::vector<int64_t> output_shape;
    bool success = false;
    std::string error_msg;
    float queue_time_ns = 0;
    float compute_time_ns = 0;
    float compute_input_time_ns = 0;
};

class TritonClient {
public:
    TritonClient() = default;
    ~TritonClient();

    bool connect(const std::string& url, int max_retries = 3, int retry_delay_ms = 1000);
    void disconnect();
    bool is_connected() const { return connected_; }

    bool register_cuda_shm(const std::string& name, void* gpu_ptr, size_t byte_size);
    bool unregister_cuda_shm(const std::string& name);
    void unregister_all_shm();

    InferResult infer(const std::string& model_name,
                      const std::string& input_shm_name,
                      const std::vector<int64_t>& input_shape,
                      const std::string& output_shm_name,
                      const std::vector<int64_t>& output_shape);

private:
    std::string url_;
    bool connected_ = false;
    void* client_ = nullptr;  // opaque triton client ptr
};
```

**Connect retry structure (placeholder body — replace with real client):**

```10:41:src/inference/triton_client.cpp
bool TritonClient::connect(const std::string& url, int max_retries, int retry_delay_ms) {
    url_ = url;

    for (int attempt = 1; attempt <= max_retries; ++attempt) {
        fprintf(stderr, "TritonClient: connecting to %s (attempt %d/%d)...\n",
                url.c_str(), attempt, max_retries);

        // TODO: Replace with actual triton::client::InferenceServerGrpcClient::Create()
        // ...

        // Placeholder: simulate connection success
        connected_ = true;
        fprintf(stderr, "TritonClient: connected to %s\n", url.c_str());
        return true;

        if (attempt < max_retries) {
            fprintf(stderr, "TritonClient: retrying in %dms...\n", retry_delay_ms);
            std::this_thread::sleep_for(std::chrono::milliseconds(retry_delay_ms));
        }
    }

    fprintf(stderr, "TritonClient: FAILED to connect after %d attempts. Exiting.\n", max_retries);
    return false;
}
```

**`FaceDetector::detect` parsing contract:**

```6:36:src/inference/face_detector.cpp
bool FaceDetector::detect(const std::string& input_shm_name,
                           const std::string& output_shm_name,
                           std::vector<RawDetection>& detections,
                           float confidence_threshold) {
    std::vector<int64_t> input_shape = {1, 3, 640, 640};
    std::vector<int64_t> output_shape = {1, 8400, 5};

    InferResult result = client_.infer(model_name_, input_shm_name,
                                        input_shape, output_shm_name, output_shape);

    if (!result.success) return false;

    // Parse output: [8400, 5] where each row is [cx, cy, w, h, conf]
    detections.clear();
    const float* data = result.output_data;
    int num_detections = static_cast<int>(result.output_shape[1]);

    for (int i = 0; i < num_detections; ++i) {
        float conf = data[i * 5 + 4];
        if (conf >= confidence_threshold) {
            RawDetection det;
            det.cx = data[i * 5 + 0];
            det.cy = data[i * 5 + 1];
            det.w = data[i * 5 + 2];
            det.h = data[i * 5 + 3];
            det.confidence = conf;
            detections.push_back(det);
        }
    }

    return true;
}
```

---

## Verification Approach

1. **Unit tests** (`tests/test_triton_client.cpp`):
   - **`InferWithoutConnection`**: expect **`success == false`** and non-empty **`error_msg`**.
   - **`RegisterShmWithoutConnection`**: expect **`false`**.
   - After real client integration, **`ConnectToInvalidUrl`** should fail after retries (today’s placeholder may incorrectly succeed — update expectations when stubs are removed).
2. **Integration (disabled):** **`DISABLED_ConnectToLiveServer`** — enable when Triton listens on **`localhost:8001`**; exercise **`connect`**, **`register_cuda_shm`** with a real **`cudaMalloc`** pointer, **`disconnect`**.
3. **Shared memory sanity:** With production **`infer`**, confirm **`compute_input_time_ns`** in **`InferResult`** is negligible for GPU-backed inputs versus non-SHM paths.
4. **End-to-end:** Run **`FaceDetector`** against a known video frame tensor; compare raw **`RawDetection`** counts to a reference (Python **`tritonclient`** script) before NMS.
5. **Safety gap to fix with stubs:** When **`infer`** returns **`success`** but **`output_data == nullptr`**, **`FaceDetector`** must not dereference **`data`** — guard with **`if (!result.output_data) return false;`** until the client is complete.
