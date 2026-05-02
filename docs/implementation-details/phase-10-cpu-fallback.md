# Phase 10: CPU Fallback — Implementation Details

This document describes **Phase 10**: CPU-side types that **mirror** the GPU memory pool, CUDA preprocessor, and FAISS-style matcher so the pipeline can run without a CUDA device. It aligns with the design spec: [Face Detection & Recognition Pipeline Design](../superpowers/specs/2026-05-02-face-detection-recognition-pipeline-design.md) (CPU Fallback table and runtime polymorphism).

---

## What Was Implemented

### Files, classes, and functions

| File | Role |
|------|------|
| `src/gpu/cpu_memory_pool.h` | **`CPUBuffers`**: `std::vector` backing stores for **raw_frame**, **yolo_input** `[1,3,640,640]`, **arcface_input** `[M,3,112,112]`, **yolo_output**, **arcface_output** `[M,512]`. **`CPUMemoryPool`**: **`init`**, **`release`**, **`upload_frame`**, **`register_system_shm`** / **`unregister_system_shm`**, **`buffers()`**. |
| `src/gpu/cpu_memory_pool.cpp` | Size computation (same numeric contracts as GPU pool), **`memcpy`** for frame upload, placeholder system-SHM registration (stub). |
| `src/gpu/cpu_preprocessor.h` | **`CPUPreprocessParams`** (letterbox **scale**, **pad_x**, **pad_y**, sizes). **`CPUPreprocessor`**: **`preprocess_yolo`**, **`preprocess_arcface`**, **`last_params()`**. |
| `src/gpu/cpu_preprocessor.cpp` | **`HAS_OPENCV`**: `cv::resize`, letterbox canvas, **`cv::cvtColor` BGR→RGB**, **`cv::dnn::blobFromImage`** (`scale=1/255`, NCHW). **No OpenCV**: nested loops with nearest-neighbor-style sampling + BGR→RGB + `/255` (YOLO); per-face bbox sampling + ArcFace **(x−127.5)/127.5** in RGB channel order. |
| `src/matching/cpu_face_matcher.h` | **`CPUFaceMatcher`**: **`init(FaceDatabase&, threshold)`**, **`release`**, **`match(embeddings, k)`**, **`add_face`**, **`database_size()`**. |
| `src/matching/cpu_face_matcher.cpp` | Loads **`FaceRecord`**s via **`FaceDatabase::load_all()`**; **brute-force** best cosine similarity per query embedding vs all DB rows; threshold gate; **-1** / **0.0** when empty or below threshold. |

**Parallel to GPU stack (see `src/gpu/memory_pool.h`, `preprocessor`, `FaceMatcher`):**

- **`GPUBuffers`** holds **`void*`** CUDA pointers; **`CPUBuffers`** holds **`std::vector`** owning CPU memory.
- **`GPUMemoryPool::register_triton_shm`** targets **CUDA** shared memory (device pointer); **`CPUMemoryPool::register_system_shm`** is reserved for **system** shared memory (CPU pointer) with Triton’s **SystemSharedMemory** APIs.
- **`Preprocessor`** (CUDA) vs **`CPUPreprocessor`** (OpenCV or manual loops).

**Pipeline wiring:** `src/pipeline/pipeline.cpp` currently constructs **`GPUMemoryPool`**, **`Preprocessor`**, and **`FaceMatcher`** only when **`use_gpu_`** is true after device detection. The CPU types in this phase are **implemented and build-linked** (`CMakeLists.txt` lists the `.cpp` files) and are ready to be selected when **`use_gpu_` is false** (same orchestration, swapped dependencies).

---

## Architecture Decisions

### 1. Interface abstraction: parallel concrete classes, not a single inheritance hierarchy (yet)

**Decision:** **`CPUMemoryPool`**, **`CPUPreprocessor`**, and **`CPUFaceMatcher`** are **standalone classes** with method names and buffer semantics aligned to **`GPUMemoryPool`**, **`Preprocessor`**, and **`FaceMatcher`**.

**Reasoning:**

- The spec calls for **runtime polymorphism** at startup (`cudaGetDeviceCount` / `--device`) so one pipeline loop can drive either backend. Achieving that in C++ typically means **abstract interfaces** (`IMemoryPool`, `IPreprocessor`, `IFaceMatcher`) or **`std::variant` / templates**. This repo stages **symmetric APIs** first so wiring `std::unique_ptr<Base>` (or a small façade) is a thin later step without rewriting math.
- **FaceMatcher** (`face_matcher.h`) and **CPUFaceMatcher** both consume **`std::vector<std::vector<float>>`** embeddings and emit **`MatchResult`**; signatures are intentionally similar for drop-in swapping.

### 2. Device selection at startup: `cudaGetDeviceCount` and `--device`

**Decision:** **`Pipeline::detect_device()`** (`src/pipeline/pipeline.cpp`) sets **`use_gpu_`** from:

- **`config_.device_mode == "cpu"`** → CPU path;
- **`"gpu"`** → fail init if no CUDA device;
- **auto** → GPU if **`cudaGetDeviceCount`** succeeds and count **> 0**, else CPU.

**Reasoning:**

- Matches the spec: “Backend selection happens **once at startup**.” No per-frame device hopping; Triton model/instance configuration (**KIND_GPU** vs **KIND_CPU**) stays consistent for the process lifetime.
- **CUDA OOM recovery** in **`Pipeline::init`**: first **`GPUMemoryPool::init`** failure retries with **`max_faces_per_frame / 2`**; second failure sets **`use_gpu_ = false`** and drops the GPU pool—the intended end state is **CPU fallback** (CPU pool + system SHM + CPU matcher should be attached in a full integration pass).

### 3. System shared memory vs CUDA shared memory

**Decision:**

| Mode | Memory | Pointer passed to Triton | Typical registration API |
|------|--------|--------------------------|----------------------------|
| GPU | CUDA device memory | GPU virtual address | CUDA shared memory region (`RegisterCudaSharedMemory` / IPC handle) |
| CPU | OS-backed shared memory or process heap | CPU virtual address | System shared memory (`SystemSharedMemoryRegister`, etc.) |

**Reasoning:**

- **CUDA shared memory** lets the Triton **TensorRT** backend read inputs **without copying tensors through gRPC**—only control metadata.
- **System shared memory** gives the same **zero-copy tensor path** for **ONNX Runtime / CPU** backends: tensors live in **pageable or pinned host** regions Triton maps into the inference process.
- **`CPUMemoryPool`** keeps tensors in **`std::vector`** today; production CPU mode would **`mmap`** / allocate **named** segments and register those base pointers—**`register_system_shm`** is currently a **stub** (`return true`), analogous to early GPU registration stubs in Phase 2.

### 4. OpenCV preprocessing equivalence to CUDA kernels

**Decision (YOLO path, `HAS_OPENCV`):**

1. **`cv::resize`** the source BGR to **`(new_w, new_h)`** with **`INTER_LINEAR`** (bilinear).
2. Copy into a **`dst_size × dst_size`** zero canvas at **`(pad_x, pad_y)`** (letterbox).
3. **`cv::cvtColor(..., COLOR_BGR2RGB)`**.
4. **`cv::dnn::blobFromImage`** with **`scale=1/255`**, **`swapRB=false`** (already RGB), producing **NCHW** `[1,3,H,W]` in **`CV_32F`**—**memcpy** into **`yolo_input`**.

**Reasoning:**

- This mirrors the GPU **letterbox + RGB + CHW + normalize/255** contract in the spec (Stages 3 and 9). **`last_params_`** stores the same **scale / pad** semantics as **`Preprocessor`** so NMS coordinate remap stays consistent.
- OpenCV’s bilinear **resize** is closer to the spec’s GPU bilinear table than the **non-OpenCV** branch’s integer nearest-neighbor in **`preprocess_yolo`** (acceptable fallback when OpenCV is disabled, not bit-identical to GPU).

### 5. Manual fallback without OpenCV

**Decision:** **YOLO:** double loop over **`dst_size`²**; for each pixel inside the padded content rect, map \((dx,dy)\) → source \((sx,sy)\) via **`pad` and `scale`**, then copy **BGR→CHW RGB** with **`/255`**. **ArcFace:** for each face, loop **112×112**, map to bbox with linear interpolation of indices, apply **(channel − 127.5) / 127.5** in **R,G,B** plane order.

**Reasoning:**

- Keeps **functional** CPU mode in minimal builds (`-UHAS_OPENCV`).
- Trade-off: YOLO branch uses **nearest** mapping (integer **`sx,sy`**) vs GPU/OpenCV bilinear—detection quality may differ slightly.

### 6. CPU matcher: brute-force cosine similarity

**Decision:** For each query embedding, loop all **`records_`**, compute **cos θ = (a·b) / (‖a‖‖b‖)**, take **argmax**, accept if **≥ threshold_**.

**Reasoning:**

- **FAISS-CPU** / **Flat inner product on L2-normalized vectors** is equivalent for **cosine** when vectors are unit norm; this implementation **normalizes in the denominator** explicitly, so it remains correct even if upstream embeddings are **not** pre-normalized.
- Complexity **O(N·D)** per face per frame; fine for **hundreds** of identities, consistent with “**~100–200ms**” **functional** CPU latency expectations, not real-time at 1080p30.

### 7. Performance expectations (spec)

**Decision:** Document **GPU target ~11–14ms** (spec **~13ms**) vs **CPU ~100–200ms**: no CUDA preprocess or **FAISS-GPU**; host-side resize and brute-force matching dominate; Triton may run **ONNX Runtime** on CPU.

**Reasoning:** Sets acceptance criteria: CPU path validates **correctness and operability**, not the **20ms** budget.

---

## Key Code Snippets

**Letterbox parameters (aligned with GPU preprocessor):**

```12:23:src/gpu/cpu_preprocessor.cpp
bool CPUPreprocessor::preprocess_yolo(const uint8_t* src_bgr, float* dst_chw,
                                       int src_width, int src_height, int dst_size) {
    float scale_x = static_cast<float>(dst_size) / src_width;
    float scale_y = static_cast<float>(dst_size) / src_height;
    float scale = std::min(scale_x, scale_y);

    int new_w = static_cast<int>(src_width * scale);
    int new_h = static_cast<int>(src_height * scale);
    int pad_x = (dst_size - new_w) / 2;
    int pad_y = (dst_size - new_h) / 2;

    last_params_ = {src_width, src_height, dst_size, scale, pad_x, pad_y};
```

**OpenCV blob path (NCHW, `/255`):**

```33:37:src/gpu/cpu_preprocessor.cpp
    cv::Mat rgb;
    cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);

    cv::Mat blob = cv::dnn::blobFromImage(rgb, 1.0 / 255.0, cv::Size(), cv::Scalar(), false, false);
    std::memcpy(dst_chw, blob.data, 3 * dst_size * dst_size * sizeof(float));
```

**CPU buffer sizing (mirrors GPU pool):**

```5:16:src/gpu/cpu_memory_pool.cpp
bool CPUMemoryPool::init(int frame_width, int frame_height, int max_faces) {
    size_t frame_size = static_cast<size_t>(frame_width) * frame_height * 3;
    size_t yolo_size = 1 * 3 * 640 * 640;
    size_t arcface_size = static_cast<size_t>(max_faces) * 3 * 112 * 112;
    size_t yolo_out_size = 8400 * 5;
    size_t arcface_out_size = static_cast<size_t>(max_faces) * 512;

    buffers_.raw_frame.resize(frame_size);
    buffers_.yolo_input.resize(yolo_size);
    buffers_.arcface_input.resize(arcface_size);
    buffers_.yolo_output.resize(yolo_out_size);
    buffers_.arcface_output.resize(arcface_out_size);
```

**Brute-force matcher:**

```33:52:src/matching/cpu_face_matcher.cpp
    for (size_t i = 0; i < embeddings.size(); ++i) {
        float best_score = -1.0f;
        int best_idx = -1;

        for (size_t j = 0; j < records_.size(); ++j) {
            float score = cosine_similarity(embeddings[i], records_[j].embedding);
            if (score > best_score) {
                best_score = score;
                best_idx = static_cast<int>(j);
            }
        }

        if (best_score >= threshold_ && best_idx >= 0) {
            results[i].face_id = records_[best_idx].id;
            results[i].name = records_[best_idx].name;
            results[i].confidence = best_score;
        } else {
            results[i].face_id = -1;
            results[i].confidence = 0.0f;
        }
    }
```

**Startup device gate (GPU vs CPU mode flag):**

```9:29:src/pipeline/pipeline.cpp
bool Pipeline::detect_device() {
    if (config_.device_mode == "cpu") {
        use_gpu_ = false;
        return true;
    }

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (config_.device_mode == "gpu") {
        if (err != cudaSuccess || device_count == 0) {
            fprintf(stderr, "Error: --device gpu specified but no CUDA device found\n");
            return false;
        }
        use_gpu_ = true;
        return true;
    }

    // auto-detect
    use_gpu_ = (err == cudaSuccess && device_count > 0);
    return true;
}
```

---

## Verification Approach

1. **Build matrix:** Compile **with** and **without** `-DHAS_OPENCV`; ensure **`cpu_preprocessor.cpp`** and **`cpu_memory_pool.cpp`** link in both configurations.
2. **Numerical smoke:** Feed a **synthetic** solid-color or gradient BGR frame into **`CPUPreprocessor::preprocess_yolo`** and compare **CHW** sums/min/max against **`Preprocessor`** on the same frame copied to GPU (allows tolerance for bilinear vs nearest in non-OpenCV builds).
3. **Matcher:** Construct **`FaceDatabase`** with a few rows, **`CPUFaceMatcher::init`**, **`match`** with a **duplicate** embedding and assert **face_id** / **confidence ≥ threshold**; use **orthogonal** vectors for **no match** (**-1**).
4. **System SHM:** After **`TritonClient`** gains real **RegisterSystemSharedMemory**, register **`CPUMemoryPool`** buffer bases and run a **single** YOLO infer; confirm **grpc** payload size stays small (metadata-only tensor references).
5. **End-to-end (when wired):** Run **`--device cpu`** on a short clip; confirm latency order-of-magnitude **≫20ms** and functional **detections + IDs** (manual timing with metrics CSV **`total_pipeline_ms`**).
