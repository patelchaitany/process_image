# Phase 2: GPU Memory Pool & Preprocessing — Implementation Details

This document describes **Phase 2**: a **`GPUMemoryPool`** that allocates reusable CUDA buffers (including placeholders for Triton CUDA shared memory registration) and a **`Preprocessor`** that runs **fused CUDA kernels** for YOLO-style and ArcFace-style tensor preparation. It aligns with the design spec: [Face Detection & Recognition Pipeline Design](../superpowers/specs/2026-05-02-face-detection-recognition-pipeline-design.md).

---

## What Was Implemented

### Files, classes, and functions

| File | Role |
|------|------|
| `src/gpu/memory_pool.h` | **`GPUBuffers`** struct (raw frame, YOLO input/output, ArcFace input/output pointers); **`GPUMemoryPool`** with **`init`**, **`release`**, **`upload_frame`**, **`register_triton_shm`** / **`unregister_triton_shm`** (stubs), **`stream()`** accessor. |
| `src/gpu/memory_pool.cu` | Allocations, **`cudaMemcpyAsync`** H2D for raw frame, destructor/release, placeholder Triton registration. |
| `src/gpu/preprocessor.h` | **`PreprocessParams`** (source size, square **`dst_size`**, letterbox **`scale`**, **`pad_x`**, **`pad_y`**); **`Preprocessor`** with **`preprocess_yolo`**, **`preprocess_arcface`**, **`last_params()`**. |
| `src/gpu/preprocessor.cu` | **`fused_preprocess_yolo_kernel`**, **`fused_preprocess_arcface_kernel`**, host wrappers with grid/block sizing and **`cudaMemsetAsync`** for letterbox padding. |

**Kernel responsibilities:**

- **YOLO path:** One thread per destination pixel in **`dst_size × dst_size`**. Maps \((dx, dy)\) back to source via letterbox inverse scale and padding; reads **BGR** uint8 from GPU source; writes **CHW** float32 RGB planes normalized by **255.0**. Padded regions remain **0** (after memset).
- **ArcFace path:** For each face index, maps **112×112** destination pixels into the **axis-aligned bbox** \([x1,y1,x2,y2]\) on the full-resolution BGR frame; applies **ArcFace** normalization **\((x - 127.5) / 127.5\)**; writes **CHW** per face into a batched output region.

---

## Architecture Decisions

### 1. Buffer sizing: one allocation at startup, reuse every frame

**Decision:** **`GPUMemoryPool::init(frame_width, frame_height, max_faces)`** computes sizes once:

- **`raw_frame`**: `frame_width * frame_height * 3` bytes (uint8)
- **`yolo_input`**: `1 × 3 × 640 × 640 × sizeof(float)`
- **`arcface_input`**: `max_faces × 3 × 112 × 112 × sizeof(float)`
- **`yolo_output`**: `8400 × 5 × sizeof(float)` (fixed geometry matching the spec’s YOLO head)
- **`arcface_output`**: `max_faces × 512 × sizeof(float)`

**Reasoning:**

- Avoids **per-frame `cudaMalloc`** (driver churn and fragmentation).
- Sizes match the pipeline contract in the design doc (1080p upload, 640 YOLO square input, up to **`max_faces`** ArcFace tiles).
- Failure path **`goto fail`** releases partial allocations and reports **`cudaGetErrorString`**.

### 2. Kernel fusion: single read + single write per YOLO destination pixel

**Decision:** **`fused_preprocess_yolo_kernel`** performs **inverse letterbox mapping**, **BGR→RGB channel order for model input**, **uint8→float**, **÷255**, and **HWC→CHW** in one kernel.

**Reasoning:**

- The spec notes that separate passes (resize, pad, swap, transpose, normalize) add multiple global memory traversals; **fusing** targets **one load** of source BGR (when the mapped coordinate hits the valid letterboxed region) and **one structured write** to the three float planes.
- **Trade-off:** The current implementation uses **nearest-neighbor** sampling (integer **`sx`**, **`sy`** from the inverse map). The design doc’s performance table mentions bilinear resize; upgrading to **true bilinear** would add **4** source reads and fractional weights per channel while keeping a single fused kernel structure.

### 3. Letterbox vs stretch: aspect ratio preserved (YOLO)

**Decision:** **`preprocess_yolo`** computes:

```cpp
scale = min(dst_size / src_width, dst_size / src_height);
new_w = floor(src_width * scale), new_h = floor(src_height * scale);
pad_x = (dst_size - new_w) / 2, pad_y = (dst_size - new_h) / 2;
```

Output is zeroed first (**`cudaMemsetAsync`**) so **padding** stays at **0**. Only pixels inside the padded rectangle sample the source.

**Reasoning:**

- **Stretching** to **`640×640`** distorts faces and hurts detection; **letterboxing** matches common YOLO deployments and the spec’s “letterbox padding” line item.
- **`PreprocessParams last_params_`** exposes **`scale`**, **`pad_x`**, **`pad_y`** so a later **CPU NMS** stage can map boxes from **`640×640`** back to full-frame coordinates (spec Stage 8).

### 4. Triton CUDA shared memory: registration hooks (Phase 2 stub)

**Decision:** **`register_triton_shm` / `unregister_triton_shm`** in **`memory_pool.cu`** are **stubs** (`return true` / no-op), with comments indicating **Phase 3** integration via **`TritonClient`**.

**Reasoning:**

- Confirms the **intended ownership**: pool knows **sizes and pointers**; the **client** knows **server URL and registration API**.
- **`init`** stays usable for preprocess-only tests without a running Triton.

### 5. CUDA stream: async H2D and kernel launches

**Decision:** **`GPUMemoryPool`** creates **`cudaStream_t stream_`** in **`init`** and uses it in **`upload_frame`** when no override stream is passed. **`Preprocessor`** methods take **`cudaStream_t stream`** and pass it to **`cudaMemsetAsync`**, **`<<<>>>`** launches, and rely on callers to synchronize when needed.

**Reasoning:**

- Matches the spec’s **async** pipeline: overlap **H2D** with CPU work where possible, and align with **monitoring** via CUDA events on the same stream in later phases.

### 6. Sampling in the kernels (“interpolation”)

**YOLO:** For each **(dx, dy)** in the **640×640** grid, the kernel computes source integer indices **(sx, sy)** from the **inverse letterbox** transform and reads **one** BGR triplet when in bounds — **nearest-neighbor** on the source grid.

**ArcFace:** For **112×112**, **sx** / **sy** are **linearly interpolated** along bbox width/height with **`static_cast<int>(...)`** — effectively **nearest** sampling on a per-dimension grid (not full **bilinear** with four neighbors).

**Reasoning:**

- Keeps kernels **simple** and **predictable** for a first real-time implementation; quality can be refined if mAP or embedding stability requires it.

---

## Reasoning Summary

| Topic | Choice | Why |
|--------|--------|-----|
| Allocations | Upfront pool | Stable latency, no malloc in hot loop |
| YOLO preprocess | Fused kernel | Fewer memory passes |
| Geometry | Letterbox | Preserves aspect ratio for YOLO |
| Padding | `cudaMemsetAsync` | Gray/pad-free zeros; model sees uniform border |
| Triton SHM | Stub in pool | Wiring completed with `TritonClient` (Phase 3) |
| Streams | Default pool stream + param | Async memcpy and kernels |

---

## Key Code Snippets

**Buffer layout:**

```7:38:src/gpu/memory_pool.h
struct GPUBuffers {
    void* raw_frame = nullptr;       // 1920*1080*3 uint8
    void* yolo_input = nullptr;      // 1*3*640*640 float32
    void* arcface_input = nullptr;   // M*3*112*112 float32
    void* yolo_output = nullptr;     // output tensor
    void* arcface_output = nullptr;  // M*512 float32
};

class GPUMemoryPool {
public:
    GPUMemoryPool() = default;
    ~GPUMemoryPool();

    bool init(int frame_width, int frame_height, int max_faces);
    void release();

    bool upload_frame(const uint8_t* cpu_data, size_t size, cudaStream_t stream = nullptr);
    // ...
    bool register_triton_shm(const std::string& triton_url);
    void unregister_triton_shm();
```

**Allocation sizes and reuse:**

```17:38:src/gpu/memory_pool.cu
    raw_frame_size_ = static_cast<size_t>(frame_width) * frame_height * 3;
    size_t yolo_size = 1 * 3 * 640 * 640 * sizeof(float);
    size_t arcface_size = static_cast<size_t>(max_faces) * 3 * 112 * 112 * sizeof(float);
    size_t yolo_out_size = 8400 * 5 * sizeof(float);
    size_t arcface_out_size = static_cast<size_t>(max_faces) * 512 * sizeof(float);

    err = cudaMalloc(&buffers_.raw_frame, raw_frame_size_);
    // ... further cudaMalloc calls ...
    initialized_ = true;
```

**Letterbox + fused YOLO launch:**

```77:104:src/gpu/preprocessor.cu
bool Preprocessor::preprocess_yolo(const void* src_bgr_gpu, void* dst_chw_gpu,
                                    int src_width, int src_height, int dst_size,
                                    cudaStream_t stream) {
    float scale_x = static_cast<float>(dst_size) / src_width;
    float scale_y = static_cast<float>(dst_size) / src_height;
    float scale = std::min(scale_x, scale_y);

    int new_w = static_cast<int>(src_width * scale);
    int new_h = static_cast<int>(src_height * scale);
    int pad_x = (dst_size - new_w) / 2;
    int pad_y = (dst_size - new_h) / 2;

    last_params_ = {src_width, src_height, dst_size, scale, pad_x, pad_y};

    // Zero output first (for letterbox padding)
    cudaMemsetAsync(dst_chw_gpu, 0, 3 * dst_size * dst_size * sizeof(float), stream);

    dim3 block(16, 16);
    dim3 grid((dst_size + block.x - 1) / block.x,
              (dst_size + block.y - 1) / block.y);

    fused_preprocess_yolo_kernel<<<grid, block, 0, stream>>>(
        static_cast<const uint8_t*>(src_bgr_gpu),
        static_cast<float*>(dst_chw_gpu),
        src_width, src_height, dst_size, scale, pad_x, pad_y);

    return cudaGetLastError() == cudaSuccess;
}
```

**Core per-pixel YOLO mapping (nearest-neighbor source sample):**

```6:36:src/gpu/preprocessor.cu
__global__ void fused_preprocess_yolo_kernel(
    const uint8_t* __restrict__ src,
    float* __restrict__ dst,
    int src_width, int src_height,
    int dst_size, float scale, int pad_x, int pad_y)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx >= dst_size || dy >= dst_size) return;

    float r = 0.0f, g = 0.0f, b = 0.0f;

    int sx = static_cast<int>((dx - pad_x) / scale);
    int sy = static_cast<int>((dy - pad_y) / scale);

    if (sx >= 0 && sx < src_width && sy >= 0 && sy < src_height &&
        dx >= pad_x && dy >= pad_y) {
        int src_idx = (sy * src_width + sx) * 3;
        b = src[src_idx + 0] / 255.0f;
        g = src[src_idx + 1] / 255.0f;
        r = src[src_idx + 2] / 255.0f;
    }

    // CHW layout: R plane, G plane, B plane
    int plane_size = dst_size * dst_size;
    int pixel_idx = dy * dst_size + dx;
    dst[0 * plane_size + pixel_idx] = r;
    dst[1 * plane_size + pixel_idx] = g;
    dst[2 * plane_size + pixel_idx] = b;
}
```

---

## Verification Approach

1. **Build:** Ensure the CUDA target compiles **`memory_pool.cu`** and **`preprocessor.cu`** with the rest of the project (Phase 0 CMake already enables CUDA).
2. **Device-side checks (recommended additions):** Unit tests can fill a **`Frame`** on CPU, **`upload_frame`**, run **`preprocess_yolo`**, **`cudaMemcpy`** result back, and assert:
   - Corner padding pixels are **0** on all three planes.
   - Center pixels scale monotonically with a synthetic gradient image.
   - **`last_params()`** matches expected **`pad_*`** for a known resolution (e.g. 1920×1080 → 640).
3. **`cuda-memcheck`** / **`compute-sanitizer`**: Run a short harness to catch OOB access if bbox or letterbox math changes.
4. **Triton E2E (Phase 3+):** After shared memory registration is live, confirm **`compute_input_time_ns`** stays minimal — evidence that the pool’s **`yolo_input`** buffer is the tensor Triton reads without a large PCIe input copy per request.
