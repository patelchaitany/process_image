# Phase 4: Post-Processing (NMS & Face Crop) — Implementation Details

This document describes **Phase 4**: CPU-side non-maximum suppression and decoding of YOLO raw output, plus GPU-side face cropping and ArcFace-oriented preprocessing into a packed `[N, 3, 112, 112]` CHW tensor. It aligns with the design spec: [Face Detection & Recognition Pipeline Design](../superpowers/specs/2026-05-02-face-detection-recognition-pipeline-design.md).

---

## What Was Implemented

### Files, classes, and functions

| File | Role |
|------|------|
| `src/postprocess/nms.h` | **`Detection`** (xyxy + confidence); **`apply_nms`**, **`filter_and_decode`**. |
| `src/postprocess/nms.cpp` | Greedy NMS with IoU; confidence filter + **cxcywh → xyxy** with letterbox undo + clip to image. |
| `src/postprocess/face_cropper.h` | **`FaceCropper::crop_and_preprocess`** — uploads bboxes, launches ArcFace preprocessing kernels per face. |
| `src/postprocess/face_cropper.cu` | Host-side orchestration: H2D bbox upload, loop over faces calling **`fused_preprocess_arcface_kernel`**. |
| `src/gpu/preprocessor.cu` | Defines **`fused_preprocess_arcface_kernel`** (shared with **`Preprocessor::preprocess_arcface`**). |
| `tests/test_nms.cpp` | Tests for **`apply_nms`** (empty, single, overlap cases) and **`filter_and_decode`** (confidence filter). |

**Key APIs**

- **`std::vector<Detection> apply_nms(const std::vector<Detection>&, float iou_threshold = 0.45f)`** — classic greedy NMS after sorting indices by descending confidence.
- **`std::vector<Detection> filter_and_decode(...)`** — scans raw YOLO tensor `[num_detections, 5]` (cx, cy, w, h, conf), filters by threshold, maps boxes to original image space.
- **`bool FaceCropper::crop_and_preprocess(const void* src_frame_gpu, const std::vector<Detection>&, void* dst_batch_gpu, int src_width, int src_height, cudaStream_t stream)`** — writes one face per **slice** in **`dst_batch_gpu`**, CHW layout, 112×112.

---

## Architecture Decisions

### 1. NMS variant: greedy suppression after confidence sort

**Decision:** Sort detection **indices** by **`confidence` descending**, then iterate: keep a box, mark any later box with **IoU > threshold** as suppressed.

**Reasoning:**

- Matches the common **“standard” NMS** used in many detector pipelines and the timing assumptions in the spec (~tens of candidates, sub-millisecond).
- Sorting by score first ensures **highest-confidence** boxes survive overlapping clusters, which is what **`NMSTest::HigherConfidenceKept`** asserts.

### 2. NMS on CPU (not GPU)

**Decision:** Run **`apply_nms`** on the CPU over **`Detection`** vectors produced after **`filter_and_decode`**.

**Reasoning:**

- After **confidence filtering**, the number of candidates per frame is small (spec: on the order of tens, not thousands of anchors). A CPU loop over ~20–50 boxes is **cheaper than** a GPU kernel launch, synchronization, and D2H/H2D around NMS-shaped data structures.
- YOLO raw output is still copied to CPU for parse/NMS in the overall design; keeping NMS on CPU **avoids** implementing parallel NMS or sort on GPU for little gain at this scale.

### 3. Coordinate transform chain (model → letterbox space → original image)

**Decision:** Decode in **`filter_and_decode`** in one step:

1. Raw model outputs **center-x, center-y, width, height** in the **letterboxed / scaled** space (same space as the **640×640** model input).
2. Convert to **xyxy** in that space: \((cx \pm w/2,\ cy \pm h/2)\).
3. **Undo letterbox**: subtract **`pad_x`, `pad_y`** and divide by **`scale`** to map back to **original** **`orig_width × orig_height`** coordinates.
4. **Clip** each corner to **[0, orig_w]** / **[0, orig_h]** and drop degenerate boxes.

**Reasoning:**

- The preprocessor computes **`scale`, `pad_x`, `pad_y`** for letterboxing; NMS should operate in a **single consistent space**. Doing **undo + clip** here means **`Detection`** passed to **`FaceCropper`** is already in **full-frame BGR buffer coordinates**, matching **`src_frame_gpu`** geometry.
- Clipping avoids **out-of-bounds** reads in the crop kernel when faces touch image borders.

### 4. Face cropper: per-face CUDA dispatch and packed batch output

**Decision:**

- **`FaceCropper`** copies **`x1,y1,x2,y2`** for all faces to a temporary **`d_bboxes`** buffer, then launches **`fused_preprocess_arcface_kernel`** **once per face index** **`i`**, with **`dst`** pointing at the start of the full batch buffer.
- The kernel writes with **`face_offset = face_idx * 3 * plane_size`** so face **`i`** occupies **`[i*3*112*112, ...)`** in **CHW** form — i.e. a logical **`[N,3,112,112]`** contiguous layout.

**Reasoning:**

- **Per-face launch** reuses the same 2D grid (**112×112 threads per block grid**) as a single-face preprocess; **`face_idx`** selects the bbox and the **output plane offset**. This matches **`Preprocessor::preprocess_arcface`**, which uses the same kernel pattern.
- **Batch packing** in one **`dst_batch_gpu`** buffer allows a single Triton input tensor with shape **`[num_faces, 3, 112, 112]`** without an extra packing kernel.

**Note:** **`FaceCropper`** currently allocates **`d_bboxes`** with **`cudaMalloc`** per call and frees it before return; a future optimization could **reuse** a pooled device buffer to avoid allocator overhead on the hot path.

### 5. “Resize” and ArcFace normalization in the kernel

**Decision:** In **`fused_preprocess_arcface_kernel`**, each output pixel **`(dx, dy)`** maps into the face rectangle in **source image space** with a **linear mapping** in continuous coordinates, then **samples source with integer truncation** (nearest-neighbor on the grid). Channel values are transformed as **\((pixel - 127.5) / 127.5\)** in **BGR order** from the uint8 buffer. Output is **CHW** float.

**Reasoning:**

- **\((x - 127.5) / 127.5\)** is the standard **ArcFace input normalization** (maps [0,255] to approximately [-1, 1]), fused into the same kernel as crop/mapping to reduce memory traffic.
- The design spec discusses **bilinear** resize for quality; the **current** kernel uses **nearest-neighbor** sampling via **`static_cast<int>(...)`** on mapped indices. If quality regressions appear on small faces, **true bilinear** sampling in the kernel would be the natural upgrade.

### 6. Source frame layout

**Decision:** Assume **BGR uint8**, **`HWC`**, **`src_width * 3`** tightly packed rows (consistent with **`Frame`** from Phase 1 and **`fused_preprocess_yolo_kernel`**).

---

## Key Code Snippets

**Greedy NMS (sort + IoU suppression):**

```22:48:src/postprocess/nms.cpp
std::vector<Detection> apply_nms(const std::vector<Detection>& detections,
                                  float iou_threshold) {
    if (detections.empty()) return {};

    std::vector<int> indices(detections.size());
    for (size_t i = 0; i < indices.size(); ++i) indices[i] = static_cast<int>(i);

    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return detections[a].confidence > detections[b].confidence;
    });

    std::vector<bool> suppressed(detections.size(), false);
    std::vector<Detection> result;

    for (int idx : indices) {
        if (suppressed[idx]) continue;
        result.push_back(detections[idx]);

        for (int other : indices) {
            if (suppressed[other] || other == idx) continue;
            if (compute_iou(detections[idx], detections[other]) > iou_threshold) {
                suppressed[other] = true;
            }
        }
    }

    return result;
}
```

**Letterbox undo + original-image xyxy:**

```68:82:src/postprocess/nms.cpp
        // cxcywh in model coords -> xyxy in original image coords
        float x1 = (cx - w / 2.0f - pad_x) / scale;
        float y1 = (cy - h / 2.0f - pad_y) / scale;
        float x2 = (cx + w / 2.0f - pad_x) / scale;
        float y2 = (cy + h / 2.0f - pad_y) / scale;

        // Clip to image bounds
        x1 = std::max(0.0f, std::min(x1, static_cast<float>(orig_width)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(orig_height)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(orig_width)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(orig_height)));

        if (x2 > x1 && y2 > y1) {
            candidates.push_back({x1, y1, x2, y2, conf});
        }
```

**Per-face kernel launch and ArcFace offset packing:**

```42:48:src/postprocess/face_cropper.cu
    for (size_t i = 0; i < detections.size(); ++i) {
        fused_preprocess_arcface_kernel<<<grid, block, 0, stream>>>(
            static_cast<const uint8_t*>(src_frame_gpu),
            static_cast<float*>(dst_batch_gpu),
            d_bboxes, src_width, src_height,
            static_cast<int>(i), dst_size);
    }
```

```50:74:src/gpu/preprocessor.cu
    float x1 = bboxes[face_idx * 4 + 0];
    float y1 = bboxes[face_idx * 4 + 1];
    float x2 = bboxes[face_idx * 4 + 2];
    float y2 = bboxes[face_idx * 4 + 3];

    float face_w = x2 - x1;
    float face_h = y2 - y1;

    int sx = static_cast<int>(x1 + dx * face_w / dst_size);
    int sy = static_cast<int>(y1 + dy * face_h / dst_size);

    float r = 0.0f, g = 0.0f, b_val = 0.0f;
    if (sx >= 0 && sx < src_width && sy >= 0 && sy < src_height) {
        int src_idx = (sy * src_width + sx) * 3;
        b_val = (src[src_idx + 0] - 127.5f) / 127.5f;
        g     = (src[src_idx + 1] - 127.5f) / 127.5f;
        r     = (src[src_idx + 2] - 127.5f) / 127.5f;
    }

    int plane_size = dst_size * dst_size;
    int face_offset = face_idx * 3 * plane_size;
    int pixel_idx = dy * dst_size + dx;
    dst[face_offset + 0 * plane_size + pixel_idx] = r;
    dst[face_offset + 1 * plane_size + pixel_idx] = g;
    dst[face_offset + 2 * plane_size + pixel_idx] = b_val;
```

---

## Verification Approach

| Area | How it is verified |
|------|---------------------|
| **NMS semantics** | **`tests/test_nms.cpp`**: empty input, single box, non-overlapping set, full overlap (one survivor), partial overlap (expects 2), higher confidence wins with lower threshold. |
| **Filter/decode** | Same file: **`FilterDecodeTest`** checks confidence gate and empty output when all confidences are low. |
| **Integration** | End-to-end pipeline tests (when run) should show **`FaceCropper`** output feeding ArcFace with correct **NCHW** batch shape; GPU correctness can be spot-checked by dumping a face tensor and comparing to a reference **(x-127.5)/127.5** normalization on a cropped ROI. |

**Suggested manual checks**

- Run **`test_nms`** as part of the project test target (`ctest` / `gtest` via CMake).
- With a known synthetic **`raw_output`** and known **`scale` / pad**, assert **`filter_and_decode`** returns expected xyxy in original resolution.

---

## Summary

Phase 4 delivers **CPU NMS** suitable for **low candidate counts**, a clear **coordinate pipeline** from **YOLO cxcywh** through **letterbox parameters** to **full-resolution xyxy**, and **GPU batched face tensors** for ArcFace via **per-face kernel launches**, **CHW layout**, and **in-kernel ArcFace normalization**. The implementation shares **`fused_preprocess_arcface_kernel`** with **`Preprocessor::preprocess_arcface`** for one consistent preprocessing path.
