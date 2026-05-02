# Phase 5: ArcFace Recognition (Triton) — Implementation Details

This document describes **Phase 5**: the **`FaceRecognizer`** wrapper that runs the **arcface** model on Triton using **CUDA shared memory** handles from Phase 3 (`TritonClient`), with **variable batch size** per frame and **L2-normalized** 512-dimensional embeddings. It aligns with the design spec: [Face Detection & Recognition Pipeline Design](../superpowers/specs/2026-05-02-face-detection-recognition-pipeline-design.md).

---

## What Was Implemented

### Files, classes, and functions

| File | Role |
|------|------|
| `src/inference/face_recognizer.h` | **`FaceRecognizer`** — constructor, **`recognize`**, **`EMBEDDING_DIM`**. |
| `src/inference/face_recognizer.cpp` | **`recognize`** implementation: shape wiring, **`TritonClient::infer`**, row-major embedding copy, **L2 normalize** per face. |

**Public API**

```7:21:src/inference/face_recognizer.h
class FaceRecognizer {
public:
    explicit FaceRecognizer(TritonClient& client, const std::string& model_name = "arcface");

    bool recognize(const std::string& input_shm_name,
                   const std::string& output_shm_name,
                   int num_faces,
                   std::vector<std::vector<float>>& embeddings);

    static constexpr int EMBEDDING_DIM = 512;

private:
    TritonClient& client_;
    std::string model_name_;
};
```

---

## Architecture Decisions

### 1. Reuse `TritonClient` (Phase 3) instead of a second gRPC stack

**Decision:** **`FaceRecognizer`** holds a **reference** to **`TritonClient&`** and calls **`client_.infer(model_name_, input_shm_name, input_shape, output_shm_name, output_shape)`**.

**Reasoning:**

- One client manages **connection**, **shared memory registration naming**, and **infer** plumbing; recognition is just a **second model name** and **different tensor shapes**.
- Avoids duplicating retry logic, error handling, and metadata handling across **`FaceDetector`** and **`FaceRecognizer`**.

### 2. Dynamic batch size: `num_faces` at runtime

**Decision:** Each **`recognize`** call sets:

- **`input_shape = { num_faces, 3, 112, 112 }`**
- **`output_shape = { num_faces, EMBEDDING_DIM }`**

with **`num_faces`** determined **per frame** from detection + NMS.

**Reasoning:**

- ArcFace on Triton is configured for **dynamic batching** in the spec; the client should **not** assume a fixed batch (e.g. 8) if fewer faces exist — that would waste compute or require padding policies.
- Early return **`if (num_faces <= 0) return true`** avoids a zero-sized infer when there are no faces (callers still get an empty **`embeddings`** vector).

### 3. Output layout and `embeddings` allocation

**Decision:** On success, resize **`embeddings`** to **`num_faces`**, each inner vector to **`EMBEDDING_DIM`**, and fill from **`result.output_data`** as **contiguous rows**: face **`i`** starts at **`data + i * EMBEDDING_DIM`**.

**Reasoning:**

- Matches Triton’s typical **row-major** batch layout **`[N, 512]`**.
- **`std::vector<std::vector<float>>`** matches downstream **`FaceMatcher::match`** without copying into a flat buffer at the pipeline boundary (trade-off: more allocations; clarity and symmetry with matcher API).

### 4. L2 normalization after inference

**Decision:** For each face, compute **\( \|e\|_2 \)** over the 512 floats; divide each component by the norm (or zero if norm is zero).

**Reasoning:**

- **Cosine similarity** between two vectors equals the **inner product** when both are **unit-length**: \(\cos(\theta) = e_1 \cdot e_2\) if \(\|e_1\|=\|e_2\|=1\).
- The design spec’s **FAISS** path uses **inner product** on normalized vectors as **cosine similarity** matching. Normalizing here keeps **FaceRecognizer** outputs consistent even if the served model’s last layer is not strictly unit-norm on every engine/version.
- **`FaceMatcher`** / enrollment tools should also store **normalized** embeddings so **database vectors and query vectors** live in the same geometry.

### 5. `EMBEDDING_DIM = 512`

**Decision:** Single **`static constexpr int EMBEDDING_DIM = 512`** used for output shape and copy loops.

**Reasoning:**

- Matches the **ArcFace ResNet-100** head described in the spec (**512-dim** embeddings).
- Centralizes the dimension so a mistaken magic number is less likely at call sites.

---

## Key Code Snippets

**Infer shapes and dynamic `N`:**

```11:18:src/inference/face_recognizer.cpp
    if (num_faces <= 0) return true;

    std::vector<int64_t> input_shape = {num_faces, 3, 112, 112};
    std::vector<int64_t> output_shape = {num_faces, EMBEDDING_DIM};

    InferResult result = client_.infer(model_name_, input_shm_name,
                                        input_shape, output_shm_name, output_shape);
```

**Per-row L2 normalization:**

```24:38:src/inference/face_recognizer.cpp
    for (int i = 0; i < num_faces; ++i) {
        embeddings[i].resize(EMBEDDING_DIM);
        const float* emb = data + i * EMBEDDING_DIM;

        // L2 normalize
        float norm = 0.0f;
        for (int j = 0; j < EMBEDDING_DIM; ++j) {
            norm += emb[j] * emb[j];
        }
        norm = std::sqrt(norm);

        for (int j = 0; j < EMBEDDING_DIM; ++j) {
            embeddings[i][j] = (norm > 0) ? emb[j] / norm : 0.0f;
        }
    }
```

---

## Verification Approach

| Check | Method |
|-------|--------|
| **Shape contract** | With **`num_faces = 1`**, confirm Triton receives **`[1,3,112,112]`** and returns **`[1,512]`** (server logs or client assertions on **`InferResult`**). |
| **Unit norm** | After **`recognize`**, for each embedding assert **`abs(sum of squares - 1.0f) < epsilon`** (unless zero vector fallback). |
| **Integration** | Run pipeline on a frame with known face: embedding should be **stable** across runs with same TRT model (allow small FP noise). |
| **Zero faces** | Call **`recognize(..., 0, embeddings)`** — expect **`true`** and **`embeddings.empty()`**. |

**Automated tests:** Add a gtest that mocks **`TritonClient::infer`** to return a fixed non-normalized vector and asserts the post-processed embedding has unit L2 norm.

---

## Summary

**`FaceRecognizer`** is a thin, **batch-aware** wrapper: it binds **variable `num_faces`** to Triton **dynamic batch** shapes, reads **`[N,512]`** output from shared memory via **`TritonClient`**, and guarantees **L2-normalized** rows for **cosine / inner-product** matching downstream. **`EMBEDDING_DIM`** is fixed at **512** per the ArcFace design.
