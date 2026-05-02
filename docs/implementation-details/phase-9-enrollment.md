# Phase 9: Face enrollment tool (`enroll_faces.py`)

This document describes the Python enrollment CLI that fills the SQLite `faces` table from a directory of per-person photo folders, using the same YOLO and ArcFace models served by Triton as the C++ pipeline.

## What was implemented

### Files

| File | Role |
|------|------|
| `tools/enroll_faces.py` | Full enrollment workflow: Triton gRPC inferencing, detection, NMS, embedding extraction, averaging, SQLite inserts, summary output |
| `tools/requirements.txt` | `tritonclient[grpc]`, `numpy`, `opencv-python` |

### Key functions

- `parse_args()` — `argparse` with `--faces-dir`, `--db`, `--triton`, model names, `--confidence`, `--replace`.
- `connect_triton(url)` — constructs `grpcclient.InferenceServerClient`, checks `is_server_live()`.
- `preprocess_yolo` / `preprocess_arcface` — letterbox resize and normalization consistent with training/serving expectations (BGR→RGB, CHW, ArcFace `(x-127.5)/127.5`).
- `run_yolo` / `run_arcface` — wrap `InferInput`, `InferRequestedOutput`, `client.infer`, `as_numpy`.
- `decode_yolo_output` — threshold, map letterbox coords back to original image, clip boxes.
- `nms` / `compute_iou` — greedy non-maximum suppression.
- `init_database` — `CREATE TABLE` / index; optional `DROP TABLE IF EXISTS faces` when replacing.
- `embedding_to_blob` — `float32` numpy to bytes for SQLite `BLOB`.
- `main` — iterate person subfolders, images, aggregate embeddings per person, `INSERT`, print per-spec summaries.

## Architecture decisions

### Python Triton client (`tritonclient.grpc`)

The script uses **`grpcclient.InferenceServerClient`** for low-latency, binary-friendly inference against the running server.

- **YOLO:** `InferInput("images", shape, "FP32")` + `set_data_from_numpy` on the preprocessed blob; requested output `"output0"`.
- **ArcFace:** `InferInput("input", shape, "FP32")`; requested output `"output"`.

**Reasoning:** Matches typical Triton naming for these models, keeps enrollment aligned with production inference (same server, same tensor layouts as the C++ client’s model config). No local TensorRT or ONNX runtime dependency in Python.

### NMS reimplemented to mirror C++ greedy NMS

Python `nms`:

1. Sort detections by confidence **descending**.
2. Repeatedly take the **best** remaining box, push to `keep`, drop any other box with `IoU(best, other) > iou_threshold` (default **0.45**, same as `PipelineConfig::nms_iou_threshold`).

C++ `apply_nms` sorts indices by confidence, then for each kept box marks overlapping higher-IoU candidates suppressed—**equivalent greedy behavior** with the same IoU definition (intersection over union of axis-aligned boxes).

**Reasoning:** Enrollment must produce crops comparable to runtime detection so averaged embeddings match what the live pipeline sees.

### Embedding averaging and L2 normalization

Per valid image:

1. Run ArcFace; squeeze batch dimension if needed.
2. **L2-normalize** the single-face embedding: `embedding / np.linalg.norm(embedding)` if `norm > 0`.

After collecting all valid embeddings for a person:

1. **`avg_embedding = np.mean(person_embeddings, axis=0)`**
2. **Renormalize:** `avg_embedding /= np.linalg.norm(avg_embedding)` so the stored vector is unit length for cosine-style matching with the C++/FAISS path.

**Reasoning:** ArcFace outputs are meaningful on the unit hypersphere; averaging in embedding space then renormalizing yields one representative template per identity, reducing variance across pose/lighting while preserving comparability with normalized query embeddings at runtime.

### Edge cases

| Case | Behavior |
|------|----------|
| **0 faces** after NMS | Log `0 faces detected, skipping`, increment skipped count, no embedding added. |
| **Multiple faces** | Log `N faces detected, skipping (ambiguous)`, skip image—no heuristic to pick which face is the enrolled identity. |
| **Unreadable image** (`cv2.imread` returns `None`) | `cannot read, skipping`. |
| **Empty crop** (degenerate box) | `invalid crop, skipping`. |
| **No valid embeddings for a person** | Print `→ {person}: no valid embeddings, NOT enrolled` — **no** `INSERT`. |

**Reasoning:** The design spec explicitly requires skipping ambiguous multi-face images and zero-face images to avoid polluting the database with wrong or empty templates.

### `--replace` semantics

When `--replace` is set, `init_database` runs:

```sql
DROP TABLE IF EXISTS faces;
```

before:

```sql
CREATE TABLE IF NOT EXISTS faces ( ... );
CREATE INDEX IF NOT EXISTS idx_faces_name ON faces(name);
```

**Reasoning:** A full rebuild guarantees the SQLite file matches the new enrollment run without stale rows or ID skew relative to a separately rebuilt FAISS index in the C++ app. (The C++ loader should reopen the DB after enrollment to refresh counts.)

### Folder structure convention

Under the root `--faces-dir`, **each immediate subdirectory name is the person’s `name` column** in SQLite. Only image files with extensions `{.jpg, .jpeg, .png, .bmp, .tiff, .tif, .webp}` are considered; non-directories at the root are ignored.

**Reasoning:** Matches the design spec’s `faces/alice/`, `faces/bob/` layout and supports any OpenCV-readable format in that list.

### Output format (spec alignment)

The script prints:

1. `Connecting to Triton at {url}... OK` (or FAILED + exit).
2. For each person: `Processing {name}/ ({k} images)...`
3. Per image: lines such as `1 face detected, embedding extracted`, `0 faces detected, skipping`, `N faces detected, skipping (ambiguous)`, `cannot read, skipping`.
4. Per person closeout: `→ {name} enrolled (averaged M embeddings)` or `→ {name}: no valid embeddings, NOT enrolled`.
5. Final: `Summary: {enrolled}/{total_persons} persons enrolled, {used}/{total_images} images used, {skipped} skipped` and `Database written to {db}`.

The Unicode arrow `→` in the script matches the spec’s example output character.

**Reasoning:** Operators can grep logs for failures and reconcile counts with filesystem layout without opening SQLite.

## Key code snippets

**Triton infer wrappers:**

```78:97:tools/enroll_faces.py
def run_yolo(client, model_name, input_blob):
    """Run YOLO inference on Triton, return raw output."""
    inputs = [grpcclient.InferInput("images", input_blob.shape, "FP32")]
    inputs[0].set_data_from_numpy(input_blob)

    outputs = [grpcclient.InferRequestedOutput("output0")]

    result = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    return result.as_numpy("output0")


def run_arcface(client, model_name, input_blob):
    """Run ArcFace inference on Triton, return embedding."""
    inputs = [grpcclient.InferInput("input", input_blob.shape, "FP32")]
    inputs[0].set_data_from_numpy(input_blob)

    outputs = [grpcclient.InferRequestedOutput("output")]

    result = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    return result.as_numpy("output")
```

**Greedy NMS:**

```132:150:tools/enroll_faces.py
def nms(detections, iou_threshold=0.45):
    """Simple greedy NMS."""
    if not detections:
        return []

    dets = sorted(detections, key=lambda d: d[4], reverse=True)
    keep = []

    while dets:
        best = dets.pop(0)
        keep.append(best)
        remaining = []
        for d in dets:
            iou = compute_iou(best, d)
            if iou <= iou_threshold:
                remaining.append(d)
        dets = remaining

    return keep
```

**Averaging, renormalization, insert, summary:**

```286:305:tools/enroll_faces.py
        # Average embeddings for this person
        if person_embeddings:
            avg_embedding = np.mean(person_embeddings, axis=0)
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

            cursor.execute(
                "INSERT INTO faces (name, embedding) VALUES (?, ?)",
                (person_name, embedding_to_blob(avg_embedding))
            )
            conn.commit()
            enrolled_count += 1
            print(f"  \u2192 {person_name} enrolled (averaged {len(person_embeddings)} embeddings)")
        else:
            print(f"  \u2192 {person_name}: no valid embeddings, NOT enrolled")

    conn.close()

    print(f"\nSummary: {enrolled_count}/{total_persons} persons enrolled, "
          f"{used_images}/{total_images} images used, {skipped_images} skipped")
    print(f"Database written to {args.db}")
```

## Verification approach

1. **Environment:** `pip install -r tools/requirements.txt` in a virtualenv; start Triton with the YOLO and ArcFace models used by the C++ client.
2. **Smoke:** Create `faces/test_person/one.jpg` with a single clear face; run `python tools/enroll_faces.py --faces-dir ./faces --db ./test.db --triton localhost:8001`; expect one row in `faces`, unit-norm `embedding` blob length `512 * 4` bytes.
3. **Edge cases:** Add an empty image directory, a multi-face crowd photo, and a corrupt file; confirm logs match the skip messages and `Summary` counts match expectations (`skipped` increments).
4. **Replace:** Enroll once, run again with `--replace`; verify only the second run’s identities remain (`SELECT COUNT(*) FROM faces`).
5. **C++ parity:** Open the resulting `faces.db` with the pipeline’s `--db`; confirm recognized names match folder names for well-lit single-face enrollments.
