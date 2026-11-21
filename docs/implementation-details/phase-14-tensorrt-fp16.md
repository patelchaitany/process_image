# Phase A: TensorRT FP16 via Triton (implementation plan)

This phase turns on **ONNX Runtime + TensorRT FP16** inside Triton for `yolo26_face` and `arcface`. The C++ pipeline, shared-memory buffers, and ONNX weights stay the same. Only **how Triton executes** the graphs changes.

**Baseline (before Phase A):** `platform: "onnxruntime_onnx"` with no `optimization` block — generic GPU ONNX Runtime (FP32 path). Crowded `input_4`: ArcFace median ~62 ms at ~25 faces, ~67 ms at ~28 faces.

**Target:** Lower `arcface_inference_ms` and more stable `yolo_inference_ms` after warmup, with embeddings still usable for FAISS match (smoke-test enrollments after deploy).

---

## Scope

| In scope | Out of scope (Phase B+) |
|----------|-------------------------|
| `config.pbtxt` for both models | Latest-frame live ingest |
| `download_models.py` templates | Face tracking / embed-on-demand |
| TRT engine cache on disk | Smaller model (`w600k_mbf`) |
| RUNBOOK + warmup tooling | C++ pipeline changes |
| Verification steps | INT8 quantization |

---

## Files changed

| File | Action |
|------|--------|
| `models/model_repository/arcface/config.pbtxt` | Add TRT FP16 + cache; extend `preferred_batch_size` to include 16, 32 |
| `models/model_repository/yolo26_face/config.pbtxt` | Add TRT FP16 + cache |
| `models/model_repository/arcface/trt_cache/.gitkeep` | Cache dir (engines gitignored) |
| `models/model_repository/yolo26_face/trt_cache/.gitkeep` | Cache dir |
| `tools/download_models.py` | GPU configs include same TRT block; create `trt_cache` dirs on `--gpu` |
| `tools/warmup_arcface.py` | Docstring: TRT + ORT warmup |
| `tools/warmup_yolo.py` | **New** — warm batch 1, 4, 8 |
| `tools/warmup_triton.py` | **New** — runs ArcFace + YOLO warmup |
| `RUNBOOK.md` | TRT startup timing, cache, warmup steps |
| `docs/implementation-details/phase-14-tensorrt-fp16.md` | This document |

**Not changed:** `src/**`, `enroll_faces.py`, `model.onnx` binaries.

---

## Triton config (what we add)

Both GPU models get:

```text
optimization {
  execution_accelerators {
    gpu_execution_accelerator: [{
      name: "tensorrt"
      parameters { key: "precision_mode" value: "FP16" }
      parameters { key: "max_workspace_size_bytes" value: "1073741824" }
      parameters { key: "trt_engine_cache_enable" value: "1" }
      parameters { key: "trt_engine_cache_path" value: "/models/<model_name>/trt_cache" }
    }]
  }
}
```

**Why each knob**

- `tensorrt` — ORT delegates supported ops to TensorRT (fusion, kernel selection).
- `FP16` — T4 Tensor Cores; main latency win vs FP32 ORT.
- `max_workspace_size_bytes` — 1 GiB workspace for TRT builder (safe on 16 GB T4 with ~3 GB app usage).
- `trt_engine_cache_*` — Persist engines under the mounted model repo so **container restart** reloads cache instead of rebuilding from scratch.

**Input/output types stay `TYPE_FP32`** — client and C++ still send FP32; TRT uses FP16 internally.

**Docker path:** `-v $(pwd)/models/model_repository:/models` → cache path `/models/arcface/trt_cache` is correct inside the container.

---

## Operational sequence (after code merge)

1. **Stop** existing Triton container (if any).
2. **Start** Triton (step 4 in RUNBOOK). First load with TRT may take **longer than before** until engines exist or cache is populated.
3. **Wait** for `ready` HTTP 200.
4. **Warm** models:
   ```bash
   python3 tools/warmup_triton.py --triton localhost:8001
   ```
5. **Smoke test** recognition (optional but recommended):
   ```bash
   python3 tools/enroll_faces.py --faces-dir ./faces/ --db ./faces.db --triton localhost:8001
   ```
6. **Benchmark** metrics:
   ```bash
   ./build/process_image --input ./input/input_4.mp4 --triton localhost:8001 --db ./faces.db --device auto
   ```
   Compare new `metrics/*.csv` to `metrics_input_4_2026-10-11_113921.csv` (`arcface_inference_ms`, `yolo_inference_ms`, `total_pipeline_ms`).

---

## What is *not* slow per video file

- Switching `input_3.mp4` → `input_4.mp4` with **Triton left running** does **not** recompile TRT.
- Slow path is **Triton (re)start**, **config change**, or **first inference** for a batch size before warmup/cache.

### First cold TensorRT build (measured on this studio, Oct 2025)

One-time engine builds on a cold cache are **much** slower than steady-state inference:

| Step | Observed wall time |
|------|-------------------|
| YOLO warmup batch 1 | ~402 s |
| YOLO warmup batches 4, 8 | ~183 s each |
| ArcFace warmup batch 1 | ~112 s (in progress for 2..32) |

After `trt_cache/` is populated, **restart Triton + warmup** should mostly **load** engines (seconds per batch, not minutes). Run `warmup_triton.py` once after each container start; do not cancel mid-run on first deploy.

---

## Engine bundle in git

| Artifact | Purpose |
|----------|---------|
| `assets/trt_engine_cache_t4_sm75_fp16.zip` | Committed engines (~82 MB) for T4 / sm75 |
| `tools/restore_trt_cache.py` | Unpack zip → `models/model_repository/*/trt_cache/` |
| `tools/pack_trt_cache.py` | Rebuild zip after a new warmup (commit updated zip) |

Live `.engine` files under `models/` stay gitignored; only the zip in `assets/` is tracked.

```bash
python3 tools/restore_trt_cache.py   # after download_models.py --real --gpu
```

`config.pbtxt` TRT settings are reproduced by `python3 tools/download_models.py --real --gpu`.

---

## Rollback

Remove the `optimization { ... }` block from both `config.pbtxt` files (or run `download_models.py` with templates reverted), restart Triton. No C++ rebuild required.

---

## Success criteria

- `/v2/models/arcface/config` JSON includes `execution_accelerators` with `tensorrt` and `FP16`.
- `trt_cache/` contains engine files after first successful warm (filenames vary by TRT version).
- `input_4`: material drop in `arcface_inference_ms` p50/p95 vs baseline CSV (exact % measured on studio).
- Enrollment still matches known faces on a short clip (manual check).

---

## Implementation status

- [x] Plan document (this file)
- [x] `config.pbtxt` + `download_models.py` + warmup tools + RUNBOOK
- [x] Triton restarted; config exposes TensorRT FP16 + cache paths
- [x] `warmup_triton.py` finished (cached re-warm ~1–2 s for all batches)
- [ ] Benchmark CSV vs `metrics_input_4_2026-10-11_113921.csv` after warmup completes
