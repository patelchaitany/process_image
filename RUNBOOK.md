# Run steps (this studio)

## 1. Run setup.sh (apt packages + Triton C++ Client SDK + Python basics)

```bash
cd ~/process_image
chmod +x setup.sh
sudo ./setup.sh
```

## 2. Install libs and dependencies

```bash
conda install -y -c pytorch -c nvidia faiss-gpu

pip install onnx onnxruntime numpy opencv-python insightface ultralytics
```

## 3. Download real pre-trained ONNX models

```bash
python3 tools/download_models.py --real --gpu
```

## 4. Restore TensorRT engine cache (recommended)

Skips the **multi-hour** cold TensorRT build on a **Tesla T4 (sm75)** with Triton
`24.01` and the FP16 `config.pbtxt` in this repo. The bundle is versioned in git:

`assets/trt_engine_cache_t4_sm75_fp16.zip`

```bash
cd ~/process_image
python3 tools/restore_trt_cache.py
```

Dry-run (list files only):

```bash
python3 tools/restore_trt_cache.py --dry-run
```

**When to run**

| Situation | Command |
|-----------|---------|
| New clone / wiped `models/`, after step 3 | `python3 tools/restore_trt_cache.py` |
| You rebuilt engines after a Triton/TRT upgrade | `python3 tools/pack_trt_cache.py` then commit the updated zip |
| Zip already restored; only Triton restarted | Skip step 4; run step 9 warmup (~seconds) |
| No T4 / different GPU | Skip step 4; use step 9 warmup (slow first time) |

## 5. Start Triton server

Models use **ONNX Runtime + TensorRT FP16** (see `config.pbtxt` under each model).
TensorRT engine files are cached under `models/model_repository/*/trt_cache/` on the
host (same volume mount as `/models` in the container).

**First start after enabling TRT** can take noticeably longer while engines build.
Subsequent starts reuse `trt_cache/` when paths match.

```bash
docker run --gpus all -d --name triton \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v $(pwd)/models/model_repository:/models \
    nvcr.io/nvidia/tritonserver:24.01-py3 \
    tritonserver --model-repository=/models \
      --backend-config=onnxruntime,default-max-batch-size=8
```

## 6. Check if the server is running

```bash
curl -s -o /dev/null -w "live %{http_code}\n" localhost:8000/v2/health/live
curl -s -o /dev/null -w "ready %{http_code}\n" localhost:8000/v2/health/ready
curl -s localhost:8000/v2/models/yolo26_face | python3 -m json.tool
curl -s localhost:8000/v2/models/arcface | python3 -m json.tool
```

`ready` must be `200` before continuing.

## 7. Build

```bash
cd ~/process_image
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## 8. Install the Triton Python client

```bash
pip install "tritonclient[grpc]"
```

## 9. Warm Triton models (YOLO + ArcFace)

Run after every Triton container start, before any video. After **step 4** this
is usually **~1–2 seconds** (loads cached engines). Without step 4, the first
run can take a long time while engines build. Changing video files does **not**
require re-warming if Triton was left running.

```bash
cd ~/process_image
python3 tools/warmup_triton.py --triton localhost:8001
```

Or separately:

```bash
python3 tools/warmup_yolo.py --triton localhost:8001
python3 tools/warmup_arcface.py --triton localhost:8001
```

Full plan and rollback: `docs/implementation-details/phase-14-tensorrt-fp16.md`.

## 10. Enroll faces into the database

`faces/` must contain one subfolder per person.

```bash
python3 tools/enroll_faces.py \
    --faces-dir ./faces/ \
    --db ./faces.db \
    --triton localhost:8001
```

## Metrics-only run

After steps 1–10, this writes per-frame metrics and a detections CSV. Skip steps 11 and 12 when you only want those files.

```bash
./build/process_image \
  --input ./input/input_3.mp4 \
  --triton localhost:8001 \
  --db ./faces.db \
  --device auto
```

## 11. Run the pipeline

```bash
./build/video_analytics_service \
  --triton localhost:8001 \
  --db ./faces.db \
  --port 50051 \
  --max-sessions 30 \
  --device auto
```

## 12. Streamlit server

```bash
cd ~/process_image
pip install -r client/requirements.txt
streamlit run --server.port 8501 --server.address 0.0.0.0 client/app.py
```

Open:

```text
https://<port>-<LIGHTNING_CLOUD_SPACE_ID>.cloudspaces.litng.ai
```

Replace `<port>` with `8501`. Replace `<LIGHTNING_CLOUD_SPACE_ID>` with the output of:

```bash
echo "$LIGHTNING_CLOUD_SPACE_ID"
```

## After a studio or container restart

Repeat from step 5, then step 6, then step 9, before step 10 or a metrics run.

If `models/` was deleted, run step 3 then **step 4** (`restore_trt_cache.py`) before
starting Triton. Steps 1–3, 7–8 can be skipped when the build and packages already exist.
