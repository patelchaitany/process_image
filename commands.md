# Setup & Run Commands (GPU Cluster - Ubuntu 24.04)

## 1. Verify GPU

```bash
nvidia-smi
```

## 2. Run setup.sh (apt packages + Triton C++ Client SDK + Python basics)

```bash
cd ~/process_image
chmod +x setup.sh
sudo ./setup.sh
```

## 3. Install NVIDIA Container Toolkit (for Docker GPU support)

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## 4. Install FAISS-GPU

```bash
# Option A: via conda
conda install -y -c pytorch -c nvidia faiss-gpu

# Option B: via pip (CPU-only fallback)
pip3 install --break-system-packages faiss-cpu
```

## 5. Install Python dependencies for model download

```bash
pip3 install --break-system-packages onnx onnxruntime numpy opencv-python insightface ultralytics
```

## 6. Download real pre-trained ONNX models

```bash
cd ~/process_image
python3 tools/download_models.py --real --gpu
```

This creates:
```
models/model_repository/
├── yolo26_face/         # YOLO face detection  [N,3,640,640] → [N,8400,5]
│   ├── config.pbtxt
│   └── 1/model.onnx
└── arcface/             # ArcFace recognition  [N,3,112,112] → [N,512]
    ├── config.pbtxt
    └── 1/model.onnx
```

## 7. Start Triton Inference Server (Docker, GPU)

```bash
cd ~/process_image

docker run --gpus all -d --name triton \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v $(pwd)/models/model_repository:/models \
    nvcr.io/nvidia/tritonserver:24.01-py3 \
    tritonserver --model-repository=/models
```

Wait ~20 seconds, then verify:

```bash
curl -s localhost:8000/v2/health/live
curl -s localhost:8000/v2/models | python3 -m json.tool
```

## 8. Build the project

```bash
cd ~/process_image

# T4 GPU (default, compute capability 75)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# For other GPUs, override the architecture:
#   A100  → -DCMAKE_CUDA_ARCHITECTURES=80
#   A10   → -DCMAKE_CUDA_ARCHITECTURES=86
#   L4    → -DCMAKE_CUDA_ARCHITECTURES=89
#   H100  → -DCMAKE_CUDA_ARCHITECTURES=90
```

## 9. Enroll faces into the database

```bash
cd ~/process_image

# faces/ directory should have subfolders per person:
#   faces/alice/photo1.jpg
#   faces/bob/photo1.jpg photo2.jpg
#   ...

python3 tools/enroll_faces.py \
    --faces-dir ./faces/ \
    --db ./faces.db \
    --triton localhost:8001
```

## 10. Run the pipeline

```bash
cd ~/process_image

./build/process_image \
    --input /path/to/video.mp4 \
    --triton localhost:8001 \
    --db ./faces.db \
    --config config/pipeline.yaml \
    --bbox-csv ./output/detections.csv
```

### All CLI options

```
--input <path|url>        MP4 file path or RTSP URL (required)
--triton <url>            Triton server URL (default: localhost:8001)
--config <path>           YAML config file (default: config/pipeline.yaml)
--db <path>               SQLite face database (default: ./faces.db)
--output-csv <path>       Metrics CSV path (default: ./metrics/metrics.csv)
--bbox-csv <path>         Detection CSV path (default: ./output/detections.csv)
--confidence <float>      YOLO confidence threshold (default: 0.5)
--match-threshold <float> Face similarity threshold (default: 0.6)
--device <gpu|cpu|auto>   Force device mode (default: auto)
--no-console              Suppress per-frame console output
```

## Cleanup

```bash
# Stop Triton
docker stop triton && docker rm triton

# Clean build
rm -rf build/
```

## Media server 

docker run --rm \
  --name mediamtx \
  --network host \
  bluenviron/mediamtx:1
