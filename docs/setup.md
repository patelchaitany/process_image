# Setup Guide

## Platform Support


| Platform                      | GPU Mode (~13ms)           | CPU Mode (~100-200ms) |
| ----------------------------- | -------------------------- | --------------------- |
| Linux + NVIDIA T4/A100        | Yes                        | Yes                   |
| Linux (no GPU)                | No                         | Yes                   |
| macOS (Apple Silicon / Intel) | No                         | Yes                   |
| Docker (any host)             | Depends on GPU passthrough | Yes                   |


macOS does not support CUDA. The pipeline auto-detects this and falls back to CPU mode using ONNX Runtime (via Triton) and OpenCV for preprocessing.

---

## Prerequisites

### All Platforms

- CMake >= 3.18
- C++17 compiler (GCC 9+, Clang 11+, Apple Clang 13+)
- FFmpeg >= 5.0 (libavcodec, libavformat, libswscale)
- SQLite3 >= 3.36
- yaml-cpp >= 0.7
- Python 3.8+ (for enrollment tool)
- Docker (for running Triton Inference Server)

### GPU Mode (Linux only)

- NVIDIA GPU (T4 recommended, any Turing+ works)
- CUDA Toolkit >= 11.8
- NVIDIA Triton Inference Server (with TensorRT backend)
- FAISS-GPU >= 1.7
- Triton C++ Client Library

### CPU Mode (Mac / Linux without GPU)

- OpenCV >= 4.5 (for preprocessing)
- FAISS-CPU >= 1.7
- Triton Inference Server (CPU-only, via Docker)

---

## macOS Setup (Apple Silicon / Intel)

### 1. Install system dependencies

```bash
# Using Homebrew
brew install cmake ffmpeg sqlite yaml-cpp opencv faiss pkg-config

# Python dependencies
pip3 install tritonclient[grpc] numpy opencv-python
```

### 2. Start Triton Inference Server (Docker, CPU mode)

```bash
# Pull the CPU-only Triton image
docker pull nvcr.io/nvidia/tritonserver:24.01-py3

# Start Triton with your model repository
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v /path/to/models/model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models --backend-config=onnxruntime,default-max-batch-size=8
```

Make sure your model configs use `KIND_CPU`:

```
instance_group [{ kind: KIND_CPU }]
```

### 3. Build the pipeline

```bash
cd /path/to/process_image

# Configure (CUDA will not be found on Mac -- this is expected)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -j$(sysctl -n hw.ncpu)
```

If CMake cannot find CUDA, it will produce an error since the project declares `LANGUAGES CXX CUDA`. To build on Mac, temporarily comment out CUDA from the project declaration or use the conditional approach:

```bash
# Build with CPU-only mode
cmake -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=OFF
```

### 4. Set up face database

```bash
# Create faces directory with person subfolders
mkdir -p faces/alice faces/bob

# Add photos (one or more per person)
cp ~/photos/alice/*.jpg faces/alice/
cp ~/photos/bob/*.jpg faces/bob/

# Run enrollment
python3 tools/enroll_faces.py \
  --faces-dir ./faces/ \
  --db ./faces.db \
  --triton localhost:8001
```

### 5. Run the pipeline

```bash
./build/process_image \
  --input video.mp4 \
  --triton localhost:8001 \
  --device cpu \
  --db ./faces.db
```

---

## Linux Setup (with NVIDIA GPU)

### 1. Install system dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake pkg-config \
  libavcodec-dev libavformat-dev libswscale-dev libavutil-dev \
  libsqlite3-dev \
  libyaml-cpp-dev \
  libopencv-dev

# CUDA Toolkit (if not already installed)
# Follow: https://developer.nvidia.com/cuda-downloads

# FAISS-GPU
conda install -c pytorch faiss-gpu
# Or build from source: https://github.com/facebookresearch/faiss

# Triton C++ Client
# Download from: https://github.com/triton-inference-server/client
# Or install via the Triton client SDK package

# Python
pip install tritonclient[grpc] numpy opencv-python
```

### 2. Start Triton Inference Server

```bash
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v /path/to/models/model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models
```

### 3. Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### 4. Run

```bash
./build/process_image \
  --input video.mp4 \
  --triton localhost:8001 \
  --db ./faces.db \
  --output-csv ./metrics/results.csv
```

---

## Docker Setup (Any Platform)

For a fully containerized setup:

```bash
# Build the pipeline container
docker build -t process_image .

# Run (CPU mode)
docker run --rm \
  -v $(pwd)/data:/data \
  process_image \
  --input /data/video.mp4 \
  --triton host.docker.internal:8001 \
  --db /data/faces.db \
  --device cpu

# Run (GPU mode, Linux with NVIDIA runtime)
docker run --gpus all --rm \
  -v $(pwd)/data:/data \
  process_image \
  --input /data/video.mp4 \
  --triton host.docker.internal:8001 \
  --db /data/faces.db
```

---

## Model Repository Setup

The Triton model repository needs two models:

```
model_repository/
├── yolo26_face/
│   ├── config.pbtxt
│   └── 1/
│       └── model.onnx
└── arcface/
    ├── config.pbtxt
    └── 1/
        └── model.onnx
```

### yolo26_face/config.pbtxt

```protobuf
name: "yolo26_face"
platform: "onnxruntime_onnx"
max_batch_size: 8

input [
  {
    name: "images"
    data_type: TYPE_FP32
    dims: [ 3, 640, 640 ]
  }
]
output [
  {
    name: "output0"
    data_type: TYPE_FP32
    dims: [ 8400, 5 ]
  }
]

dynamic_batching {
  preferred_batch_size: [ 1, 4, 8 ]
  max_queue_delay_microseconds: 100
}

instance_group [{ kind: KIND_CPU }]  # Use KIND_GPU for GPU mode
```

### arcface/config.pbtxt

```protobuf
name: "arcface"
platform: "onnxruntime_onnx"
max_batch_size: 32

input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [ 3, 112, 112 ]
  }
]
output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 512 ]
  }
]

dynamic_batching {
  preferred_batch_size: [ 1, 4, 8 ]
  max_queue_delay_microseconds: 100
}

instance_group [{ kind: KIND_CPU }]  # Use KIND_GPU for GPU mode
```

---

## Verifying the Setup

```bash
# 1. Check Triton is running
curl -s localhost:8000/v2/health/live
# Should return: {"live":true}

# 2. Check models are loaded
curl -s localhost:8000/v2/models | python3 -m json.tool

# 3. Run NMS unit tests (no GPU/Triton needed)
./build/test_nms

# 4. Run face database tests (no GPU/Triton needed)
./build/test_face_database

# 5. Run enrollment on test faces
python3 tools/enroll_faces.py --faces-dir ./test_faces/ --db ./test.db --triton localhost:8001

# 6. Run full pipeline on a video
./build/process_image --input test_video.mp4 --triton localhost:8001 --db ./test.db
```

---

## Troubleshooting


| Issue                                 | Solution                                                    |
| ------------------------------------- | ----------------------------------------------------------- |
| `CMake Error: No CUDA toolkits found` | Expected on Mac. Use `-DENABLE_CUDA=OFF` or run in CPU mode |
| `Triton connection refused`           | Ensure Docker container is running, check port 8001         |
| `Model not found`                     | Verify model_repository path and model names match config   |
| `FFmpeg: No such file or directory`   | Ensure input file path is correct and readable              |
| `FAISS not found`                     | Install via conda (`faiss-cpu` on Mac) or from source       |
| `Low FPS in CPU mode`                 | Expected (~5-10 fps). Use GPU for real-time performance     |


