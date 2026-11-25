# fastpipe

A multi-camera face pipeline that keeps every frame on the GPU: C++17, CUDA and TensorRT, one process. It is a second
implementation beside the Triton pipeline in this repository's `src/`, built to run as many cameras as possible on one
GPU at a bounded delay.

Per batch of frames, one CUDA graph runs: NVDEC decode → detect (YOLOv8n-face, 640×384) → track (Kalman, on the GPU)
→ zone lookup → recognise (EdgeFace-S) → gallery match. Out come boxes, track IDs, names and zones for every frame.
Design and kernel details: [ARCHITECTURE.md](ARCHITECTURE.md).

## Measured

RTX 3060 Laptop GPU (6 GB), WSL2, CUDA 12.6, TensorRT 10.16. Cameras are 720p at 30 fps showing a dense scene (about
51 faces per frame); detection runs on 10 frames per second and the tracker fills the rest. A camera-second is done
when every box, ID and name for its 30 frames is delivered; the rule is 95% of camera-seconds done within 1 s.

| Recogniser | Cameras | Seconds done within 1 s | p95 delay | GPU busy |
|---|---|---|---|---|
| EdgeFace-S (default) | 40 | 100% | 0.04 s | 35% |
| EdgeFace-S | 48 | 100% | 0.05 s | 40% |
| EdgeFace-S | 64 | 99.9% | 0.48 s | 51% |
| ArcFace R50 | 40 | 95.2% | 1.00 s | 94% |

EdgeFace-S buys the headroom and costs matching quality: at the same false-match rate it re-matches the same face less
often than ArcFace R50. The numbers, the comparison of 17 detectors and 14 recognisers on labelled test sets, and the
licences are in [benchmarks/models/RESULTS.md](benchmarks/models/RESULTS.md).

## Layout

| Path | Content |
|---|---|
| `src/` | The pipeline: decoder, CUDA kernels, GPU tracker, TensorRT wrapper, scheduler |
| `tools/` | Engine build, stress and profiling scripts, validation against ONNX Runtime |
| `live/` | Browser test page for real cameras: draw zones, name faces ([live/README.md](live/README.md)) |
| `benchmarks/models/` | Model comparison: results, evaluators, candidate list |

## Build and run

Linux or WSL2 with CUDA 12.x, TensorRT 10.x and the FFmpeg 6 development packages (`tools/wsl_trt_install.sh`).

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
tools/build_engines.sh        # TensorRT FP16 engines and each recogniser's match thresholds
build/fastpipe --det engines/det_r384_fp16.plan --rec engines/rec_edgeface_s_fp16.plan --counts 16,32,40
```

The scripts in `tools/` carry the paths of the machine they were written on (`E:\bench` on Windows, `/root/fastpipe`
in WSL); change them for yours.

## Models

Model files are not in this repository. `benchmarks/models/CANDIDATES.md` lists where each comes from, and
`tools/prep_onnx.py` prepares the ONNX files the engines are built from.

| Role | Default | Other engines |
|---|---|---|
| Detector | YOLOv8n-face (lindevs), 640×384 | |
| Recogniser | EdgeFace-S | ArcFace R50, EdgeFace-Base, MobileFaceNet (`--rec`, or `REC=r50` for the tools) |

Each recogniser engine has a `.thr` file with its own match thresholds, because cosine scores are not comparable
between models.

**Licences.** The YOLOv8n-face weights are AGPL-3.0 (trained with Ultralytics). The EdgeFace weights are CC BY-NC-SA
4.0 and the InsightFace weights (ArcFace R50, MobileFaceNet) are for non-commercial research. None of these can go
into a commercial product as they are; see the licence table in the results.
