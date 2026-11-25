# Tools

Scripts for measuring, checking and testing the pipeline. None of them is needed to run it: the pipeline and its
setup scripts are in `../../fastpipe/`.

Lines marked (WSL) run on the GPU machine inside WSL, (Windows) in the Windows Python environment there, (Mac) on the
machine you work from. Paths inside the scripts are the benchmark machine's (`E:\bench`, `/root/fastpipe`). The WSL
scripts can be run where they are: `bash /mnt/e/bench/fastpipe-dev/tools/check/dump720.sh`.

`REC=<name>` picks the recogniser engine for any script that runs `fastpipe`: `edgeface_s` (default),
`edgeface_base`, `r50`, `mbf`.

## bench/: speed and capacity

| Script | Purpose |
|---|---|
| `stress.sh` | (WSL) Stress run at given camera counts: `stress.sh <tag> <recogniser> <counts> [flags]`. `slarun.sh` expects a copy in `E:\bench\wsl` |
| `slarun.sh` | (Mac) One stress run over ssh, printing the end-to-end delay figures |
| `kb.sh` | (WSL) GPU time and memory bandwidth of each custom kernel |
| `prof.sh` | (WSL) Nsight Systems timeline at a camera count |
| `profile.sh` | (WSL) Nsight Systems timeline, then Nsight Compute on the custom kernels |
| `layer_report.py` | Summarise a `trtexec` layer profile: slowest layers and copies at the engine boundary |

## check/: is the output right

| Script | Purpose |
|---|---|
| `dump720.sh` | (WSL) Dump boxes and embeddings for 60 frames of the 720p clip |
| `compare_dump.py` | (Windows) Compare a dump with ONNX Runtime: detector input, detections, embeddings |
| `boxeval.sh` | (WSL) Boxes skipped or added when detecting on every 2nd or 3rd frame |
| `kfeval.sh`, `kfeval_dense.sh` | (WSL) Kalman-filled boxes against detected boxes, normal and dense clip |
| `kfsweep.sh` | (WSL) The same over a grid of Kalman noise settings |
| `track_scores.sh` | (WSL) Dump 450 frames of both test clips once per recogniser |
| `track_scores.py` | (Windows) Same-person and different-people match scores from such a dump, labelled by tracking |

## clips/: test video

| Script | Purpose |
|---|---|
| `mkclips.sh` | (WSL) 1080p clips as CCTV cameras send them: 10 fps H.264 and 25 fps H.265 |
| `mkclips30.sh` | (WSL) 1080p clips at 30 fps |
| `mkclips720.sh` | (WSL) 720p clips at 30 fps, H.264 and H.265 |
| `mkdense.sh`, `mkdense45.sh` | (WSL) Dense 720p clips (about 51 faces per frame) built from bands of the crowd clip |

## int8/: the 8-bit experiment

| Script | Purpose |
|---|---|
| `fix_opset.py` | (Windows) Make the 8-bit ArcFace R50 ONNX file acceptable to TensorRT |
| `cmp_int8.py` | Compare the embeddings of two dumps (8-bit against 16-bit float engine) |

## exp/: one experiment at a time on the remote GPU machine

Each takes a lock, so several people or agents can share one GPU. They need `FP_SSH_DIR`: a folder holding an
`ssh_config` with a `Host laptop` entry.

| Script | Purpose |
|---|---|
| `exp_sla.sh` | One 24 s stress run under the strict end-to-end rule; prints one JSON line |
| `exp_boxeval.sh` | Skipped and extra boxes on 540 frames |
| `exp_build.sh` | Build a modified copy of the sources as `/root/fp-<name>` |
| `exp_wsl.sh`, `exp_win.sh` | Run a script inside WSL, or a command on Windows, under the lock |
| `exp_wsl_restart.sh` | Restart WSL after an out-of-memory crash |
| `lock.sh` | The lock and the ssh settings, sourced by the others |
| `wsl/` | The WSL halves of the scripts above |
