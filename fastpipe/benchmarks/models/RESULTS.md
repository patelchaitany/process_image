# Small face models: accuracy, size and speed

Tested 25 November 2026 on the RTX 3060 laptop. 17 face detectors and 14 face recognisers, all scored on public
hand-labelled test sets with the standard protocols. Model sources and licences are in [CANDIDATES.md](CANDIDATES.md).

## Answer

**Recogniser.** EdgeFace-S (14.8 MB) is the best small model. It is within 1.9 points of the current ArcFace R50
(174 MB) on all five test sets, its lowest score is 92.6, and it takes 0.094 ms per face in TensorRT against
0.514 ms, which is 5.5 times faster.

- Smaller still: EdgeFace-XS (7.3 MB, lowest score 91.7) and EdgeFace-XXS (5.1 MB, lowest score 90.2).
- Closest to the current model: EdgeFace-Base (73 MB), within 0.9 points on every set and 2.7 times faster.
- What the gap means: on frontal-against-profile pairs ArcFace R50 gets 0.7% wrong and EdgeFace-S 2.0%. On
  low-quality pairs it is 4.3% against 6.2%.
- These scores use the best threshold for each set. At the strict threshold a gallery needs, the gap is wider: see
  [What the switch costs](#what-the-switch-costs).

**Detector.** At the input size the pipeline uses now, the current YOLOv8n-face (12 MB) is already near the best
small model.

- 9 of the 17 detectors score 90 or more on both easy and medium faces. The smallest is SCRFD-1G (2.5 MB);
  SCRFD-2.5G (3.3 MB) equals the current detector there and is 3 points lower on hard faces and 5 lower on dense
  images.
- On hard faces and on dense images, no detector reaches 90 at input 640. The best is SCRFD-34G (39 MB): 85.2 and 80.0.
- Input size matters more than model size. At input 1280 the 3.3 MB SCRFD-2.5G scores 87.1 on hard faces and 85.0
  on dense images, above every model at 640. SCRFD-10G at 1280 scores 89.5 and 87.8. Easy faces drop 2 to 3 points
  at 1280.

**Licences.** Most of the best models, including the two the pipeline uses now, cannot go into a commercial
product as they are. See [Licences](#licences).

**Decision.** EdgeFace-S is now the pipeline's default recogniser; the detector is unchanged. It cuts GPU load at
40 dense cameras from 94% to 35%. It also re-matches the same face less often than ArcFace R50 once both are set to
the same false-match rate: 28% against 42% on the test clips' small faces. ArcFace R50 and EdgeFace-Base stay one
switch away. Details in [In the pipeline](#in-the-pipeline).

1. Recogniser: EdgeFace-S when camera count or GPU headroom matters most; ArcFace R50 when matching quality matters
   most. EdgeFace-Base matches R50 on the aligned test sets but not in the pipeline, so it is not a middle option.
2. Detector: keep the current one. A smaller detector saves about 0.3 ms per frame and loses dense-scene accuracy.
   For dense scenes, raise the detector input size before changing the model.
3. For a commercial product, settle the licence first. YuNet-n with SFace is the only small pair tested that is
   clearly licensed for commercial use, and it is weaker: 60 on dense images, lowest recogniser score 90.2.

## In the pipeline

Each recogniser was built as a TensorRT FP16 engine and run in `fastpipe` without other changes: all three take a
112x112 RGB face scaled to -1..1 and return 512 numbers.

### Capacity

Dense scene (about 51 faces per frame), 720p at 30 fps, detection on 10 frames per second, re-check every 3 s. The
rule is the strict one: a camera-second is done when every box, ID and name for its 30 frames is delivered, and 95%
of camera-seconds must be done within 1 s. One run per row (`results/pipeline_runs.jsonl`); a repeat of EdgeFace-S at 40
cameras gave 100% and 36% GPU.

| Recogniser | Cameras | Seconds done within 1 s | p95 delay | GPU busy | GPU memory | Video decoder busy |
|---|---|---|---|---|---|---|
| ArcFace R50 | 40 | 95.2% | 1.00 s | 94% | 3.4 GB | 51% |
| EdgeFace-Base | 40 | 100% | 0.06 s | 47% | 2.8 GB | 43% |
| **EdgeFace-S** | 40 | 100% | 0.04 s | 35% | 2.6 GB | 46% |
| EdgeFace-Base | 48 | 100% | 0.07 s | 55% | 3.2 GB | 53% |
| **EdgeFace-S** | 48 | 100% | 0.05 s | 40% | 2.9 GB | 51% |
| EdgeFace-Base | 64 | 99.3% | 0.48 s | 80% | 3.8 GB | 79% |
| **EdgeFace-S** | 64 | 99.9% | 0.48 s | 51% | 3.5 GB | 69% |

ArcFace R50 was not run above 40 cameras. Nothing was run above 64: the first EdgeFace-Base run at 64 crashed WSL
(it passed after a WSL restart), and the video decoder is 69 to 79% busy there.

### Match threshold

A cosine score does not mean the same thing for two recognisers, so each engine now carries its own thresholds in
`<engine>.thr`, written by `tools/setup/build_engines.sh` and read by `fastpipe` at start (`--match-thr` and
`--unknown-thr` override them).

| Recogniser | Match at or above | Unknown below |
|---|---|---|
| ArcFace R50 | 0.45 | 0.25 |
| EdgeFace-S | 0.58 | 0.38 |
| EdgeFace-Base | 0.46 | 0.26 |
| MobileFaceNet | 0.49 | 0.29 |

They give every model the false-match rate ArcFace R50 has at 0.45 and 0.25. On the five pair sets pooled (15,500
different-person pairs), 0.1% of different-person pairs reach 0.231 with ArcFace R50, 0.364 with EdgeFace-S, 0.242
with EdgeFace-Base and 0.267 with MobileFaceNet; each model's thresholds are R50's plus that difference
(`eval/thresholds.py`, `results/thresholds.jsonl`). The pipeline measurement below agrees for EdgeFace-S: it reaches R50's
false-match rate at 0.58 on one clip and 0.56 on the other.

### What the switch costs

Accuracy on balanced pairs (the tables further down) hides most of the difference. What a gallery system feels is
how many true matches survive at a threshold strict enough to keep false matches rare.

**Aligned test pairs.** Same-person pairs missed at each model's match threshold:

| Pair set | ArcFace R50 at 0.45 | EdgeFace-Base at 0.46 | EdgeFace-S at 0.58 |
|---|---|---|---|
| LFW (good quality) | 2.4% | 2.0% | 6.3% |
| CFP-FP (profile) | 30.1% | 31.6% | 58.1% |
| AgeDB-30 (age gap) | 43.5% | 39.4% | 68.3% |
| CPLFW (pose) | 40.7% | 40.5% | 58.2% |
| XQLFW (low quality) | 57.1% | 52.2% | 65.7% |

At these thresholds 1 of the 15,500 different-person pairs matches with ArcFace R50 and none with the other two.

**In the pipeline.** `tools/check/track_scores.py` labels the test clips by tracking: one track at two moments is the same
person, two faces in one frame are different people. Embeddings are the pipeline's own (face cut out by its box, no
landmark alignment). 450 frames of each clip:

| Clip | Recogniser | Same face matched 1 s later | 3 s later | Different people matched |
|---|---|---|---|---|
| Normal (22 faces per frame) | ArcFace R50 | 41.8% | 29.5% | 0.52% |
| | EdgeFace-Base | 28.7% | 19.2% | 0.30% |
| | EdgeFace-S | 28.3% | 15.8% | 0.53% |
| Dense (51 faces per frame) | ArcFace R50 | 42.7% | 16.5% | 0.71% |
| | EdgeFace-Base | 28.0% | 8.2% | 0.46% |
| | EdgeFace-S | 28.6% | 11.6% | 0.51% |

- **These clips are mostly tiny faces.** 77% of the same-face pairs in the normal clip and 90% in the dense clip are
  under 32 px wide. For faces 80 px or wider (58 pairs, normal clip) the 1 s figures are 79% for ArcFace R50, 71% for
  EdgeFace-Base and 67% for EdgeFace-S.
- **All three are far below their test-set accuracy here.** Small faces and box crops without alignment cost more
  than the choice of recogniser. A failed re-check also clears the track's name, so names come and go.
- **EdgeFace-Base loses its advantage in the pipeline.** Its embeddings also agree least with the reference model on
  pipeline crops (below).

### Embedding check

Pipeline embeddings (NV12 crop on the GPU, FP16 engine) against the reference (OpenCV crop, ONNX Runtime FP32) for
the same 999 boxes, `tools/check/compare_dump.py`: mean cosine 0.987 for ArcFace R50, 0.987 for EdgeFace-S and 0.977 for
EdgeFace-Base.

### Switching recogniser

- Tools: `REC=r50` (or `edgeface_base`, `mbf`) before any script in `tools/`; `stress.sh` takes it as an argument.
- Camera test page: `FP_REC=rec_r50_fp16.plan` in the bridge's environment.
- Directly: `fastpipe --rec engines/rec_r50_fp16.plan`. The thresholds follow the engine.

## How it was tested

- **Detectors:** WIDER FACE validation set, 3,226 images with 39,708 labelled faces, official protocol (a detection
  counts when it overlaps a labelled face by 50% or more). Scores are average precision on the easy (7,211 faces),
  medium (13,319) and hard (31,958) subsets. Every model gets the same picture: longer side scaled to 640, padded
  to a square, one pass.
- **Dense hard:** the hard subset on the 245 images that have 30 or more labelled faces (17,184 faces).
- **Found at 90% precision:** the share of labelled faces found at the score threshold where 9 of 10 boxes are
  real faces.
- **Recognisers:** same-person / different-person pairs of aligned 112x112 faces, 10-fold accuracy: LFW (6,000
  pairs), CFP-FP (7,000, frontal against profile), AgeDB-30 (6,000, large age gap), CPLFW (6,000, pose), XQLFW
  (6,000, low quality against good quality). The embedding is the face plus its mirror image, the usual protocol.
- **Small faces:** LFW with the face shrunk to the stated width and scaled back up. "Both" shrinks both photos of
  a pair (camera against camera); "one" shrinks only the second (camera against a good enrolment photo).
- **Speed:** TensorRT 10.16, FP16, GPU compute time only. Detectors: one 640x640 image. Recognisers: batch of 64
  where the file allows it. SFace and the four MobileNet files are fixed at batch 1, so their per-face time is not
  comparable with the others.

**Check against published figures.** The five SCRFD models land within 0.35 points of the SCRFD README on all three
subsets. ArcFace R50 and MobileFaceNet land within 0.25 points of the InsightFace model zoo. EdgeFace matches its
README within 0.1 on LFW, AgeDB-30 and CPLFW, but scores 1.9 to 2.7 points higher than the README on CFP-FP; I have
not explained that difference.

## Detectors

| Model | Size MB | Easy | Medium | Hard | Dense hard | Dense hard found at 90% precision | TensorRT ms / frame | Licence | Commercial use |
|---|---|---|---|---|---|---|---|---|---|
| yunet-s | 0.2 | 87.7 | 83.7 | 59.8 | 45.7 | 30% | 0.42 | MIT | yes |
| yunet-n | 0.3 | 90.0 | 87.0 | 69.9 | 60.3 | 44% | 0.50 | BSD-3 | yes |
| retinaface-mnet0.25 | 1.7 | 89.3 | 83.2 | 54.5 | 37.6 | 21% | 0.65 | MIT | yes* |
| scrfd-500m | 2.5 | 90.8 | 88.3 | 69.2 | 58.8 | 42% | 0.46 | InsightFace weights | no |
| scrfd-1g | 2.5 | 92.3 | 90.6 | 74.5 | 65.9 | 50% | 0.82 | InsightFace weights | no |
| scrfd-2.5g | 3.3 | 93.8 | 91.9 | 77.0 | 69.2 | 55% | 0.69 | InsightFace weights | no |
| centerface | 7.3 | 92.5 | 89.2 | 59.1 | 41.2 | 28% | 1.10 | MIT | yes* |
| yolov10n-face | 9.3 | 93.6 | 91.5 | 78.4 | 70.4 | 56% | 1.16 | GPL-3.0 | copyleft |
| yolov11n-face | 10.6 | 93.9 | 91.6 | 80.0 | 73.9 | 59% | 1.20 | GPL-3.0 | copyleft |
| yolov12n-face | 10.6 | 92.8 | 89.6 | 71.4 | 63.1 | 58% | 2.90 | GPL-3.0 | copyleft |
| yolov5n-face | 11.0 | 93.9 | 91.7 | 81.3 | 75.7 | 64% | 1.62 | GPL-3.0 | copyleft |
| yolov8n-face (current) | 12.2 | 93.6 | 91.5 | 79.9 | 74.0 | 60% | 1.02 | AGPL-3.0 (Ultralytics) | copyleft |
| retinaface-mnetv2 | 12.5 | 93.3 | 89.7 | 68.3 | 56.1 | 38% | 1.47 | MIT | yes* |
| yolov8n-face (akanametov) | 12.7 | 93.3 | 89.5 | 68.9 | 59.5 | 58% | 1.80 | GPL-3.0 | copyleft |
| scrfd-10g | 16.9 | 95.3 | 93.9 | 82.7 | 77.1 | 65% | 1.65 | InsightFace weights | no |
| yolov5s-face | 29.2 | 94.5 | 92.8 | 84.0 | 79.5 | 69% | 1.54 | GPL-3.0 | copyleft |
| scrfd-34g | 39.2 | 96.0 | 94.8 | 85.2 | 80.0 | 68% | 6.74 | InsightFace weights | no |

`yes*`: MIT code, but the weights were trained on WIDER FACE, whose images are licensed for non-commercial use.

### Larger input

Only the three SCRFD files with a free input size were run at 1280. Time here is ONNX Runtime per image; TensorRT
was not timed at 1280.

| Model | Size MB | Input | Easy | Medium | Hard | Dense hard | Dense hard found at 90% precision | ONNX Runtime ms / image |
|---|---|---|---|---|---|---|---|---|
| scrfd-500m | 2.5 | 640 | 90.8 | 88.3 | 69.2 | 58.8 | 42% | 14.9 |
| scrfd-500m | 2.5 | 1280 | 89.0 | 88.6 | 83.5 | 80.7 | 69% | 30.8 |
| scrfd-2.5g | 3.3 | 640 | 93.8 | 91.9 | 77.0 | 69.2 | 55% | 13.3 |
| scrfd-2.5g | 3.3 | 1280 | 91.0 | 91.2 | 87.1 | 85.0 | 75% | 28.3 |
| scrfd-10g | 16.9 | 640 | 95.3 | 93.9 | 82.7 | 77.1 | 65% | 23.3 |
| scrfd-10g | 16.9 | 1280 | 92.5 | 92.9 | 89.5 | 87.8 | 79% | 55.4 |

## Recognisers

| Model | Size MB | LFW | CFP-FP | AgeDB-30 | CPLFW | XQLFW | Lowest | TensorRT ms / face | Licence | Commercial use |
|---|---|---|---|---|---|---|---|---|---|---|
| mobilenetv1-0.25 | 1.5 | 98.58 | 86.91 | 90.13 | 82.45 | 64.12 | 64.1 | 0.205 (batch 1) | MIT code, research data | unclear |
| mobilenetv3-small | 5.0 | 99.25 | 89.36 | 92.85 | 85.05 | 71.05 | 71.0 | 0.268 (batch 1) | MIT code, research data | unclear |
| edgeface-xxs | 5.1 | 99.57 | 96.16 | 94.97 | 90.22 | 91.80 | 90.2 | 0.042 (batch 64) | CC BY-NC-SA 4.0 | no |
| edgeface-xs | 7.3 | 99.73 | 97.41 | 96.05 | 91.65 | 92.90 | 91.7 | 0.074 (batch 64) | CC BY-NC-SA 4.0 | no |
| mobilenetv2 | 9.1 | 99.53 | 92.53 | 95.03 | 86.98 | 64.70 | 64.7 | 0.443 (batch 1) | MIT code, research data | unclear |
| mobilefacenet-w600k | 13.6 | 99.68 | 97.77 | 96.43 | 92.52 | 92.65 | 92.5 | 0.264 (batch 64) | InsightFace weights | no |
| mobilenetv3-large | 14.0 | 99.42 | 92.66 | 95.33 | 86.98 | 70.77 | 70.8 | 0.406 (batch 1) | MIT code, research data | unclear |
| edgeface-s | 14.8 | 99.77 | 98.00 | 97.03 | 92.57 | 93.78 | 92.6 | 0.094 (batch 64) | CC BY-NC-SA 4.0 | no |
| sface | 38.7 | 99.68 | 97.03 | 94.83 | 91.17 | 90.18 | 90.2 | 0.398 (batch 1) | Apache-2.0 | yes |
| edgeface-base | 73.1 | 99.83 | 98.90 | 97.60 | 93.73 | 94.78 | 93.7 | 0.188 (batch 64) | CC BY-NC-SA 4.0 | no |
| adaface-ir18 | 96.1 | 99.58 | 97.71 | 96.32 | 92.28 | 92.43 | 92.3 | 0.265 (batch 64) | MIT code, research data | unclear |
| arcface-r50-w600k (current) | 174.4 | 99.82 | 99.27 | 98.15 | 94.45 | 95.67 | 94.5 | 0.514 (batch 64) | InsightFace weights | no |
| adaface-ir50 | 174.4 | 99.82 | 98.87 | 97.73 | 94.15 | 94.92 | 94.2 | 0.529 (batch 64) | MIT code, research data | unclear |
| auraface-r100 | 260.7 | 99.48 | 95.46 | 95.73 | 89.78 | 75.75 | 75.8 | 0.927 (batch 64) | Apache-2.0 | yes |

### Without the mirror image

The pipeline embeds each face once, without its mirror image. The ten leading models were run again that way. The
order does not change.

| Model | Size MB | LFW | CFP-FP | AgeDB-30 | CPLFW | XQLFW | Lowest | Largest change from the table above |
|---|---|---|---|---|---|---|---|---|
| edgeface-xxs | 5.1 | 99.58 | 96.27 | 94.87 | 90.37 | 91.80 | 90.4 | +0.15 |
| edgeface-xs | 7.3 | 99.75 | 97.37 | 96.02 | 91.82 | 92.73 | 91.8 | -0.17 |
| mobilefacenet-w600k | 13.6 | 99.62 | 97.84 | 96.48 | 92.47 | 92.38 | 92.4 | -0.27 |
| edgeface-s | 14.8 | 99.82 | 98.06 | 96.83 | 92.40 | 93.88 | 92.4 | -0.20 |
| sface | 38.7 | 99.63 | 96.97 | 94.92 | 90.53 | 90.25 | 90.2 | -0.64 |
| edgeface-base | 73.1 | 99.83 | 98.84 | 97.57 | 93.87 | 94.62 | 93.9 | -0.16 |
| adaface-ir18 | 96.1 | 99.58 | 97.53 | 96.38 | 91.80 | 92.10 | 91.8 | -0.48 |
| arcface-r50-w600k (current) | 174.4 | 99.80 | 99.24 | 98.03 | 94.40 | 95.42 | 94.4 | -0.25 |
| adaface-ir50 | 174.4 | 99.78 | 98.97 | 97.57 | 93.90 | 94.83 | 93.9 | -0.25 |
| auraface-r100 | 260.7 | 99.23 | 94.30 | 95.15 | 88.97 | 75.73 | 75.7 | -1.16 |

### Small faces

| Model | Size MB | Both 28 px | Both 20 px | Both 14 px | One 28 px | One 20 px | One 14 px |
|---|---|---|---|---|---|---|---|
| mobilenetv1-0.25 | 1.5 | 96.43 | 92.82 | 83.40 | 97.57 | 93.92 | 84.18 |
| mobilenetv3-small | 5.0 | 97.90 | 95.03 | 88.92 | 98.47 | 96.57 | 90.00 |
| edgeface-xxs | 5.1 | 99.27 | 98.45 | 95.85 | 99.42 | 99.23 | 97.27 |
| edgeface-xs | 7.3 | 99.52 | 98.77 | 96.77 | 99.57 | 99.33 | 98.02 |
| mobilenetv2 | 9.1 | 98.75 | 96.52 | 85.43 | 99.25 | 97.68 | 88.35 |
| mobilefacenet-w600k | 13.6 | 99.38 | 98.95 | 96.63 | 99.60 | 99.35 | 98.53 |
| mobilenetv3-large | 14.0 | 98.80 | 96.27 | 88.68 | 99.03 | 97.28 | 90.52 |
| edgeface-s | 14.8 | 99.63 | 99.15 | 97.48 | 99.75 | 99.48 | 98.55 |
| sface | 38.7 | 99.27 | 98.50 | 95.78 | 99.47 | 99.22 | 97.35 |
| edgeface-base | 73.1 | 99.82 | 99.48 | 98.40 | 99.78 | 99.73 | 99.08 |
| adaface-ir18 | 96.1 | 99.35 | 99.00 | 97.18 | 99.52 | 99.43 | 98.68 |
| arcface-r50-w600k (current) | 174.4 | 99.73 | 99.63 | 98.70 | 99.75 | 99.75 | 99.38 |
| adaface-ir50 | 174.4 | 99.70 | 99.65 | 99.15 | 99.80 | 99.68 | 99.52 |
| auraface-r100 | 260.7 | 98.83 | 97.40 | 90.72 | 99.18 | 98.35 | 92.83 |

## Licences

| Model | Licence | Commercial product |
|---|---|---|
| YOLOv8n-face (current detector) | Trained with Ultralytics: AGPL-3.0 | Only if the product's code is released under AGPL, or with a licence from Ultralytics |
| ArcFace R50, MobileFaceNet, all SCRFD (InsightFace) | Code MIT, weights non-commercial research only | No |
| EdgeFace | Code BSD-3, weights CC BY-NC-SA 4.0 | No |
| YOLOv5-face, akanametov YOLO-face | GPL-3.0 | Copyleft |
| AdaFace, yakhyo MobileNets | Code MIT, trained on research-only data | Unclear |
| YuNet | MIT / BSD-3 | Yes |
| SFace | Apache-2.0 | Yes |
| AuraFace R100 | Apache-2.0 | Yes, but 75.8 on low-quality pairs |

The routes to a commercial product are a licence from the owner, training the same architecture on data you may
use (SCRFD and EdgeFace code are permissively licensed), or YuNet with SFace as they are.

## Not tested

- **Landmark alignment in the pipeline.** The pipeline crops each face by its box; the current detector gives no
  landmarks. The test-set scores are on five-point-aligned faces, and the pipeline measurement above shows how much
  lower matching is without it. SCRFD (the KPS files), YuNet and YOLOv5-face output the five landmarks.
- **Matching against enrolment photos.** The pipeline measurement compares a face with itself seconds later, not with
  a separate enrolment photo, and the people in one clip are few (about 22 and 51).
- **CCTV footage with known identities.** The pair sets are photographs. XQLFW and the shrunk LFW are the nearest
  stand-ins for camera quality.
- **More than 64 cameras**, and ArcFace R50 above 40.
- **8-bit quantised models**, and models without a downloadable ONNX file (listed in CANDIDATES.md).

## Files

| File | Content |
|---|---|
| `CANDIDATES.md` | Every model considered: source, download link, size, published accuracy, licence |
| `results/det_results.jsonl`, `results/det_results_size.jsonl` | Detector results at input 640 and at 1280 |
| `results/rec_results.jsonl`, `results/rec_results_noflip.jsonl` | Recogniser results with and without the mirror image |
| `results/thresholds.jsonl` | Score distributions on the pair sets, for the match thresholds |
| `results/track_scores.jsonl` | Match scores inside the pipeline on the two test clips |
| `results/pipeline_runs.jsonl` | The capacity runs in the pipeline |
| `results/trt_speed.log` | TensorRT timings |
| `eval/eval_det.py`, `eval/eval_rec.py` | The evaluators: `python eval_det.py <model>`, `python eval_rec.py <model>` |
| `eval/thresholds.py` | Produces `thresholds.jsonl` |
| `eval/make_tables.py` | Prints the tables above from `results/` |
| `eval/onnx_info.py` | Prints a model file's inputs and outputs |
| `eval/run/` | The batch files that ran everything on the test machine: model download, each evaluation, TensorRT timing |
| `tools/check/track_scores.py` (in `fastpipe/tools`) | Produces `track_scores.jsonl` from a pipeline dump |
