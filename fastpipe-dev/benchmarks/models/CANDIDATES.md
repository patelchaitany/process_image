# Face model candidates for the CCTV benchmark (checked 2026-11-25)

How this was checked:
- Every URL marked **OK** was checked with `curl -sIL`: the final status was HTTP 200 and the size is the final `content-length`. MB means MiB (bytes / 1048576).
- For each ONNX file, **I/O** is the model's real graph input and output signature. It was read from the last 300 to 400 KB of the file with an HTTP Range request (graph inputs and outputs are stored after the weights) and parsed with a small protobuf reader. No whole model over 5 MB was downloaded.
- **not confirmed** means I could not verify the item. Accuracy figures are copied from the linked source, along with its test protocol.
- Licence summary: InsightFace code is MIT, but **all InsightFace pretrained models are for non-commercial research only**. The InsightFace model zoo README and the python-package README both state this, and it covers every `buffalo_*`, `antelopev2` and SCRFD weight file, including re-hosted copies.

---

## 1. Detectors

| # | Model | Repo / page | Direct weights URL (verified) | Size | Params / FLOPs | Input (verified I/O + preprocessing) | Output | WIDER FACE val (Easy / Med / Hard), protocol, source | Licence (code / weights), commercial? |
|---|---|---|---|---|---|---|---|---|---|
| D1 | **SCRFD-500M-KPS** (`det_500m.onnx`, buffalo_s / buffalo_sc) | https://github.com/deepinsight/insightface/tree/master/detection/scrfd | https://github.com/yakhyo/uniface/releases/download/weights/scrfd_500m_kps.onnx **OK** (byte-identical size to `https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_500m.onnx`) | 2.41 MB (2,524,817 B) | 0.57 M / 0.5 GF @VGA | `input.1` [1,3,?,?] dynamic H,W. RGB, (x-127.5)/128 | 9 tensors without a batch dim: scores [N,1] x3, bbox [N,4] x3, kps [N,10] x3 for strides 8/16/32 (N = 12800/3200/800 at 640x640) | 90.97 / 88.44 / 69.49 at VGA (640x480) — [SCRFD README](https://github.com/deepinsight/insightface/blob/master/detection/scrfd/README.md). The non-KPS 500M scores 90.91 / 89.49 / **82.03** at original size | MIT / InsightFace non-commercial. No |
| D2 | **SCRFD-2.5G-KPS** (`det_2.5g.onnx`, buffalo_m) | same | https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_2.5g.onnx (3,292,009 B; release API, I/O read OK). Alternatives: https://huggingface.co/RuteNL/SCRFD-face-detection-ONNX/resolve/main/2.5g_bnkps.onnx **OK** 3.13 MB; https://huggingface.co/hsuyabc/scrfd_2.5g_bnkps.onnx/resolve/main/scrfd_2.5g_bnkps.onnx **OK** 3.13 MB | 3.14 MB | 0.82 M / 2.5 GF | buffalo_m copy: `input.1` [1,3,?,?], outputs dynamic [?,1] / [?,4] / [?,10]. RuteNL copy: **fixed** [1,3,640,640], named outputs `score_8..32`, `bbox_8..32`, `kps_8..32` with a batch dim [1,N,k] | as D1 | 93.80 / 92.02 / 77.13 at VGA — SCRFD README | MIT / non-commercial (RuteNL's HF card says Apache-2.0, but it re-hosts InsightFace weights). No |
| D3 | **SCRFD-10G-KPS** (`det_10g.onnx`, buffalo_l; `scrfd_10g_bnkps.onnx`, antelopev2) | same | https://github.com/yakhyo/uniface/releases/download/weights/scrfd_10g_kps.onnx **OK** (same size as det_10g from yakhyo/face-reidentification and as RuteNL `10g_bnkps.onnx` **OK**) | 16.13 MB | 4.23 M / 10 GF | `input.1` [1,3,?,?] dynamic | as D1 | 95.40 / 94.01 / 82.80 at VGA — SCRFD README | MIT / non-commercial. No |
| D4 | SCRFD-500M (no keypoints) | same | https://huggingface.co/RuteNL/SCRFD-face-detection-ONNX/resolve/main/500m.onnx **OK** | 2.18 MB | 0.57 M / 0.5 GF | fixed [1,3,640,640] | `score_*`, `bbox_*` only (no kps, so it cannot feed 5-point alignment) | 90.57 / 88.12 / 68.51 at VGA; 90.91 / 89.49 / 82.03 at original size — SCRFD README | non-commercial. No |
| D5 | SCRFD-1G (no keypoints) | same | https://huggingface.co/RuteNL/SCRFD-face-detection-ONNX/resolve/main/1g.onnx **OK** | 2.42 MB | 0.64 M / 1 GF | fixed [1,3,640,640] | `score_*`, `bbox_*` | 92.38 / 90.57 / 74.80 at VGA — SCRFD README | non-commercial. No |
| D6 | SCRFD-34G / 34G-GNKPS | same | `.../RuteNL/SCRFD-face-detection-ONNX/resolve/main/34g.onnx` and `34g_gnkps.onnx`: listed by the HF API at 37.35 / 37.6 MB, **not curl-verified** | ~37 MB | 9.80 M / 34 GF | not read | as D1 | 96.06 / 94.92 / 85.29 at VGA (34G) | non-commercial. No |
| D7 | **YuNet-s** (`face_detection_yunet_2023mar.onnx`) | https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet | https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx **OK** | 0.22 MB (232,589 B) | not confirmed for -s | `input` [1,3,640,640] in the file (OpenCV DNN reshapes it). **BGR, raw 0-255, no mean/std** (cv2 `blobFromImage` defaults); pad H,W up to a multiple of 32 | 12 tensors: `cls_8/16/32` [1,HW/s²,1], `obj_*` [..,1], `bbox_*` [..,4], `kps_*` [..,10] | 0.8844 / 0.8656 / 0.7503 with OpenCV Zoo tools/eval ([README](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/README.md)). The same README's intro gives 0.834 / 0.824 / 0.708. Test scale not stated | MIT (model dir LICENSE). **Yes** |
| D7b | YuNet-s, dynamic-shape re-export (`2026may`) | same | https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2026may.onnx **OK** | 0.21 MB | — | symbolic H,W (README: for ONNX Runtime / OpenCV 5) | same | same weights as 2023mar per README | MIT. Yes |
| D7c | YuNet-s int8 | same | https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar_int8.onnx **OK** | 0.09 MB | — | — | same | 0.8810 / 0.8629 / 0.7503 (quant) — README | MIT. Yes |
| D8 | **YuNet-n** (larger, from libfacedetection.train) | https://github.com/ShiqiYu/libfacedetection.train | https://raw.githubusercontent.com/ShiqiYu/libfacedetection.train/master/onnx/yunet_n_640_640.onnx **OK**; dynamic: https://raw.githubusercontent.com/ShiqiYu/libfacedetection.train/master/onnx/yunet_n_dynamic.onnx **OK**; also `yunet_s_dynamic.onnx` **OK** 0.22 MB | 0.30 MB | 75,856 params (paper, [Springer](https://link.springer.com/article/10.1007/s11633-023-1423-y)) | same as YuNet-s (I/O not read; same output names expected, not confirmed) | same | **0.892 / 0.883 / 0.811**, released `yunet_n.pth`, eval command uses `--mode origin` (original image size) — libfacedetection.train README | BSD-3-Clause (repo). Yes |
| D9 | **RetinaFace MobileNet-0.25** (yakhyo re-train) | https://github.com/yakhyo/retinaface-pytorch | https://github.com/yakhyo/uniface/releases/download/weights/retinaface_mv1_0.25.onnx **OK** (also `.../retinaface-pytorch/releases/download/v0.0.1/retinaface_mv1_0.25.onnx`) | 1.65 MB | ~0.44 M (InsightFace RetinaFace-mnet025, per SCRFD table) | `input` [B,3,H,W] fully dynamic. **BGR minus (104,117,123)**, no scaling | `loc` [B,P,4], `conf` [B,P,2] (softmax already applied), `landmarks` [B,P,10] | README tables: multi-scale 88.48 / 87.02 / 80.61; original size 90.70 / 88.12 / 73.82; "<16 px filtered" re-train original size 91.68 / 89.69 / 61.49. Which table matches the current release file is **not confirmed** (README says the 2024-11-27 weights filter faces under 16 px) | MIT. Weights trained on WIDER FACE (WIDER is CC BY-NC-ND per HF card). Commercial use unclear |
| D9b | RetinaFace MobileNet-0.5 / MobileNetV2 | same | `.../uniface/releases/download/weights/retinaface_mv1_0.50.onnx` **OK** 6.33 MB; `.../retinaface_mv2.onnx` **OK** 11.92 MB; also mv1 15.86 MB, r18 45.84 MB, r34 84.40 MB, r50 104.38 MB **OK** | — | — | same | same | mv2: multi-scale 91.70 / 91.03 / 86.60; original 94.04 / 92.26 / 83.59 | MIT |
| D10 | **YOLOv5n-face** (deepcam-cn) | https://github.com/deepcam-cn/yolov5-face | ONNX export: https://github.com/yakhyo/yolov5-face-onnx-inference/releases/download/weights/yolov5n_face.onnx **OK**. Also `yolov5s_face.onnx` **OK** 27.80 MB, `yolov5m_face.onnx` **OK** 81.27 MB. Original .pt: GDrive/Baidu only | 10.51 MB | 1.726 M / 2.111 GF | `input` [B,3,**640,640 fixed**]. Letterbox with 114 pad, RGB, /255 | `output` [B,25200,16]: decoded cx,cy,w,h (pixels in 640 space), obj, 10 landmark coords, cls | 93.61 / 91.52 / 80.53 (yolov5n); yolov5n-0.5 (0.447 M) 90.76 / 88.12 / 73.82. Single scale, "max side 640" — yolov5-face README | GPL-3.0 code and weights. Copyleft |
| D11 | **YOLOv8-lite-s / YOLOv8n-face** (derronqi) | https://github.com/derronqi/yolov8-face | https://github.com/yakhyo/yolov8-face-onnx-inference/releases/download/weights/yolov8-lite-s.onnx **OK** 7.40 MB; https://github.com/yakhyo/yolov8-face-onnx-inference/releases/download/weights/yolov8n-face.onnx **OK** 11.79 MB | 7.40 / 11.79 MB | not published | `images` [B,3,H,W] (uniface only supports 640). Letterbox 114, RGB, /255 | 3 raw heads [1,80,H/s,W/s] for s=8/16/32: 64 DFL + 1 class logit + 15 kps (x,y,vis) | lite-s 93.4 / 91.2 / 78.6; n 94.6 / 92.3 / 79.6 at test size 640 — derronqi README | GPL-3.0. Copyleft |
| D12 | **YOLOv8n-face (lindevs)** — already in hand | https://github.com/lindevs/yolov8-face | https://github.com/lindevs/yolov8-face/releases/download/1.0.1/yolov8n-face-lindevs.onnx **OK** | 11.68 MB | 3.006 M / 8.1 GF | `images` [1,3,640,640] fixed. Ultralytics: letterbox, RGB, /255 | `output0` [1,5,8400] = cx,cy,w,h,score (no keypoints, no NMS) | 93.79 / 91.82 / 79.38 (protocol not stated) — lindevs README | Repo MIT, but trained with Ultralytics from YOLOv8 initial weights. Treat as **AGPL-3.0** |
| D13 | **akanametov YOLOv8n-face** | https://github.com/akanametov/yolo-face | https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.onnx (12,715,424 B; I/O read OK) | 12.13 MB | not published | `images` [1,3,640,640] | `output0` [1,**300,21**]: NMS is inside the graph (x1,y1,x2,y2,score,cls + 5x(x,y,v) kps; layout inferred from 21 = 6 + 15, not confirmed) | **not published** (README shows only PR curves) | GPL-3.0 repo (Ultralytics fork). Copyleft |
| D14 | akanametov YOLOv10n-face | same | https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov10n-face.onnx (I/O read OK) | 8.89 MB | n/p | [1,3,640,640] | [1,300,6] end-to-end (x1,y1,x2,y2,score,cls). No keypoints | not published | GPL-3.0 |
| D15 | akanametov YOLOv11n-face | same | https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov11n-face.onnx (I/O read OK) | 10.09 MB | n/p | [1,3,640,640] | [1,5,8400] raw (needs NMS). No keypoints | not published | GPL-3.0 |
| D16 | akanametov YOLOv12n / 12s-face | same | `.../1.0.0/yolov12n-face.onnx` 10.12 MB; `.../1.0.0/yolov12s-face.onnx` 35.58 MB (I/O read OK). yolov11s and yolov8s are **.pt only**. yolo26n/s/m-face are .pt only (yolo26n-face.pt 5.64 MB) | 10.12 MB | n/p | [1,3,640,640] | [1,300,6] with NMS inside | not published | GPL-3.0 |
| D17 | **CenterFace** | https://github.com/Star-Clouds/CenterFace | dynamic re-export: https://github.com/yakhyo/uniface/releases/download/weights/centerface.onnx **OK** 6.97 MB. Original: https://raw.githubusercontent.com/Star-Clouds/CenterFace/master/models/onnx/centerface.onnx **OK** 7.18 MB | 6.97 MB | not confirmed | uniface: `input` [B,3,H,W] dynamic. **RGB, raw 0-255, no normalisation**; H,W rounded up to a multiple of 32 | `heatmap` [B,1,H/4,W/4], `scale` [B,2,..], `offset` [B,2,..], `landmarks` [B,10,..] | Single inference on original image (SIO): 92.2 / 91.1 / 78.2. Multi-scale + flip: 93.5 / 92.4 / 87.5 — CenterFace README | MIT. Weights trained on WIDER |
| D18 | BlazeFace short-range (MediaPipe) | https://github.com/yakhyo/mediapipe-face-mesh-onnx | https://github.com/yakhyo/uniface/releases/download/weights/face_detection_short_range.onnx **OK** | 0.44 MB | n/p | `input` [B,3,128,128]. RGB, (x-127.5)/127.5, letterboxed | `regressors` [B,896,16], `scores` [B,896,1] | not published. uniface notes it is tuned for faces within about 2 m: **unsuitable for CCTV** | Apache-2.0 (MediaPipe), per uniface. Not re-checked |
| D19 | EResFD-1x | https://github.com/clovaai/EResFD | PyTorch only, split checkpoint in repo `weights/eresfd_16.pth.{base,conf,loc,act_conf,act_loc}.pth` (base **OK**, 0.50 MB). No ONNX: needs repo model code plus `torch.onnx.export` | ~0.5 MB | n/p | — | SSD-style | 0.890 / 0.879 / 0.804 (480x640 latency table) — README | Apache-2.0 |
| D20 | LFFD v1 / v2 | https://github.com/YonghaoHe/LFFD-A-Light-and-Fast-Face-Detector-for-Edge-Devices | MXNet `.params` and Caffe `.caffemodel` in repo (`face_detection/saved_model/...`, 6-9 MB). No ONNX found | — | — | — | — | v1 0.910 / 0.881 / 0.780 (val) — README | MIT |
| D21 | DamoFD (0.5G / 2.5G / 10G / 34G) | https://github.com/ly19965/EasyFace/tree/master/face_project/face_detection/DamoFD | ModelScope pipeline only. **No ONNX or direct file found** | — | — | — | — | numbers not confirmed | not confirmed |
| D22 | YOLOv7-face (lite-t/lite-s/tiny/s) | https://github.com/derronqi/yolov7-face | Google Drive / Baidu `.pt` only (**not curl-verifiable**) | — | 0.8 / 3.0 / 13.2 / 16.8 GF | — | — | lite-t 88.7 / 85.2 / 71.5; lite-s 92.7 / 89.9 / 78.5; tiny 94.7 / 92.6 / 82.1 at 640 — README | GPL-3.0 |

Note: libfacedetection (C++) is YuNet, so see D7 and D8.

---

## 2. Recognisers (all expect 112x112 aligned crops using the ArcFace 5-point template)

| # | Model | Repo / page | Direct weights URL (verified) | Size | Params / FLOPs | Input (I/O verified) + preprocessing | Emb. | Published accuracy (source) | Licence (code / weights), commercial? |
|---|---|---|---|---|---|---|---|---|---|
| R1 | **EdgeFace-XXS** | https://github.com/otroshi/edgeface (Idiap) | ONNX: https://github.com/yakhyo/edgeface-onnx/releases/download/weights/edgeface_xxs.onnx **OK**. Official .pt: https://huggingface.co/Idiap/EdgeFace-XXS/resolve/main/edgeface_xxs.pt **OK** 4.79 MB | 4.89 MB | 1.24 M / 94.72 MF | `input` [B,3,112,112]. **RGB, (x-127.5)/127.5** | 512 | LFW 99.57, CALFW 94.83, CPLFW 90.27, CFP-FP 93.63, AgeDB-30 94.92 — [EdgeFace README](https://github.com/otroshi/edgeface). IJB-C: not in README | Code BSD-3 / **HF weights CC BY-NC-SA 4.0**. No |
| R2 | **EdgeFace-XS (γ=0.6)** | same | https://github.com/yakhyo/edgeface-onnx/releases/download/weights/edgeface_xs_gamma_06.onnx **OK** | 6.94 MB | 1.77 M / 154 MF | same | 512 | LFW 99.73, CFP-FP 94.71, AgeDB 96.08, CALFW 95.28, CPLFW 91.58 | same. No |
| R3 | **EdgeFace-S (γ=0.5)** | same | https://github.com/yakhyo/edgeface-onnx/releases/download/weights/edgeface_s_gamma_05.onnx **OK** | 14.11 MB | 3.65 M / 306 MF | same | 512 | LFW 99.78, CFP-FP 95.74, AgeDB 97.03, CALFW 95.55, CPLFW 92.48 | same. No |
| R3b | EdgeFace-Base | same | `.../edgeface-onnx/releases/download/weights/edgeface_base.onnx` **OK** | 69.69 MB | 18.23 M / 1399 MF | same | 512 | LFW 99.83, CFP-FP 97.01, AgeDB 97.60 | same. No |
| R4 | **AdaFace IR-18 (WebFace4M)** | https://github.com/mk-minchul/AdaFace | ONNX: https://github.com/yakhyo/adaface-onnx/releases/download/weights/adaface_ir_18.onnx **OK** | 91.65 MB | not stated | `input` [B,3,112,112]. **BGR** (AdaFace README stresses this), (x-127.5)/127.5 | 512 | LFW 99.53, CFP-FP 97.26, AgeDB 96.47, CALFW 95.52, CPLFW 92.28; **IJB-C 1e-4: 94.99**, IJB-B 93.03 — AdaFace README. TinyFace: not confirmed | Code MIT. Weights: no separate licence; trained on WebFace4M (research-only data). Not confirmed for commercial |
| R5 | AdaFace IR-50 (WebFace4M) | same | `.../adaface-onnx/releases/download/weights/adaface_ir_50.onnx` **OK** | 166.31 MB | — | same (BGR) | 512 | LFW 99.78, CFP-FP 98.97, AgeDB 97.78; IJB-C 96.98 | as above |
| R6 | **SFace** (`face_recognition_sface_2021dec.onnx`) | https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface | https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx **OK**; int8bq `..._int8bq.onnx` **OK** 10.17 MB | 36.90 MB | MobileFaceNet-type (per README) | `data` [1,3,112,112] fixed batch 1 (old MXNet-style export: weights also appear as graph inputs). `cv2.FaceRecognizerSF`: **RGB (swapRB=true), raw 0-255, no mean/std**; crop with `alignCrop` from YuNet landmarks | **128** (`fc1`) | Accuracy 0.9940 (OpenCV Zoo tools/eval, i.e. LFW); int8bq 0.9942. CFP / AgeDB / IJB-C: not confirmed | **Apache-2.0** (model dir LICENSE). Training data not confirmed. Licence permits commercial use |
| R7 | InsightFace **w600k_mbf** (buffalo_s / sc) — in hand | https://github.com/deepinsight/insightface/tree/master/model_zoo | https://github.com/yakhyo/uniface/releases/download/weights/w600k_mbf.onnx **OK** (also inside buffalo_sc.zip, 14.27 MB **OK**) | 12.98 MB | MBF | `input.1` [None,3,112,112] (output declared [1,512]). RGB, (x-127.5)/127.5 | 512 | LFW 99.70, CFP-FP 98.00, AgeDB 96.58, IJB-C 95.02, MR-ALL 71.87 — model_zoo README | MIT / non-commercial. No |
| R8 | InsightFace **w600k_r50** (buffalo_l / m) — in hand | same | https://github.com/yakhyo/uniface/releases/download/weights/w600k_r50.onnx **OK** 166.30 MB | 166.30 MB | R50 | same | 512 | LFW 99.83, CFP-FP 99.33, AgeDB 98.23, IJB-C 97.25 | non-commercial |
| R9 | antelopev2 **glintr100** | same | https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip **OK** 343.95 MB (contains glintr100.onnx and scrfd_10g_bnkps.onnx; listing read from the zip's central directory) | ~249 MB | R100 | same | 512 | Glint360K R100: LFW 99.817, CFP-FP 99.143, AgeDB 98.450, IJB-C 97.13 (model_zoo table, `Glint360k` row) | non-commercial |
| R10 | yakhyo **MobileFace** (MobileNetV1-0.25 / V2 / V3-S / V3-L, MS1MV2) | https://github.com/yakhyo/face-recognition | `.../uniface/releases/download/weights/mobilenetv1_0.25.onnx` **OK** 1.38 MB; `mobilenetv2.onnx` **OK** 8.66 MB; `mobilenetv3_small.onnx` **OK** 4.78 MB; `mobilenetv3_large.onnx` **OK** 13.39 MB (same files at `.../face-recognition/releases/download/v0.0.1/*_mcp.onnx`) | 1.4-13.4 MB | 0.36 / 2.29 / 1.25 / 3.52 M | `input` [1,3,112,112] **fixed batch 1**. RGB, (x-127.5)/127.5 (uniface default) | 512 | LFW / CALFW / CPLFW / AgeDB: MNv1-0.25 98.76 / 92.02 / 82.37 / 90.02; MNv2 99.55 / 94.87 / 86.89 / 95.16; MNv3-S 99.30 / 93.77 / 85.29 / 92.79; MNv3-L 99.53 / 94.56 / 86.79 / 95.13. CFP-FP and IJB-C: not published | Code MIT; trained on MS1MV2 (research-only data) |
| R11 | yakhyo Sphere20 / Sphere36 | same | `.../uniface/releases/download/weights/sphere20.onnx` **OK** 93.47 MB; `sphere36.onnx` **OK** 132.03 MB | big | 24.5 / 34.6 M | same | 512 | Sphere20 LFW 99.67, AgeDB 96.58 | MIT / MS1MV2 |
| R12 | **AuraFace v1** (R100, fal) | https://huggingface.co/fal/AuraFace-v1 | https://github.com/yakhyo/auraface-onnx/releases/download/weights/auraface_r100.onnx **OK** (HF also has `glintr100.onnx` 248.62 MB) | 248.62 MB | R100 | `input` [B,3,112,112], (x-127.5)/127.5 (uniface default RGB) | 512 | LFW 99.65, CALFW 94.70, CPLFW 90.93 — HF card. CFP / AgeDB / IJB-C: not confirmed | **Apache-2.0, card says trained for commercial use**. The SCRFD file in the same HF repo is still InsightFace's. Only large commercially-licensed recogniser found |
| R13 | HyperFace IR-50 (Idiap, synthetic data) | https://github.com/yakhyo/hyperface-onnx | `.../hyperface-onnx/releases/download/weights/hyperface_ir_50.onnx` **OK** | 166.30 MB | IR-50 | [B,3,112,112] | output dim symbolic (L2-normalised in-graph, inferred from `Div` name) | not confirmed | not confirmed |
| R14 | GhostFaceNetV1-1.3-1 / 1.3-2 (MS1MV3) | https://github.com/HamadYA/GhostFaceNets | Keras `.h5` only: https://github.com/HamadYA/GhostFaceNets/releases/download/v1.2/GhostFaceNet_W1.3_S1_ArcFace.h5 **OK** 16.47 MB; `.../v1.3/GhostFaceNet_W1.3_S2_ArcFace.h5` **OK** 16.37 MB; `.../v1.0/GN_W0.5_S2_ArcFace_epoch16.h5` (3.66 MB per API) | 16.4 MB | not confirmed | no ONNX: needs TF/Keras plus `tf2onnx` (custom layers may need the repo's code). Normalisation **not confirmed** | 512 (not confirmed) | S1: LFW 99.73, CFP-FP 96.83, AgeDB 98.00, IJB-B 93.12, **IJB-C 94.94**. S2: 99.68 / 93.31 / 96.92, IJB-C 93.45 — README | Code MIT; MS1MV3 research-only data |
| R15 | arcface_torch R18 / R34 (MS1MV3, Glint360K) ONNX | https://github.com/deepinsight/insightface/tree/master/model_zoo | Google Drive links only (e.g. R18 Glint360K `drive.google.com/file/d/1Z0eoO1Wqv32K8TdFHKqrlrxv46_W4390`). **Not curl-verifiable** | — | R18 ≈ 2.6 GF | RGB, (x-127.5)/127.5 | 512 | MR-ALL only in model_zoo: R18-MS1MV3 68.33, R34-MS1MV3 77.37, R18-Glint 72.07, R34-Glint 83.02. R18 on WF42M-PFC0.3 IJB-C 95.77 (arcface_torch README; no public download link seen) | non-commercial |

### Recognisers checked and dropped (no directly verifiable weights, or licence blocks use)
| Model | Status |
|---|---|
| MagFace | weights on Google Drive / Baidu only (PyTorch). Code Apache-2.0 |
| ElasticFace | Google Drive only. **CC BY-NC-SA 4.0** |
| PocketNet | GDrive / in repo (not verified). **CC BY-NC-SA 4.0** |
| MixFaceNets (M/S/XS) | Dropbox folder only. MixFaceNet-XS: 1.04 M, 161.9 MF, LFW 99.60, AgeDB 95.85, IJB-C 90.73 at FAR 1e-6 (README) |
| TopoFR | Google Drive / Baidu only. No licence file |
| VarGFaceNet | code MIT. No direct weights found |
| SwiftFaceFormer, ShuffleFaceNet | no official downloadable weights found (not confirmed) |

---

## 3. Labelled test data

| Dataset | Direct URL | Status / size | Notes |
|---|---|---|---|
| WIDER FACE val images | https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data/WIDER_val.zip | **OK** 345.94 MB | HF dataset is not gated. Licence CC BY-NC-ND 4.0 per HF card |
| WIDER FACE annotations | https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data/wider_face_split.zip | **OK** 3.42 MB | |
| WIDER official eval tools (contains GT .mat) | http://shuoyang1213.me/WIDERFACE/support/eval_script/eval_tools.zip | **OK** 8.05 MB | official MATLAB eval |
| `wider_easy_val.mat` | https://raw.githubusercontent.com/ShiqiYu/libfacedetection.train/master/data/widerface/labelv2/val/gt/wider_easy_val.mat | **OK** 0.39 MB | raw single files |
| `wider_medium_val.mat` | .../gt/wider_medium_val.mat | **OK** 0.39 MB | |
| `wider_hard_val.mat` | .../gt/wider_hard_val.mat | **OK** 0.40 MB | |
| `wider_face_val.mat` | .../gt/wider_face_val.mat | **OK** 0.37 MB | |
| `lfw.bin` (112x112, InsightFace pickle) | https://huggingface.co/datasets/gaunernst/face-recognition-eval/resolve/main/lfw.bin | **OK** 62.06 MB | Mirrors: `datasets/Icar/val_sets/.../lfw.bin` **OK** 61.73 MB; https://github.com/leondgarse/Keras_insightface/releases/download/v1.0.0/lfw.bin **OK** 61.52 MB. Pickle file: load only with numpy/pickle in an isolated process |
| `cfp_fp.bin` | https://huggingface.co/datasets/gaunernst/face-recognition-eval/resolve/main/cfp_fp.bin | **OK** 72.91 MB | Icar mirror **OK** 72.73 MB |
| `agedb_30.bin` | https://huggingface.co/datasets/gaunernst/face-recognition-eval/resolve/main/agedb_30.bin | **OK** 72.98 MB | Icar mirror **OK**. The same gaunernst repo also has calfw.bin, cplfw.bin, cfp_ff.bin, talfw.bin (sizes per HF API) |
| **XQLFW aligned 112** | https://github.com/Martlgap/xqlfw/releases/download/1.0/xqlfw_aligned_112.zip | **OK** 55.18 MB | **Direct, no registration** (official page https://martlgap.github.io/xqlfw/pages/download.html). Pairs: https://github.com/Martlgap/xqlfw/releases/download/1.0/xqlfw_pairs.txt **OK** 0.15 MB. Original images: `.../xqlfw.zip` **OK** 186.18 MB. Quality scores: `.../xqlfw_scores.txt` **OK**. Page states no licence ("All Rights Reserved"). There is also a 3-shard webdataset copy in `LSIbabnikz/xqlfw_bin` (format not confirmed) |
| TinyFace (native LR, about 20x16 px, 5,139 IDs, 1:N) | Google Drive `drive.google.com/open?id=1xTZc7lNmWN33ECO2AKH6FycGdiqIK7W0`; Baidu `pan.baidu.com/s/1QKzSZCTAQEkzE29fzVBCGg` | 148 MB (per https://qmul-tinyface.github.io). **Not curl-verifiable** (GDrive) | page states no registration |
| QMUL-SurvFace (463,507 imgs, 15,573 IDs) | Google Drive `drive.google.com/open?id=13ch6BPaexlKt8gXB_I8aX7p1G3yPm2Bl`; Baidu `pan.baidu.com/s/1O55042SMxLYqBWdGGQ_4_g` | 389 MB (per https://qmul-survface.github.io). Not curl-verifiable | no sign-up; "research purposes" |
| SCface (5 surveillance cameras, 130 subjects, 3 distances) | none | — | **Registration: signed release agreement plus institutional cover letter emailed to FER Zagreb** (https://www.scface.org/). Students cannot sign |
| DroneSURF | — | not checked | request form (not confirmed) |

---

## 4. Notes on decoding, by family

**SCRFD (InsightFace `scrfd.py` convention; applies to D1-D6)**
- Preprocessing: RGB, `(x - 127.5) / 128`, NCHW float32. InsightFace resizes to fit 640x640 keeping aspect ratio, pastes at the top-left, pads with zeros, and divides boxes by the scale afterwards.
- Output order: `[score_s8, score_s16, score_s32, bbox_s8, bbox_s16, bbox_s32, kps_s8, kps_s16, kps_s32]`. In buffalo files the outputs are numeric names with no batch dim. In RuteNL files they are named (`score_8`, ...) with a batch dim.
- **2 anchors per location**: 12800 = 80·80·2 at 640. Anchor centres are `(x*stride, y*stride)` for each grid cell, each repeated twice.
- Scores already have sigmoid applied.
- bbox output is **distance (l,t,r,b) multiplied by stride**: `x1 = cx - l*s, y1 = cy - t*s, x2 = cx + r*s, y2 = cy + b*s`.
- kps: `cx + k*s`, `cy + k*s` for 5 points.
- Then apply NMS (0.4).
- The InsightFace-hosted 500m/10g files report fixed output N for 640 even though input H,W are dynamic. InsightFace's own code runs them at other sizes, but test this in TensorRT.

**YuNet (D7, D8)**
- The intended decoder is `cv2.FaceDetectorYN` (`setInputSize` = padded size).
- Manual decode:
  - Input is BGR 0-255, padded to a multiple of 32.
  - For stride s in (8, 16, 32), each cell (r,c) has 1 prior: `score = sqrt(clip(cls,0,1)*clip(obj,0,1))`.
  - Box centre: `cx = (c + bbox0)*s`, `cy = (r + bbox1)*s`.
  - Box size: `w = exp(bbox2)*s`, `h = exp(bbox3)*s`.
  - Landmarks: `kps_x = (c + k)*s`, `kps_y = (r + k)*s`.
  - Then NMS (source: opencv `modules/objdetect/src/face_detect.cpp`).
- Output order is `cls_*, obj_*, bbox_*, kps_*`.
- Use `2026may` or `yunet_*_dynamic.onnx` with ONNX Runtime for non-640 input.

**RetinaFace (yakhyo, D9)**
- Input: BGR minus (104,117,123), dynamic H,W.
- Priors: steps [8,16,32], min_sizes [[16,32],[64,128],[256,512]], so 2 priors per cell, in normalised (cx,cy,w,h).
- Box decode with variances [0.1,0.2]: `cxcy = p_xy + loc_xy*0.1*p_wh`, `wh = p_wh*exp(loc_wh*0.2)`.
- Landmarks: `p_xy + lm*0.1*p_wh`.
- Multiply by input W,H.
- `conf[...,1]` is the face probability (softmax is already in the graph).

**CenterFace (D17)**
- Input: RGB raw 0-255, H,W multiple of 32.
- Output stride 4. For heatmap peaks over the threshold at (r,c):
  - `s0 = exp(scale[0])*4` (h), `s1 = exp(scale[1])*4` (w).
  - `x1 = (c + offset[1] + 0.5)*4 - s1/2`, `y1 = (r + offset[0] + 0.5)*4 - s0/2`.
  - Landmark j: `x = lm[2j+1]*s1 + x1`, `y = lm[2j]*s0 + y1`.
- Then NMS. Source: `Star-Clouds/CenterFace/prj-python/centerface.py`, matched by uniface.

**YOLOv5-face (D10)**
- Input: letterbox 640 (pad 114, centred), RGB/255.
- Output [B,25200,16] is already decoded in-graph: cx,cy,w,h in 640 pixel space, col 4 = obj, cols 5-14 = 5 landmarks (x,y), col 15 = cls.
- Score = obj (uniface) or obj*cls (original repo). Then NMS and un-letterbox.

**YOLOv8-face, derronqi/yakhyo export (D11)**
- 3 raw maps [1,80,H,W].
- Per cell: 64 = 4x16 DFL logits. Softmax over 16, take the expectation to get distance d, then `x1 = (gx+0.5-d_l)*s`, etc.
- Channel 64 = class logit (apply sigmoid).
- Channels 65-79 = 5x(x,y,vis): `kx = (px*2 + gx)*s`, with **no +0.5** on the keypoint grid (uniface code).

**Ultralytics-style exports (D12-D16)**
- `[1,5,8400]` (lindevs, akanametov v11n): channels = cx,cy,w,h,score over 8400 anchors (80²+40²+20²). Decoded in pixels, sigmoid applied, **NMS needed**, no keypoints.
- `[1,300,6]` (akanametov v10n/v12n/v12s): NMS is in the graph (`nms=True` export, per README). Rows are x1,y1,x2,y2,score,class. Coordinates are in letterbox 640 space.
- `[1,300,21]` (akanametov v8n): same plus 15 keypoint values (layout not confirmed; inspect before use).
- All of these are fixed 640x640 and need the standard Ultralytics letterbox (114 pad, RGB/255).

**Recognisers**
- All 512-d except SFace (128-d). L2-normalise, then cosine.
- Colour order matters: **AdaFace = BGR**. EdgeFace, InsightFace (w600k / glint), yakhyo MobileFace/Sphere and AuraFace = RGB with (x-127.5)/127.5. SFace = RGB raw 0-255 (OpenCV `FaceRecognizerSF`).
- Fixed batch 1: yakhyo MobileFace and SFace. The rest have a dynamic batch, though w600k_mbf declares output [1,512].
