"""(Windows venv) Check the C++ pipeline against the ONNX Runtime reference on the same frames.
  1. K1: detector input tensor of frame 0 vs RGB/255 letterbox of the OpenCV-decoded frame
  2. detections: C++ (K1 -> TRT -> K2) vs reference (ORT det_r384 + cv2 NMS), matched at IoU > 0.5
  3. embeddings: C++ (K3 crop from NV12 -> TRT) vs reference (cv2 crop/resize of the same box -> ORT), cosine
usage: compare_dump.py dump.bin [video] [rec_onnx]"""
import os, struct, sys
import cv2, numpy as np, onnxruntime as ort
sys.path.insert(0, r"E:\bench")
import fast_win as fw

dump = sys.argv[1]
video = sys.argv[2] if len(sys.argv) > 2 else r"E:\bench\sub_rot0.mp4"
rec_onnx = sys.argv[3] if len(sys.argv) > 3 else r"E:\bench\models\cpp\rec_edgeface_s.onnx"
ort.preload_dlls(); ort.set_default_logger_severity(3)
det = ort.InferenceSession(r"E:\bench\models\det_r384.onnx", providers=["CUDAExecutionProvider"])
rec = ort.InferenceSession(rec_onnx, providers=["CUDAExecutionProvider"])

frames_cpp = []
with open(dump, "rb") as f:
    while True:
        h = f.read(8)
        if len(h) < 8: break
        k, n = struct.unpack("ii", h)
        boxes = np.frombuffer(f.read(20 * n), np.float32).reshape(n, 5)
        emb = np.frombuffer(f.read(4 * 512 * n), np.float32).reshape(n, 512)
        frames_cpp.append((boxes, emb))
# reference decode with FFmpeg, which honours the stream's colour tags (this clip: BT.709), like the C++ path
import glob, subprocess
ff = sorted(glob.glob(r"E:\bench\ffmpeg\**\ffmpeg.exe", recursive=True))[0]
probe = subprocess.run([ff.replace("ffmpeg.exe", "ffprobe.exe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height", "-of", "csv=p=0", video], capture_output=True, text=True, check=True).stdout
FW, FH = [int(x) for x in probe.strip().split(",")[:2]]
raw = subprocess.run([ff, "-v", "error", "-i", video, "-frames:v", str(len(frames_cpp)), "-pix_fmt", "rgb24",
                      "-f", "rawvideo", "pipe:1"], capture_output=True, check=True).stdout
frames_full = list(np.frombuffer(raw, np.uint8).reshape(-1, FH, FW, 3))
SC = min(640 / FW, 360 / FH)            # frame -> 640x360 detector content (then 12-row letterbox to 384)
frames = [f if (FW, FH) == (640, 360) else cv2.resize(f, (round(FW * SC), round(FH * SC)), interpolation=cv2.INTER_LINEAR)
          for f in frames_full]
print(f"video {FW}x{FH}, detector content scale {SC:.4f}")

def crop_full(img, b):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, int(b[0])), max(0, int(b[1])), min(w, int(b[2])), min(h, int(b[3]))
    if x2 - x1 < 4 or y2 - y1 < 4: return None
    return cv2.resize(img[y1:y2, x1:x2], (112, 112), interpolation=cv2.INTER_LINEAR)

# 1. K1 tensor
t = np.fromfile(dump + ".detin", np.float16).astype(np.float32).reshape(3, 384, 640)
ref = np.full((3, 384, 640), 114 / 255, np.float32)
ref[:, 12:372] = frames[0].astype(np.float32).transpose(2, 0, 1) / 255
d = np.abs(t - ref)
print(f"K1 detector input vs reference: mean |diff| = {d.mean() * 255:.2f} /255, p99 = {np.percentile(d, 99) * 255:.1f} /255 "
      f"(includes decoder + colour-conversion differences)")

# 2. detections
n_cpp = n_ref = matched = 0
ious, cos_all = [], []
for i, (cb, ce) in enumerate(frames_cpp):
    rb = fw.postprocess(det.run(None, {"frames": frames[i][None]})[0][0], 12) / SC   # back to full-frame pixels
    n_cpp += len(cb); n_ref += len(rb)
    if len(cb) and len(rb):
        m = fw.iou_matrix(cb[:, :4], rb)
        best = m.max(1); matched += int((best > 0.5).sum()); ious += best[best > 0.5].tolist()
    # 3. embeddings for the C++ boxes, reference crop path
    if len(cb):
        crops = [crop_full(frames_full[i], b) for b in cb[:, :4]]
        keep = [j for j, c in enumerate(crops) if c is not None]
        if keep:
            e = rec.run(None, {rec.get_inputs()[0].name: np.stack([crops[j] for j in keep]).astype(np.float32).transpose(0, 3, 1, 2) * (1 / 127.5) - 1.0})[0] \
                if rec.get_inputs()[0].type == "tensor(float)" else rec.run(None, {"crops": np.stack([crops[j] for j in keep])})[0]
            a = ce[keep] / np.linalg.norm(ce[keep], axis=1, keepdims=True)
            b = e / np.linalg.norm(e, axis=1, keepdims=True)
            cos_all += (a * b).sum(1).tolist()
print(f"detections over {len(frames_cpp)} frames: C++ {n_cpp}, reference {n_ref}, matched {matched} "
      f"({100 * matched / max(1, n_ref):.1f}% of reference), mean IoU {np.mean(ious):.3f}")
c = np.array(cos_all)
print(f"embeddings (same boxes): cosine mean {c.mean():.4f}, p5 {np.percentile(c, 5):.4f}, min {c.min():.4f}, n={len(c)}")
