"""Score face detectors on WIDER FACE validation (3,226 labelled images, 39,708 faces) with the official protocol.

  python eval_det.py <key|all> [--size 640] [--limit N] [--out results.jsonl]

For each model: AP on the easy / medium / hard subsets (IoU 0.5, as the official eval_tools compute it), the same on
dense images only (30 or more labelled faces), the share of faces found when 95% and 90% of the boxes are real faces,
file size, parameter count and GPU time per image (ONNX Runtime, CUDA, batch 1).

Every model sees the same picture: the image scaled so its longer side is --size, padded to a square.
"""
import argparse, json, os, sys, time
import cv2
import numpy as np
import onnx
import onnxruntime as ort

try:
    ort.preload_dlls()                 # CUDA / cuDNN from the pip packages (Windows venv)
except Exception:                      # noqa: BLE001
    pass
ort.set_default_logger_severity(3)

ROOT = os.environ.get("FP_DATA", r"E:\bench\data")
MODELS = os.environ.get("FP_MODELS", r"E:\bench\models")

# key: (file, family, extra)
REG = {
    "yolov8n-face (current)": ("yolov8n-face.onnx", "yolo", {}),
    "scrfd-500m":       ("zoo/det_500m.onnx", "scrfd", {}),
    "scrfd-1g":         ("zoo/scrfd_1g.onnx", "scrfd", {}),
    "scrfd-2.5g":       ("zoo/det_2.5g.onnx", "scrfd", {}),
    "scrfd-10g":        ("zoo/det_10g.onnx", "scrfd", {}),
    "scrfd-34g":        ("zoo/scrfd_34g.onnx", "scrfd", {}),
    "yunet-s":          ("zoo/face_detection_yunet_2023mar.onnx", "yunet", {}),
    "yunet-n":          ("zoo/yunet_n_640_640.onnx", "yunet", {}),
    "retinaface-mnet0.25": ("zoo/retinaface_mv1_0.25.onnx", "retina", {}),
    "retinaface-mnetv2":   ("zoo/retinaface_mv2.onnx", "retina", {}),
    "yolov5n-face":     ("zoo/yolov5n_face.onnx", "yolo", {"v5face": True}),
    "yolov5s-face":     ("zoo/yolov5s_face.onnx", "yolo", {"v5face": True}),
    "yolov8n-face (akanametov)":  ("zoo/aka_yolov8n-face.onnx", "yolo", {"end2end": True}),
    "yolov10n-face":    ("zoo/aka_yolov10n-face.onnx", "yolo", {"end2end": True}),
    "yolov11n-face":    ("zoo/aka_yolov11n-face.onnx", "yolo", {}),
    "yolov12n-face":    ("zoo/aka_yolov12n-face.onnx", "yolo", {"end2end": True}),
    "centerface":       ("zoo/centerface.onnx", "center", {}),
}
if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "det_models.json")):
    for k, v in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "det_models.json"))).items():
        REG[k] = (v["file"], v["family"], v.get("extra", {}))


# ----------------------------------------------------------------------------------------------- pre-processing
def letterbox(img, S, pad, center):
    h, w = img.shape[:2]
    s = min(S / w, S / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((S, S, 3), pad, np.uint8)
    ox, oy = ((S - nw) // 2, (S - nh) // 2) if center else (0, 0)
    out[oy:oy + nh, ox:ox + nw] = r
    return out, s, ox, oy


def nms(b, s, thr):
    if len(b) == 0:
        return np.zeros((0,), int)
    x1, y1, x2, y2 = b.T
    area = (x2 - x1) * (y2 - y1)
    order = s.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (area[i] + area[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= thr]
    return np.array(keep, int)


# ----------------------------------------------------------------------------------------------- model families
class Det:
    conf, nms_thr = 0.02, 0.45

    def __init__(self, path, extra):
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(path, so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0]
        self.extra = extra
        self.outs = [o.name for o in self.sess.get_outputs()]

    def fixed_size(self):
        sh = self.inp.shape
        return sh[2] if isinstance(sh[2], int) and isinstance(sh[3], int) else None

    def run(self, blob):
        return self.sess.run(None, {self.inp.name: blob})


class Yolo(Det):
    """Ultralytics-style face models: RGB / 255, grey letterbox; output [1, 4+1(+kps), N] (cx, cy, w, h, score ...)."""
    pad, center = 114, True

    def blob(self, lb):
        return np.ascontiguousarray(lb[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0

    def decode(self, outs, S):
        o = outs[0][0]
        if o.shape[0] < o.shape[1]:
            o = o.T                                   # -> [N, C]
        if self.extra.get("end2end"):                 # [N, 6+]: x1, y1, x2, y2, score, class (NMS inside the graph)
            m = o[:, 4] > 0
            return o[m, :4], o[m, 4]
        if self.extra.get("v5face"):                  # [N, 16]: cx, cy, w, h, objectness, 10 landmark values, class
            o = o.copy()
            o[:, 4] = o[:, 4] * o[:, 15]
        sc = o[:, 4]
        m = sc > self.conf
        o, sc = o[m], sc[m]
        b = np.stack([o[:, 0] - o[:, 2] / 2, o[:, 1] - o[:, 3] / 2, o[:, 0] + o[:, 2] / 2, o[:, 1] + o[:, 3] / 2], 1)
        return b, sc


class Scrfd(Det):
    """InsightFace SCRFD: RGB, (x - 127.5) / 128, zero padding bottom/right; per stride: score, box distances x stride."""
    pad, center = 0, False
    nms_thr = 0.4

    def blob(self, lb):
        return ((np.ascontiguousarray(lb[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) - 127.5) / 128.0)

    def decode(self, outs, S):
        n = len(outs)
        fmc = 3 if n in (6, 9) else 5
        strides = [8, 16, 32] if fmc == 3 else [8, 16, 32, 64, 128]
        na = 2 if fmc == 3 else 1
        bs, ss = [], []
        for i, st in enumerate(strides):
            sc = outs[i].reshape(-1)
            d = outs[i + fmc].reshape(-1, 4) * st
            g = S // st
            ys, xs = np.mgrid[:g, :g]
            c = np.stack([xs, ys], -1).astype(np.float32).reshape(-1, 2) * st
            c = np.repeat(c, na, 0)
            m = sc > self.conf
            c, d, sc = c[m], d[m], sc[m]
            bs.append(np.stack([c[:, 0] - d[:, 0], c[:, 1] - d[:, 1], c[:, 0] + d[:, 2], c[:, 1] + d[:, 3]], 1))
            ss.append(sc)
        return np.concatenate(bs), np.concatenate(ss)


class Yunet(Det):
    """OpenCV Zoo YuNet 2023mar: BGR 0..255, no normalisation; score = sqrt(cls * obj); boxes relative to the cell."""
    pad, center = 0, False
    nms_thr = 0.3

    def blob(self, lb):
        return np.ascontiguousarray(lb.transpose(2, 0, 1)[None], dtype=np.float32)

    def decode(self, outs, S):
        o = dict(zip(self.outs, outs))
        bs, ss = [], []
        for st in (8, 16, 32):
            cls = o[f"cls_{st}"].reshape(-1)
            obj = o[f"obj_{st}"].reshape(-1)
            bb = o[f"bbox_{st}"].reshape(-1, 4)
            g = S // st
            ys, xs = np.mgrid[:g, :g]
            sc = np.sqrt(np.clip(cls, 0, 1) * np.clip(obj, 0, 1))
            m = sc > self.conf
            cx = (xs.reshape(-1)[m] + bb[m, 0]) * st
            cy = (ys.reshape(-1)[m] + bb[m, 1]) * st
            w = np.exp(bb[m, 2]) * st
            h = np.exp(bb[m, 3]) * st
            bs.append(np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1))
            ss.append(sc[m])
        return np.concatenate(bs), np.concatenate(ss)


class Retina(Det):
    """RetinaFace (biubug6): BGR minus (104, 117, 123); SSD priors, variances 0.1 / 0.2; outputs loc, conf, landms."""
    pad, center = 0, False
    nms_thr = 0.4

    def blob(self, lb):
        return np.ascontiguousarray((lb.astype(np.float32) - np.array([104, 117, 123], np.float32)).transpose(2, 0, 1)[None])

    def priors(self, S):
        out = []
        for st, sizes in zip((8, 16, 32), ((16, 32), (64, 128), (256, 512))):
            g = int(np.ceil(S / st))
            for i in range(g):
                for j in range(g):
                    for ms in sizes:
                        out.append([(j + 0.5) * st / S, (i + 0.5) * st / S, ms / S, ms / S])
        return np.array(out, np.float32)

    def decode(self, outs, S):
        by = {o.shape[-1]: o.reshape(-1, o.shape[-1]) for o in outs}
        loc, conf = by[4], by[2]
        if not hasattr(self, "_p") or len(self._p) != len(loc):
            self._p = self.priors(S)
        p = self._p
        sc = conf[:, 1]
        if sc.max() > 1.0 or sc.min() < 0.0:          # logits
            e = np.exp(conf - conf.max(1, keepdims=True))
            sc = (e / e.sum(1, keepdims=True))[:, 1]
        m = sc > self.conf
        loc, p, sc = loc[m], p[m], sc[m]
        cxy = p[:, :2] + loc[:, :2] * 0.1 * p[:, 2:]
        wh = p[:, 2:] * np.exp(loc[:, 2:] * 0.2)
        b = np.concatenate([cxy - wh / 2, cxy + wh / 2], 1) * S
        return b, sc


class Center(Det):
    """CenterFace: RGB 0..255, no normalisation; heat-map at stride 4, log sizes, sub-cell offsets."""
    pad, center = 0, False
    nms_thr = 0.3

    def blob(self, lb):
        return np.ascontiguousarray(lb[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)

    def decode(self, outs, S):
        o = dict(zip(self.outs, outs))
        names = list(o)
        hm = o.get("heatmap", outs[0])[0, 0]
        sc_ = o.get("scale", outs[1])[0]
        of = o.get("offset", outs[2])[0]
        ys, xs = np.nonzero(hm > max(self.conf, 0.05))
        s = hm[ys, xs]
        h = np.exp(sc_[0, ys, xs]) * 4
        w = np.exp(sc_[1, ys, xs]) * 4
        x1 = np.clip((xs + of[1, ys, xs] + 0.5) * 4 - w / 2, 0, None)
        y1 = np.clip((ys + of[0, ys, xs] + 0.5) * 4 - h / 2, 0, None)
        return np.stack([x1, y1, x1 + w, y1 + h], 1), s


FAM = {"yolo": Yolo, "scrfd": Scrfd, "yunet": Yunet, "retina": Retina, "center": Center}


# ----------------------------------------------------------------------------------------------- WIDER protocol
def iou_matrix(p, g):
    """p [P,4] x1y1x2y2, g [G,4] x1y1x2y2, with the official +1 pixel convention."""
    x1 = np.maximum(p[:, None, 0], g[None, :, 0]); y1 = np.maximum(p[:, None, 1], g[None, :, 1])
    x2 = np.minimum(p[:, None, 2], g[None, :, 2]); y2 = np.minimum(p[:, None, 3], g[None, :, 3])
    inter = np.clip(x2 - x1 + 1, 0, None) * np.clip(y2 - y1 + 1, 0, None)
    ap = (p[:, 2] - p[:, 0] + 1) * (p[:, 3] - p[:, 1] + 1)
    ag = (g[:, 2] - g[:, 0] + 1) * (g[:, 3] - g[:, 1] + 1)
    return inter / (ap[:, None] + ag[None, :] - inter)


def image_pr(pred, score_n, gt, keep, T):
    """Official image_eval + img_pr_info, vectorised. Returns [T, 2]: kept proposals and recalled faces per threshold."""
    P = len(pred)
    out = np.zeros((T, 2))
    if P == 0 or len(gt) == 0:                            # the official script skips such images
        return out
    ov = iou_matrix(pred, gt)
    gi = ov.argmax(1)
    hit = ov[np.arange(P), gi] >= 0.5
    ignore = np.ones(len(gt), bool)
    ignore[keep] = False                                  # True: face outside this subset
    prop = np.ones(P)
    prop[hit & ignore[gi]] = -1                           # matched a face that is not counted: proposal ignored
    first = np.full(len(gt), P + 1)
    valid = hit & ~ignore[gi]
    idx = np.nonzero(valid)[0]
    if len(idx):
        g_u, f = np.unique(gi[idx], return_index=True)
        first[g_u] = idx[f]
    rec_at = np.zeros(P + 2)
    np.add.at(rec_at, first[first <= P], 1)
    pred_recall = np.cumsum(rec_at)[:P]
    cum_prop = np.cumsum(prop == 1)
    thr = 1 - (np.arange(T) + 1) / T
    n_above = (score_n[None, :] >= thr[:, None]).sum(1)   # scores are sorted descending
    has = n_above > 0
    r = n_above[has] - 1
    out[has, 0] = cum_prop[r]
    out[has, 1] = pred_recall[r]
    return out


def voc_ap(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def evaluate(preds, gts, T=1000):
    """preds: per image [n,5] sorted by score descending; gts: list of dicts from wider_gt.json."""
    allmax = max((p[:, 4].max() for p in preds if len(p)), default=1.0)
    allmin = min((p[:, 4].min() for p in preds if len(p)), default=0.0)
    res = {}
    subsets = {"easy": lambda g: True, "medium": lambda g: True, "hard": lambda g: True,
               "dense_hard": lambda g: len(g["boxes"]) >= 30}
    for name, use in subsets.items():
        key = "hard" if name == "dense_hard" else name
        pr = np.zeros((T, 2))
        faces = 0
        for p, g in zip(preds, gts):
            if not use(g):
                continue
            gb = np.array(g["boxes"], float).reshape(-1, 4)
            gb = np.concatenate([gb[:, :2], gb[:, :2] + gb[:, 2:]], 1)
            keep = np.array(g[key], int)
            faces += len(keep)
            sn = (p[:, 4] - allmin) / max(allmax - allmin, 1e-9) if len(p) else np.zeros(0)
            pr += image_pr(p[:, :4], sn, gb, keep, T)
        prec = pr[:, 1] / np.maximum(pr[:, 0], 1)
        rec = pr[:, 1] / max(faces, 1)
        res[name + "_ap"] = round(100 * voc_ap(rec, prec), 2)
        for q in (0.95, 0.90):
            ok = prec >= q
            res[f"{name}_found_at_p{int(q * 100)}"] = round(100 * float(rec[ok].max()) if ok.any() else 0.0, 1)
        res[name + "_faces"] = faces
    return res


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("key")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "det_results.jsonl"))
    a = ap.parse_args()
    gts = json.load(open(os.path.join(ROOT, "wider_gt.json")))
    if a.limit:
        gts = gts[:: max(1, len(gts) // a.limit)][: a.limit]
    keys = list(REG) if a.key == "all" else a.key.split(",")
    for key in keys:
        file, fam, extra = REG[key]
        path = os.path.join(MODELS, file)
        if not os.path.exists(path):
            print(json.dumps({"model": key, "error": "missing " + path}), flush=True)
            continue
        try:
            det = FAM[fam](path, extra)
        except Exception as e:      # noqa: BLE001
            print(json.dumps({"model": key, "error": str(e)[:200]}), flush=True)
            continue
        S = det.fixed_size() or a.size
        params = sum(int(np.prod(t.dims)) for t in onnx.load(path, load_external_data=False).graph.initializer)
        preds, t_inf = [], []
        t0 = time.time()
        for g in gts:
            img = cv2.imread(os.path.join(ROOT, "WIDER_val", "images", g["event"], g["file"] + ".jpg"))
            lb, s, ox, oy = letterbox(img, S, det.pad, det.center)
            blob = det.blob(lb)
            t1 = time.perf_counter()
            outs = det.run(blob)
            t_inf.append(time.perf_counter() - t1)
            b, sc = det.decode(outs, S)
            if len(b):
                k = nms(b, sc, det.nms_thr)[:1000]
                b, sc = b[k], sc[k]
                b = (b - np.array([ox, oy, ox, oy])) / s
                h, w = img.shape[:2]
                b[:, [0, 2]] = b[:, [0, 2]].clip(0, w - 1)
                b[:, [1, 3]] = b[:, [1, 3]].clip(0, h - 1)
                o = sc.argsort()[::-1]
                preds.append(np.concatenate([b[o], sc[o, None]], 1))
            else:
                preds.append(np.zeros((0, 5)))
        r = {"model": key, "family": fam, "input": S, "provider": det.sess.get_providers()[0].replace("ExecutionProvider", ""), "images": len(gts), "file_mb": round(os.path.getsize(path) / 1e6, 2),
             "params_m": round(params / 1e6, 3), "gpu_ms_per_image": round(1000 * float(np.median(t_inf[20:] or t_inf)), 2),
             "total_s": round(time.time() - t0, 1)}
        r.update(evaluate(preds, gts))
        print(json.dumps(r), flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
