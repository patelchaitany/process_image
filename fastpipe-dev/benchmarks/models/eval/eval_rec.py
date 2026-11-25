"""Score face recognisers on labelled same-person / different-person pairs, 112x112 aligned:
LFW, CFP-FP (frontal vs profile), AgeDB-30 (age gap), CPLFW (pose), XQLFW (real low-quality vs good-quality photos).

  python eval_rec.py <key|all> [--sets lfw,cfp_fp,agedb_30,cplfw,xqlfw] [--sizes 56,40,28,20] [--no-flip] [--out results.jsonl]

Accuracy = 10-fold verification accuracy, the standard protocol (threshold picked on 9 folds, scored on the 10th;
embedding of the image plus its mirror image, L2-normalised).

CCTV check: every 112x112 face is shrunk to NxN and scaled back up before it reaches the model.
  both  : both photos of a pair are shrunk (camera against camera)
  cross : only the second photo is shrunk (camera against a good enrolment photo)
A detector box is about 0.7 of the aligned crop, so N = 56 / 40 / 28 / 20 is a face about 39 / 28 / 20 / 14 px wide.
"""
import argparse, json, os, pickle, time
import cv2
import numpy as np
import onnx
import onnxruntime as ort

try:
    ort.preload_dlls()
except Exception:      # noqa: BLE001
    pass
ort.set_default_logger_severity(3)

ROOT = os.environ.get("FP_DATA", r"E:\bench\data")
MODELS = os.environ.get("FP_MODELS", r"E:\bench\models")

# key: (file, pre-processing)   arcface: RGB (x-127.5)/127.5 | bgr: BGR (x-127.5)/127.5 | raw_rgb: RGB 0..255
REG = {
    "arcface-r50-w600k (current)": ("w600k_r50.onnx", "arcface"),
    "mobilefacenet-w600k": ("w600k_mbf.onnx", "arcface"),
    "edgeface-xxs":       ("zoo/edgeface_xxs.onnx", "arcface"),
    "edgeface-xs":        ("zoo/edgeface_xs_gamma_06.onnx", "arcface"),
    "edgeface-s":         ("zoo/edgeface_s_gamma_05.onnx", "arcface"),
    "edgeface-base":      ("zoo/edgeface_base.onnx", "arcface"),
    "adaface-ir18":       ("zoo/adaface_ir_18.onnx", "bgr"),
    "adaface-ir50":       ("zoo/adaface_ir_50.onnx", "bgr"),
    "auraface-r100":      ("zoo/auraface_r100.onnx", "arcface"),
    "sface":              ("zoo/face_recognition_sface_2021dec.onnx", "raw_rgb"),
    "mobilenetv1-0.25":   ("zoo/mobilenetv1_0.25.onnx", "arcface"),
    "mobilenetv3-small":  ("zoo/mobilenetv3_small.onnx", "arcface"),
    "mobilenetv2":        ("zoo/mobilenetv2.onnx", "arcface"),
    "mobilenetv3-large":  ("zoo/mobilenetv3_large.onnx", "arcface"),
}
_extra = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rec_models.json")
if os.path.exists(_extra):
    for k, v in json.load(open(_extra)).items():
        REG[k] = (v["file"], v["prep"])


def _decode(buf):
    im = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)        # BGR
    if im.shape[:2] != (112, 112):
        im = cv2.resize(im, (112, 112), interpolation=cv2.INTER_AREA)
    return im


class SafeUnpickler(pickle.Unpickler):
    """InsightFace .bin sets are pickles. Only plain containers and numpy arrays may be built: no other code can run."""
    OK = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"), ("numpy", "ndarray"),
          ("numpy", "dtype"), ("_codecs", "encode"), ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar")}

    def find_class(self, module, name):
        if (module, name) in self.OK:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refused {module}.{name}")


def load_bin(path):
    with open(path, "rb") as f:
        bins, issame = SafeUnpickler(f, encoding="bytes").load()
    imgs = [_decode(b if isinstance(b, (bytes, bytearray)) else np.asarray(b).tobytes()) for b in bins]
    return np.stack(imgs), np.array(issame, bool)


def load_pairs(name):
    """Returns (images [2N,112,112,3] BGR with pair k at 2k and 2k+1, same [N] bool).
    <name>.bin     : InsightFace verification set (list of encoded images, list of same/different)   [HF gaunernst/face-recognition-eval]
    <name>.parquet : columns image1, image2 (encoded bytes), target (1 = same person)   [HF cat-claws/face-verification]
    <name>-*.tar   : members <genuine|imposter>/<id>.img1, .img2                         [HF LSIbabnikz/xqlfw_bin]"""
    d = os.path.join(ROOT, "pairs")
    if os.path.exists(os.path.join(d, name + ".bin")):      # the original InsightFace file, when present
        return load_bin(os.path.join(d, name + ".bin"))
    pq = os.path.join(d, name + ".parquet")
    imgs, same = [], []
    if os.path.exists(pq):
        import pyarrow.parquet as papq
        t = papq.read_table(pq).to_pydict()
        for a, b, y in zip(t["image1"], t["image2"], t["target"]):
            imgs.append(_decode(a["bytes"]))
            imgs.append(_decode(b["bytes"]))
            same.append(bool(y))
    else:
        import glob, tarfile
        cur = {}
        for tp in sorted(glob.glob(os.path.join(d, name + "-*.tar"))):
            with tarfile.open(tp) as tf:
                for m in tf:
                    if not m.isfile():
                        continue
                    key, ext = m.name.rsplit(".", 1)
                    if ext in ("img1", "img2"):
                        cur.setdefault(key, {})[ext] = tf.extractfile(m).read()
        for key in sorted(cur):
            if "img1" in cur[key] and "img2" in cur[key]:
                imgs.append(_decode(cur[key]["img1"]))
                imgs.append(_decode(cur[key]["img2"]))
                same.append(key.split("/")[0].startswith("genuine"))
        # keep the two classes interleaved, as in the other sets, so every fold holds both
        order = np.argsort(np.arange(len(same)) % 2 * 0 + np.random.RandomState(0).permutation(len(same)))
        imgs = [imgs[2 * i + j] for i in order for j in (0, 1)]
        same = [same[i] for i in order]
    if not imgs:
        return None
    return np.stack(imgs), np.array(same, bool)


def shrink(imgs, n):
    out = np.empty_like(imgs)
    for i, im in enumerate(imgs):
        out[i] = cv2.resize(cv2.resize(im, (n, n), interpolation=cv2.INTER_AREA), (112, 112), interpolation=cv2.INTER_LINEAR)
    return out


class Rec:
    def __init__(self, path, prep, flip=True):
        self.flip = flip
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(path, so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0]
        self.prep = prep
        b = self.inp.shape[0]
        self.batch = 64 if not isinstance(b, int) else b
        self.t = []

    def blob(self, imgs):
        x = imgs.astype(np.float32)
        if self.prep in ("arcface", "raw_rgb", "rgb01"):
            x = x[..., ::-1]
        if self.prep in ("arcface", "bgr"):
            x = (x - 127.5) / 127.5
        elif self.prep == "rgb01":
            x = x / 255.0
        return np.ascontiguousarray(x.transpose(0, 3, 1, 2))

    def embed(self, imgs):
        """Image + mirror image, summed, L2-normalised. With flip off: the image alone, as the pipeline runs it."""
        out = None
        for i in range(0, len(imgs), self.batch):
            part = imgs[i:i + self.batch]
            n = len(part)
            if n < self.batch and isinstance(self.inp.shape[0], int):
                part = np.concatenate([part, np.repeat(part[-1:], self.batch - n, 0)])
            x = self.blob(part)
            t0 = time.perf_counter()
            e = self.sess.run(None, {self.inp.name: x})[0]
            self.t.append((time.perf_counter() - t0) / len(x))
            if self.flip:
                e = e + self.sess.run(None, {self.inp.name: np.ascontiguousarray(x[:, :, :, ::-1])})[0]
            e = e.reshape(len(x), -1)[:n]
            if out is None:
                out = np.zeros((len(imgs), e.shape[1]), np.float32)
            out[i:i + n] = e
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def kfold_acc(e1, e2, same, folds=10):
    d = np.sum((e1 - e2) ** 2, 1)                              # squared distance of unit vectors: 0..4
    thr = np.arange(0, 4, 0.005)
    n = len(d)
    idx = np.arange(n)
    acc = []
    for k in range(folds):
        te = idx[k * n // folds:(k + 1) * n // folds]
        tr = np.setdiff1d(idx, te)
        a = [np.mean((d[tr] < t) == same[tr]) for t in thr]
        best = thr[int(np.argmax(a))]
        acc.append(np.mean((d[te] < best) == same[te]))
    return 100 * float(np.mean(acc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("key")
    ap.add_argument("--sets", default="lfw,cfp_fp,agedb_30,cplfw,xqlfw")
    ap.add_argument("--sizes", default="40,28,20")
    ap.add_argument("--lowres-sets", default="lfw")
    ap.add_argument("--no-flip", action="store_true", help="embed the image alone, without its mirror image")
    ap.add_argument("--out", default=os.path.join(ROOT, "rec_results.jsonl"))
    a = ap.parse_args()
    sets = {}
    for s in a.sets.split(","):
        got = load_pairs(s)
        if got is None:
            print(json.dumps({"warning": "missing pair set " + s}), flush=True)
            continue
        sets[s] = got
        print(json.dumps({"set": s, "pairs": int(len(got[1])), "same": int(got[1].sum())}), flush=True)
    sizes = [int(x) for x in a.sizes.split(",") if x]
    small = {(s, n): shrink(sets[s][0], n) for s in a.lowres_sets.split(",") if s in sets for n in sizes}
    keys = list(REG) if a.key == "all" else a.key.split(",")
    for key in keys:
        file, prep = REG[key]
        path = os.path.join(MODELS, file)
        if not os.path.exists(path):
            print(json.dumps({"model": key, "error": "missing " + path}), flush=True)
            continue
        try:
            m = Rec(path, prep, flip=not a.no_flip)
            params = sum(int(np.prod(t.dims)) for t in onnx.load(path, load_external_data=False).graph.initializer)
            r = {"model": key, "prep": prep, "flip": not a.no_flip, "provider": m.sess.get_providers()[0].replace("ExecutionProvider", ""),
                 "file_mb": round(os.path.getsize(path) / 1e6, 1), "params_m": round(params / 1e6, 2)}
            t0 = time.time()
            for s, (imgs, same) in sets.items():
                e = m.embed(imgs)
                r["emb"] = int(e.shape[1])
                r[s] = round(kfold_acc(e[0::2], e[1::2], same), 2)
                for n in sizes:
                    if (s, n) not in small:
                        continue
                    el = m.embed(small[(s, n)])
                    r[f"{s}_both_{n}"] = round(kfold_acc(el[0::2], el[1::2], same), 2)
                    r[f"{s}_cross_{n}"] = round(kfold_acc(e[0::2], el[1::2], same), 2)
            r["gpu_ms_per_face"] = round(1000 * float(np.median(m.t[3:] or m.t)), 3)
            r["batch"] = m.batch
            r["total_s"] = round(time.time() - t0, 1)
        except Exception as ex:      # noqa: BLE001
            r = {"model": key, "error": str(ex)[:300]}
        print(json.dumps(r), flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
