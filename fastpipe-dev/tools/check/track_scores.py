"""(Windows venv) Match scores under real pipeline conditions, labelled by the video itself.

Input: a `fastpipe --dump N` file (every frame: every face's box and embedding, as the pipeline computes them: face cut
out by its box from the NV12 frame, no landmark alignment, FP16 engine).

Labels come from tracking the boxes through the clip:
  same person      : one track, two frames `gap` frames apart (1 s and 3 s at 30 fps: what a re-check compares)
  different people : two tracks in the same frame (two faces in one picture are never the same person)

Prints, per gap: the score distributions, and for each threshold the share of same-person pairs that would match and
the share of different-people pairs that would match. Use it to set <engine>.thr.

  python track_scores.py dump.bin [--fps 30] [--out scores.jsonl] [--tag name]
"""
import argparse, json, struct
import numpy as np

GRID = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.46, 0.50, 0.55, 0.58, 0.60]   # includes each model's match_thr


def load(path):
    frames = []
    with open(path, "rb") as f:
        while True:
            h = f.read(8)
            if len(h) < 8:
                break
            _, n = struct.unpack("ii", h)
            b = np.frombuffer(f.read(20 * n), np.float32).reshape(n, 5)
            e = np.frombuffer(f.read(4 * 512 * n), np.float32).reshape(n, 512).astype(np.float32)
            e = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)
            frames.append((b[:, :4].copy(), e))
    return frames


def iou(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def track(frames, thr=0.4, max_gap=5):
    """Track id per face per frame. A face continues a track when the two are each other's best overlap (>= thr) and
    no other live track overlaps the face by 0.2 or more: faces that pass close to another face start a new track
    instead of risking a swapped label."""
    ids, live, nxt = [], {}, 0          # live: id -> (box, last frame)
    for t, (boxes, _) in enumerate(frames):
        cur = -np.ones(len(boxes), int)
        keys = [k for k, (_, last) in live.items() if t - last <= max_gap]
        if keys and len(boxes):
            m = iou(boxes, np.stack([live[k][0] for k in keys]))
            bj, bi = m.argmax(1), m.argmax(0)
            for i, j in enumerate(bj):
                if m[i, j] >= thr and bi[j] == i and (m.shape[1] < 2 or np.sort(m[i])[-2] < 0.2) \
                        and (m.shape[0] < 2 or np.sort(m[:, j])[-2] < 0.2):
                    cur[i] = keys[j]
        for i in range(len(cur)):
            if cur[i] < 0:
                cur[i] = nxt; nxt += 1
            live[cur[i]] = (boxes[i], t)
        ids.append(cur)
    return ids, nxt


def pct(v, q):
    return round(float(np.percentile(v, q)), 3) if len(v) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    frames = load(a.dump)
    ids, ntracks = track(frames)
    # different people: every pair of faces in the same frame, every 3rd frame
    diff, diff_w = [], []
    for t in range(0, len(frames), 3):
        b, e = frames[t]
        if len(b) < 2:
            continue
        s = e @ e.T
        iu = np.triu_indices(len(b), 1)
        diff.append(s[iu])
        w = b[:, 2] - b[:, 0]
        diff_w.append(np.minimum(w[iu[0]], w[iu[1]]))
    diff = np.concatenate(diff) if diff else np.zeros(0)
    diff_w = np.concatenate(diff_w) if diff_w else np.zeros(0)
    where = [dict(zip(i.tolist(), range(len(i)))) for i in ids]          # per frame: track id -> row
    res = {"tag": a.tag, "frames": len(frames), "faces_per_frame": round(float(np.mean([len(b) for b, _ in frames])), 1),
           "tracks": int(ntracks), "different_pairs": int(len(diff)),
           "different": {"mean": round(float(diff.mean()), 3), "std": round(float(diff.std()), 3), "p99": pct(diff, 99),
                         "p99.9": pct(diff, 99.9), "p99.99": pct(diff, 99.99), "max": round(float(diff.max()), 3)},
           "different_match_pct": {str(t): round(100 * float((diff >= t).mean()), 4) for t in GRID}, "gaps": {}}
    for sec in (1, 3):
        gap = int(round(sec * a.fps))
        same, same_w = [], []
        for t in range(0, len(frames) - gap):
            b0, e0 = frames[t]
            b1, e1 = frames[t + gap]
            for tid, i in where[t].items():
                j = where[t + gap].get(tid)
                if j is not None:
                    same.append(float(e0[i] @ e1[j]))
                    same_w.append(min(b0[i, 2] - b0[i, 0], b1[j, 2] - b1[j, 0]))
        same, same_w = np.array(same), np.array(same_w)
        g = {"same_pairs": int(len(same)), "same": {"mean": round(float(same.mean()), 3) if len(same) else None,
                                                    "p5": pct(same, 5), "p10": pct(same, 10), "p25": pct(same, 25), "p50": pct(same, 50)},
             "same_match_pct": {str(t): round(100 * float((same >= t).mean()), 2) for t in GRID} if len(same) else {}}
        # by face width (the narrower face of the pair), in frame pixels
        g["same_match_pct_by_width"] = {}
        for lo, hi in ((0, 32), (32, 48), (48, 80), (80, 10000)):
            m = (same_w >= lo) & (same_w < hi)
            if m.sum() >= 50:
                g["same_match_pct_by_width"][f"{lo}-{hi}"] = {"pairs": int(m.sum()),
                                                              **{str(t): round(100 * float((same[m] >= t).mean()), 1) for t in (0.3, 0.4, 0.45, 0.46, 0.5, 0.55, 0.58, 0.6)}}
        res["gaps"][f"{sec}s"] = g
    print(json.dumps(res))
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
