"""Cosine-similarity distributions of same-person and different-person pairs, for choosing the pipeline's match
threshold. The embedding is the face alone (no mirror image), as the pipeline computes it. All five pair sets pooled.

  python thresholds.py <key>[,<key>...] [--out thresholds.jsonl]

For each model: false matches (different people scoring at or above the threshold) and missed matches (same person
scoring below it) at each threshold."""
import argparse, json, os
import numpy as np
import eval_rec as er

GRID = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.46, 0.49, 0.50, 0.55, 0.58, 0.60, 0.65]   # includes each model's match_thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("keys")
    ap.add_argument("--sets", default="lfw,cfp_fp,agedb_30,cplfw,xqlfw")
    ap.add_argument("--out", default=os.path.join(er.ROOT, "thresholds.jsonl"))
    a = ap.parse_args()
    sets = {s: er.load_pairs(s) for s in a.sets.split(",")}
    for key in a.keys.split(","):
        file, prep = er.REG[key]
        m = er.Rec(os.path.join(er.MODELS, file), prep, flip=False)
        cos, same, per_set = [], [], {}
        for s, (imgs, y) in sets.items():
            e = m.embed(imgs)
            c = np.sum(e[0::2] * e[1::2], 1)
            cos.append(c)
            same.append(y)
            per_set[s] = {"missed_match_pct": {str(t): round(100 * float((c[y] < t).mean()), 2) for t in GRID},
                          "different_top3": [round(float(x), 3) for x in np.sort(c[~y])[-3:][::-1]]}
        cos, same = np.concatenate(cos), np.concatenate(same)
        g, i = cos[same], cos[~same]
        r = {"model": key, "pairs_same": int(len(g)), "pairs_different": int(len(i)),
             "different_mean": round(float(i.mean()), 4), "different_std": round(float(i.std()), 4),
             "different_p99": round(float(np.percentile(i, 99)), 4), "different_p999": round(float(np.percentile(i, 99.9)), 4),
             "different_max": round(float(i.max()), 4),
             "different_top10": [round(float(x), 3) for x in np.sort(i)[-10:][::-1]],
             "same_mean": round(float(g.mean()), 4), "same_p1": round(float(np.percentile(g, 1)), 4),
             "same_p5": round(float(np.percentile(g, 5)), 4), "same_p10": round(float(np.percentile(g, 10)), 4),
             "false_match_pct": {str(t): round(100 * float((i >= t).mean()), 3) for t in GRID},
             "missed_match_pct": {str(t): round(100 * float((g < t).mean()), 2) for t in GRID}}
        r["per_set"] = per_set
        # threshold for a given false-match rate
        for f in (1.0, 0.1):
            r[f"threshold_at_false_match_{f}pct"] = round(float(np.percentile(i, 100 - f)), 4)
        print(json.dumps(r), flush=True)
        with open(a.out, "a") as fh:
            fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
