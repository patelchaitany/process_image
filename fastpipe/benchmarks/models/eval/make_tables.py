"""Print the result tables (Markdown) from det_results.jsonl, det_results_size.jsonl, rec_results.jsonl,
rec_results_noflip.jsonl and trt_speed.log.
  python3 make_tables.py [folder holding the three files, default ..]"""
import json, os, re, sys

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# model key -> (ONNX file name as it appears in trt_speed.log, licence, commercial use)
DET = {
    "yolov8n-face (current)":    ("yolov8n-face.onnx", "AGPL-3.0 (Ultralytics)", "copyleft"),
    "scrfd-500m":                ("det_500m.onnx", "InsightFace weights", "no"),
    "scrfd-1g":                  ("scrfd_1g.onnx", "InsightFace weights", "no"),
    "scrfd-2.5g":                ("det_2.5g.onnx", "InsightFace weights", "no"),
    "scrfd-10g":                 ("det_10g.onnx", "InsightFace weights", "no"),
    "scrfd-34g":                 ("scrfd_34g.onnx", "InsightFace weights", "no"),
    "yunet-s":                   ("face_detection_yunet_2023mar.onnx", "MIT", "yes"),
    "yunet-n":                   ("yunet_n_640_640.onnx", "BSD-3", "yes"),
    "retinaface-mnet0.25":       ("retinaface_mv1_0.25.onnx", "MIT", "yes*"),
    "retinaface-mnetv2":         ("retinaface_mv2.onnx", "MIT", "yes*"),
    "yolov5n-face":              ("yolov5n_face.onnx", "GPL-3.0", "copyleft"),
    "yolov5s-face":              ("yolov5s_face.onnx", "GPL-3.0", "copyleft"),
    "yolov8n-face (akanametov)": ("aka_yolov8n-face.onnx", "GPL-3.0", "copyleft"),
    "yolov10n-face":             ("aka_yolov10n-face.onnx", "GPL-3.0", "copyleft"),
    "yolov11n-face":             ("aka_yolov11n-face.onnx", "GPL-3.0", "copyleft"),
    "yolov12n-face":             ("aka_yolov12n-face.onnx", "GPL-3.0", "copyleft"),
    "centerface":                ("centerface.onnx", "MIT", "yes*"),
}
REC = {
    "arcface-r50-w600k (current)": ("w600k_r50.onnx", "InsightFace weights", "no"),
    "mobilefacenet-w600k":         ("w600k_mbf.onnx", "InsightFace weights", "no"),
    "edgeface-xxs":                ("edgeface_xxs.onnx", "CC BY-NC-SA 4.0", "no"),
    "edgeface-xs":                 ("edgeface_xs_gamma_06.onnx", "CC BY-NC-SA 4.0", "no"),
    "edgeface-s":                  ("edgeface_s_gamma_05.onnx", "CC BY-NC-SA 4.0", "no"),
    "edgeface-base":               ("edgeface_base.onnx", "CC BY-NC-SA 4.0", "no"),
    "adaface-ir18":                ("adaface_ir_18.onnx", "MIT code, research data", "unclear"),
    "adaface-ir50":                ("adaface_ir_50.onnx", "MIT code, research data", "unclear"),
    "auraface-r100":               ("auraface_r100.onnx", "Apache-2.0", "yes"),
    "sface":                       ("face_recognition_sface_2021dec.onnx", "Apache-2.0", "yes"),
    "mobilenetv1-0.25":            ("mobilenetv1_0.25.onnx", "MIT code, research data", "unclear"),
    "mobilenetv3-small":           ("mobilenetv3_small.onnx", "MIT code, research data", "unclear"),
    "mobilenetv2":                 ("mobilenetv2.onnx", "MIT code, research data", "unclear"),
    "mobilenetv3-large":           ("mobilenetv3_large.onnx", "MIT code, research data", "unclear"),
}


def rows(name, key=lambda r: r["model"]):
    out = {}
    p = os.path.join(D, name)
    if os.path.exists(p):
        for line in open(p):
            r = json.loads(line)
            if "error" not in r:
                out[key(r)] = r          # a later good run replaces an earlier one
    return out


def speed():
    out = {}
    p = os.path.join(D, "trt_speed.log")
    if os.path.exists(p):
        for line in open(p, errors="replace"):
            m = re.match(r"(\S+) batch (\d+) gpu_ms_per_batch ([0-9.]+)", line)
            if m:
                out[m.group(1)] = (int(m.group(2)), float(m.group(3)))
            m = re.match(r"(\S+) FAILED", line)
            if m:
                out[m.group(1)] = None
    return out


def main():
    sp = speed()
    det, rec = rows("det_results.jsonl"), rows("rec_results.jsonl")

    print("### Detectors: WIDER FACE validation, 3,226 images, 39,708 labelled faces, long side 640\n")
    print("| Model | Size MB | Easy | Medium | Hard | Dense hard | Dense hard found at 90% precision | TensorRT ms / frame | Licence | Commercial use |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for k in sorted(det, key=lambda k: det[k]["file_mb"]):
        r = det[k]
        f, lic, com = DET.get(k, ("", "", ""))
        s = sp.get(f)
        ms = "failed" if (f in sp and s is None) else (f"{s[1] / s[0]:.2f}" if s else "")
        print(f"| {k} | {r['file_mb']:.1f} | {r['easy_ap']:.1f} | {r['medium_ap']:.1f} | {r['hard_ap']:.1f} | {r['dense_hard_ap']:.1f} | "
              f"{r['dense_hard_found_at_p90']:.0f}% | {ms} | {lic} | {com} |")

    print("\n### Recognisers: same-person / different-person pairs, 10-fold accuracy %\n")
    print("| Model | Size MB | LFW | CFP-FP | AgeDB-30 | CPLFW | XQLFW | Lowest | TensorRT ms / face | Licence | Commercial use |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    main5 = ["lfw", "cfp_fp", "agedb_30", "cplfw", "xqlfw"]
    for k in sorted(rec, key=lambda k: rec[k]["file_mb"]):
        r = rec[k]
        f, lic, com = REC.get(k, ("", "", ""))
        s = sp.get(f)
        ms = "failed" if (f in sp and s is None) else (f"{s[1] / s[0]:.3f} (batch {s[0]})" if s else "")
        print(f"| {k} | {r['file_mb']:.1f} | " + " | ".join(f"{r[c]:.2f}" for c in main5) + f" | {min(r[c] for c in main5):.1f} | {ms} | {lic} | {com} |")

    print("\n### Recognisers on small faces: LFW with faces shrunk, 10-fold accuracy %\n")
    print("| Model | Size MB | Both 28 px | Both 20 px | Both 14 px | One 28 px | One 20 px | One 14 px |")
    print("|---|---|---|---|---|---|---|---|")
    for k in sorted(rec, key=lambda k: rec[k]["file_mb"]):
        r = rec[k]
        print(f"| {k} | {r['file_mb']:.1f} | " + " | ".join(f"{r[c]:.2f}" for c in
              ["lfw_both_40", "lfw_both_28", "lfw_both_20", "lfw_cross_40", "lfw_cross_28", "lfw_cross_20"]) + " |")

    big = rows("det_results_size.jsonl", key=lambda r: (r["model"], r["input"]))
    if big:
        print("\n### Detectors at a larger input: same images, long side 640 against 1280\n")
        print("| Model | Size MB | Input | Easy | Medium | Hard | Dense hard | Dense hard found at 90% precision | ONNX Runtime ms / image |")
        print("|---|---|---|---|---|---|---|---|---|")
        for k in sorted({m for m, _ in big}, key=lambda k: det[k]["file_mb"]):
            for r in [det[k]] + [big[x] for x in sorted(big) if x[0] == k]:
                print(f"| {k} | {r['file_mb']:.1f} | {r['input']} | {r['easy_ap']:.1f} | {r['medium_ap']:.1f} | {r['hard_ap']:.1f} | {r['dense_hard_ap']:.1f} | "
                      f"{r['dense_hard_found_at_p90']:.0f}% | {r['gpu_ms_per_image']:.1f} |")

    nf = rows("rec_results_noflip.jsonl")
    if nf:
        print("\n### Recognisers without the mirror image (as the pipeline runs them), 10-fold accuracy %\n")
        print("| Model | Size MB | LFW | CFP-FP | AgeDB-30 | CPLFW | XQLFW | Lowest | Largest change from the table above |")
        print("|---|---|---|---|---|---|---|---|---|")
        for k in sorted(nf, key=lambda k: nf[k]["file_mb"]):
            r = nf[k]
            ch = max((r[c] - rec[k][c] for c in main5), key=abs) if k in rec else float("nan")
            print(f"| {k} | {r['file_mb']:.1f} | " + " | ".join(f"{r[c]:.2f}" for c in main5) + f" | {min(r[c] for c in main5):.1f} | {ch:+.2f} |")


if __name__ == "__main__":
    main()
