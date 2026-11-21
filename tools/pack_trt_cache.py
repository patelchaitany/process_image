#!/usr/bin/env python3
"""Pack live trt_cache directories into assets/trt_engine_cache_t4_sm75_fp16.zip.

Run after warmup on a machine where engines are known good. Commit the updated
zip if you want teammates / new studios to skip cold TensorRT builds.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

DEFAULT_ZIP = "assets/trt_engine_cache_t4_sm75_fp16.zip"
MODELS = ("arcface", "yolo26_face")


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-repo", default=None, help="Model repository root")
    p.add_argument("--output", default=None, help=f"Output zip (default: <repo>/{DEFAULT_ZIP})")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = project_root()
    repo = Path(args.model_repo) if args.model_repo else root / "models" / "model_repository"
    out = Path(args.output) if args.output else root / DEFAULT_ZIP

    out.parent.mkdir(parents=True, exist_ok=True)
    added = 0
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for model in MODELS:
            cache = repo / model / "trt_cache"
            if not cache.is_dir():
                print(f"Missing {cache}", file=sys.stderr)
                return 1
            engines = list(cache.glob("*.engine")) + list(cache.glob("*.profile"))
            if not engines:
                print(f"No .engine/.profile in {cache} — run warmup_triton.py first", file=sys.stderr)
                return 1
            for p in sorted(engines):
                arc = f"{model}/trt_cache/{p.name}"
                zf.write(p, arcname=arc)
                print(f"  packed {arc} ({p.stat().st_size} bytes)")
                added += 1

    print(f"Wrote {out} ({out.stat().st_size} bytes, {added} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
