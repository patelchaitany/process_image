#!/usr/bin/env python3
"""Restore TensorRT engine cache from the bundled zip into the model repository.

Use after `download_models.py --real --gpu` on a fresh machine so you skip the
multi-hour cold TensorRT build. Requires the same GPU class the bundle was built
on (T4 / sm75) and Triton 24.01 with ORT+TensorRT FP16 configs.

Default archive: assets/trt_engine_cache_t4_sm75_fp16.zip
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
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--zip",
        dest="zip_path",
        default=None,
        help=f"Path to engine bundle (default: <repo>/{DEFAULT_ZIP})",
    )
    p.add_argument(
        "--model-repo",
        default=None,
        help="Model repository root (default: <repo>/models/model_repository)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List zip members only; do not write files",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = project_root()
    zip_path = Path(args.zip_path) if args.zip_path else root / DEFAULT_ZIP
    repo = Path(args.model_repo) if args.model_repo else root / "models" / "model_repository"

    if not zip_path.is_file():
        print(f"Missing bundle: {zip_path}", file=sys.stderr)
        print("Run tools/pack_trt_cache.py after a successful warmup to create it.", file=sys.stderr)
        return 1

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.namelist() if not m.endswith("/")]
        expected_prefixes = tuple(f"{m}/trt_cache/" for m in MODELS)
        bad = [m for m in members if not any(m.startswith(p) for p in expected_prefixes)]
        if bad:
            print("Unexpected paths in zip (expected arcface/trt_cache/* and yolo26_face/trt_cache/*):", file=sys.stderr)
            for m in bad[:10]:
                print(f"  {m}", file=sys.stderr)
            return 1

        if args.dry_run:
            print(f"Would extract {len(members)} file(s) from {zip_path} -> {repo}")
            for m in members:
                print(f"  {m}")
            return 0

        extracted = 0
        for member in members:
            # member: arcface/trt_cache/foo.engine
            parts = Path(member).parts
            if len(parts) < 3 or parts[1] != "trt_cache":
                continue
            model_name = parts[0]
            filename = parts[-1]
            dest_dir = repo / model_name / "trt_cache"
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / filename
            with zf.open(member) as src, open(dest, "wb") as out:
                out.write(src.read())
            extracted += 1
            print(f"  wrote {dest.relative_to(root)} ({dest.stat().st_size} bytes)")

    print(f"Restored {extracted} file(s) into {repo}")
    print("Next: start Triton (RUNBOOK step 5), then warmup_triton.py (step 9).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
