#!/usr/bin/env python3
"""Warm YOLO and ArcFace on Triton after container start (TensorRT FP16 path)."""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run warmup_yolo.py and warmup_arcface.py against a Triton server"
    )
    parser.add_argument("--triton", default="localhost:8001", help="Triton gRPC URL")
    parser.add_argument("--max-batch", type=int, default=32, help="ArcFace largest batch")
    return parser.parse_args()


def main():
    args = parse_args()
    tools = Path(__file__).resolve().parent
    py = sys.executable

    steps = [
        [py, str(tools / "warmup_yolo.py"), "--triton", args.triton],
        [
            py,
            str(tools / "warmup_arcface.py"),
            "--triton",
            args.triton,
            "--max-batch",
            str(args.max_batch),
        ],
    ]
    for cmd in steps:
        print(f"\n>>> {' '.join(cmd)}\n", flush=True)
        rc = subprocess.call(cmd)
        if rc != 0:
            return rc
    print("\nAll Triton warmups finished OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
