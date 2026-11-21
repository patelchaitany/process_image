#!/usr/bin/env python3
"""Warm YOLO face detection for batch sizes used at runtime.

After enabling TensorRT FP16 in Triton, the first inference per batch shape can
be slow while engines build or load from trt_cache. Run after every container
start, before video or enrollment.
"""

import argparse
import sys
import time

import numpy as np
import tritonclient.grpc as grpcclient


def parse_args():
    parser = argparse.ArgumentParser(
        description="Warm yolo26_face for batch sizes 1, 4, 8 on a running Triton server"
    )
    parser.add_argument("--triton", default="localhost:8001", help="Triton gRPC URL")
    parser.add_argument("--model", default="yolo26_face", help="Triton model name")
    parser.add_argument(
        "--batches",
        default="1,4,8",
        help="Comma-separated batch sizes to warm (default: 1,4,8)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    batches = []
    for part in args.batches.split(","):
        part = part.strip()
        if not part:
            continue
        n = int(part)
        if n < 1:
            print("batch sizes must be >= 1", file=sys.stderr)
            return 1
        batches.append(n)
    if not batches:
        print("no batch sizes to warm", file=sys.stderr)
        return 1

    print(f"Connecting to Triton at {args.triton}...", end=" ", flush=True)
    try:
        client = grpcclient.InferenceServerClient(args.triton)
        if not client.is_server_live():
            print("FAILED (server not live)")
            return 1
        if not client.is_model_ready(args.model):
            print(f"FAILED (model {args.model} not ready)")
            return 1
    except Exception as exc:
        print(f"FAILED ({exc})")
        return 1
    print("OK")

    print(f"Warming {args.model} for batch sizes {batches}")
    total = 0.0
    for n in batches:
        data = np.zeros((n, 3, 640, 640), dtype=np.float32)
        inp = grpcclient.InferInput("images", list(data.shape), "FP32")
        inp.set_data_from_numpy(data)
        out = grpcclient.InferRequestedOutput("output0")

        start = time.perf_counter()
        result = client.infer(args.model, inputs=[inp], outputs=[out])
        elapsed = time.perf_counter() - start
        total += elapsed

        tensor = result.as_numpy("output0")
        if tensor is None or tensor.shape != (n, 8400, 5):
            print(
                f"  batch {n}: unexpected output shape "
                f"{None if tensor is None else tensor.shape}"
            )
            return 1
        print(f"  batch {n}: {elapsed:6.2f}s", flush=True)

    print(f"Warmed batches {batches} in {total:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
