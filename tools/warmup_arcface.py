#!/usr/bin/env python3
"""Warm ArcFace for every batch size 1..max-batch.

With ONNX Runtime + TensorRT FP16, the first request per shape [N, 3, 112, 112]
can be slow while TensorRT engines build or load from trt_cache. N is the face
count in a frame. This script sends one zero-filled request per batch size so
normal video frames do not pay that cost.

Run after every Triton container start (or use warmup_triton.py for YOLO too),
once models are ready, and before processing video. Switching input files does
not require re-warming if Triton stays up.
"""

import argparse
import sys
import time

import numpy as np
import tritonclient.grpc as grpcclient


def parse_args():
    parser = argparse.ArgumentParser(
        description="Warm ArcFace for batch sizes 1..max-batch on a running Triton server"
    )
    parser.add_argument("--triton", default="localhost:8001", help="Triton gRPC URL")
    parser.add_argument("--model", default="arcface", help="Triton model name")
    parser.add_argument("--max-batch", type=int, default=32, help="Largest batch size to warm")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_batch < 1:
        print("--max-batch must be >= 1", file=sys.stderr)
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

    print(f"Warming {args.model} for batch sizes 1..{args.max_batch}")
    total = 0.0
    for n in range(1, args.max_batch + 1):
        data = np.zeros((n, 3, 112, 112), dtype=np.float32)
        inp = grpcclient.InferInput("input", list(data.shape), "FP32")
        inp.set_data_from_numpy(data)
        out = grpcclient.InferRequestedOutput("output")

        start = time.perf_counter()
        result = client.infer(args.model, inputs=[inp], outputs=[out])
        elapsed = time.perf_counter() - start
        total += elapsed

        embedding = result.as_numpy("output")
        if embedding is None or embedding.shape != (n, 512):
            print(f"  batch {n}: unexpected output shape {None if embedding is None else embedding.shape}")
            return 1
        print(f"  batch {n:2d}: {elapsed:6.2f}s", flush=True)

    print(f"Warmed batches 1..{args.max_batch} in {total:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
