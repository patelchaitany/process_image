#!/bin/bash
# (WSL) TensorRT FP16 speed of one ONNX model: trt_speed.sh <onnx path under /mnt/e/bench/models> <input name> <CxHxW e.g. 3x384x640> <batch|0 = fixed shape>
# Prints: file, batch, mean GPU compute ms per batch, ms per item.
f=$1; name=$2; chw=$3; b=$4
if [ "$b" = "0" ]; then shapes=""; b=1; else shapes="--minShapes=$name:1x$chw --optShapes=$name:${b}x$chw --maxShapes=$name:${b}x$chw --shapes=$name:${b}x$chw"; fi
out=$(/usr/src/tensorrt/bin/trtexec --onnx="$f" --fp16 $shapes --duration=4 --warmUp=500 --useSpinWait --noDataTransfers --useCudaGraph 2>&1)
ms=$(echo "$out" | grep "GPU Compute Time: min" | sed -E 's/.*mean = ([0-9.]+) ms.*/\1/')
if [ -z "$ms" ]; then echo "$(basename $f) FAILED: $(echo "$out" | grep -E "Error|error" | head -2 | cut -c1-200)"; else echo "$(basename $f) batch $b gpu_ms_per_batch $ms ms_per_item $(python3 -c "print(round($ms/$b,3))")"; fi
