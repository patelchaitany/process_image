#!/bin/bash
# (WSL) usage: exp_boxeval_wsl.sh <fp dir> <tag> <video> [fastpipe args...]
fp=$1; tag=$2; video=$3; shift 3
out=/mnt/e/bench/results/exp/$tag; rm -rf $out; mkdir -p $out
cd $fp && timeout 300 ./build/fastpipe --det /root/fastpipe/engines/det_r384_fp16.plan --rec /root/fastpipe/engines/rec_${REC:-edgeface_s}_fp16.plan \
  --video "$video" --boxeval 540 --out $out "$@" 2> $out/stderr.log
echo "exit=$?"
grep -i -E "error|terminate|what\(\)" $out/stderr.log | grep -v "lazy" | head -5
