#!/bin/bash
# usage: boxeval.sh <tag> <video pattern> [frames]
cd /root/fastpipe && mkdir -p /mnt/e/bench/results/cpp/$1
./build/fastpipe --det engines/det_r384_fp16.plan --rec engines/rec_${REC:-edgeface_s}_fp16.plan --video "$2" --boxeval ${3:-540} --out /mnt/e/bench/results/cpp/$1 2> /mnt/e/bench/results/cpp/$1/stderr.log
grep -i -E "error|terminate|what\(" /mnt/e/bench/results/cpp/$1/stderr.log | head -5
