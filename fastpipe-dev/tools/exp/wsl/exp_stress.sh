#!/bin/bash
# (WSL) usage: exp_stress.sh <fp dir> <tag> <rec edgeface_s|edgeface_base|r50|mbf> <cams> [fastpipe args...]; prints the JSON summary line
fp=$1; tag=$2; rec=$3; cams=$4; shift 4
out=/mnt/e/bench/results/exp/$tag; rm -rf $out; mkdir -p $out
cd $fp && timeout 300 ./build/fastpipe --det /root/fastpipe/engines/det_r384_fp16.plan --rec /root/fastpipe/engines/rec_${rec}_fp16.plan \
  --counts $cams --tag $tag --out $out "$@" > $out/stdout.log 2> $out/stderr.log
echo "exit=$?"
tail -n 1 $out/summary.jsonl 2>/dev/null
grep -i -E "error|terminate|what\(\)|out of memory" $out/stderr.log | grep -v "lazy" | head -5
