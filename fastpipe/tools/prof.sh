#!/bin/bash
# usage: prof.sh <cams> <tag> [extra fastpipe args]
N=$1; T=$2; shift 2
cd /root/fastpipe && mkdir -p prof
NSYS=$(ls -d /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | tail -1); NSYS=${NSYS:-nsys}
$NSYS profile -t cuda,nvtx,osrt --cuda-graph-trace=node --force-overwrite true -o prof/$T \
  build/fastpipe --det engines/det_r384_fp16.plan --rec engines/rec_${REC:-edgeface_s}_fp16.plan --counts $N --warmup 4 --window 8 --out prof/run-$T "$@" > prof/$T.stdout 2> prof/$T.stderr
tail -2 prof/$T.stderr
$NSYS stats --force-export=true --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum,cuda_gpu_mem_time_sum --format csv --output prof/$T prof/$T.nsys-rep > /dev/null 2>&1
ls prof/${T}_*.csv
cp prof/${T}_*.csv prof/$T.nsys-rep /mnt/e/bench/results/cpp/ 2>/dev/null
