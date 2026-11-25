#!/bin/bash
# (WSL) Nsight Systems timeline at a given camera count, then Nsight Compute on the four custom kernels.
set -e
cd /root/fastpipe
N=${1:-16}
DET=${DET:-engines/det_r384_fp16.plan}
REC=${REC:-engines/rec_edgeface_s_fp16.plan}
P=prof; mkdir -p $P
NSYS=$(ls -d /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | tail -1); NSYS=${NSYS:-nsys}
NCU=$(ls -d /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | tail -1)
$NSYS profile -t cuda,nvtx,osrt --cuda-graph-trace=node --force-overwrite true -o $P/n$N \
  build/fastpipe --det $DET --rec $REC --counts $N --warmup 5 --window 10 --out $P/run-n$N $EXTRA > $P/n$N.stdout 2>&1
$NSYS stats --force-export=true --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
  --format csv --output $P/n$N $P/n$N.nsys-rep > /dev/null 2>&1 || true
ls $P/n${N}_*.csv
if [ -n "$NCU" ] && [ "${SKIP_NCU:-0}" = 0 ]; then
  for k in prep_det decode_nms crop_prep match; do
    $NCU --set full -k regex:$k --launch-skip 60 --launch-count 3 --import-source yes -f -o $P/ncu_$k \
      build/fastpipe --det $DET --rec $REC --counts 8 --warmup 3 --window 4 --out $P/ncu-run > $P/ncu_$k.log 2>&1 \
      || tail -5 $P/ncu_$k.log
    $NCU --import $P/ncu_$k.ncu-rep --page details --csv > $P/ncu_$k.csv 2>/dev/null || true
  done
fi
