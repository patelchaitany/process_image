#!/bin/bash
# usage: stress.sh <tag> <rec: edgeface_s|edgeface_base|r50|mbf> <counts> [extra fastpipe args...]
tag=$1; rec=$2; counts=$3; shift 3
cd /root/fastpipe && mkdir -p /mnt/e/bench/results/cpp/$tag
./build/fastpipe --det engines/det_r384_fp16.plan --rec engines/rec_${rec}_fp16.plan --counts $counts \
  --tag $tag --out /mnt/e/bench/results/cpp/$tag "$@" 2> /mnt/e/bench/results/cpp/$tag/stderr.log | python3 -c "
import sys, json
for l in sys.stdin:
    r = json.loads(l)
    keys = ['cameras','realtime_ok','arrived_fps','done_ratio','buffer_overflow','queued_at_end','delay_slope_s_per_s','cap_p50_ms','cap_p99_ms','cap_max_ms','cap_p95_ms','decode_lag_p95_ms','first_id_p95_ms','refresh_p95_s','fps_per_cam_median','total_fps','e2e_p50_ms','e2e_p95_ms','e2e_p99_ms','done_ratio','buffer_overflow','delay_slope_s_per_s','recognitions_per_frame','det_batch_avg','det_ms_per_frame','rec_batch_avg','rec_ms_per_face','det_gpu_ms_per_s','rec_gpu_ms_per_s','track_cpu_ms_per_s','gpu_util_pct','decoder_util_pct','gpu_mem_max_mib','ring_dropped']
    print({k: r.get(k) for k in keys}, flush=True)
"
tail -3 /mnt/e/bench/results/cpp/$tag/stderr.log | grep -v engine
