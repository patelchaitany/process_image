#!/bin/bash
# usage: exp_sla.sh <fp dir name: fastpipe | fp-NAME> <tag> <cameras> <rec: edgeface_s|edgeface_base|r50|mbf> <scene: dense|normal> [fastpipe args...]
# One stress run under the strict end-to-end rule. Defaults: 10 processed fps (--analyze-fps 10), warm-up 6 s, window 18 s.
# Later args override earlier ones (e.g. add --analyze-fps 15, --det <plan>, --rec <plan>).
. "$(dirname "$0")/lock.sh"
fp=$1; tag=$2; cams=$3; rec=$4; scene=$5; shift 5
case $scene in dense) v='/mnt/e/bench/dense45_hevc_rot{r}.mp4';; normal) v='/mnt/e/bench/cam720_265_rot{r}.mp4';; *) echo "scene must be dense or normal"; exit 2;; esac
lock_gpu
$SCP "$(dirname "$0")"/wsl/*.sh laptop:E:/bench/wsl/
timeout 420 $SSH laptop "wsl -d Ubuntu-24.04 -u root -- bash /root/sync.sh exp_stress.sh /root/$fp $tag $rec $cams --video $v --analyze-fps 10 --warmup 6 --window 18 $*" | python3 -c "
import sys, json
K = ['cameras','sla_ok','sla_seconds_within_1s_pct','seconds_evaluated','sec_done_p50_ms','sec_done_p95_ms','sec_done_max_ms','gpu_util_pct','gpu_mem_max_mib','tick_ms_avg','ticks_per_s','det_batch_avg','rec_batch_avg','rec_slots_used_pct','rec_denied_per_s','gpu_requests_per_cam_s','why_new_grow_periodic_unsure_crossed_per_cam_s','tracks_created_per_cam_s','gpu_faces_per_frame','boxes_per_cam_s','named_box_pct','faces_left_unnamed_per_cam_s','name_p95_ms','out_p95_ms','out_max_ms','first_id_p50_ms','first_id_p95_ms','buffer_overflow','decoder_util_pct']
for l in sys.stdin:
    l = l.strip()
    if l.startswith('{'):
        r = json.loads(l); print(json.dumps({'tag': '$tag', **{k: r.get(k) for k in K}}))
    elif l: print(l[:300])"
