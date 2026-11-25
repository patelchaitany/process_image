#!/bin/bash
# usage: slarun.sh <tag> <cams> <fastpipe args...>   prints the SLA keys of one run   (REC=r50 for another recogniser)
# The laptop is reached through an ssh config holding "Host laptop" (address, user, key, known-hosts file).
# FP_SSH_DIR is the folder with that ssh_config; it also holds the GPU lock.
S=${FP_SSH_DIR:?set FP_SSH_DIR to the folder that holds ssh_config with a "Host laptop" entry}
tag=$1; shift
timeout 900 ssh -F $S/ssh_config laptop "wsl -d Ubuntu-24.04 -u root -- bash /root/sync.sh stress.sh $tag ${REC:-edgeface_s} $*" > /dev/null
timeout 60 ssh -F $S/ssh_config laptop "type E:\bench\results\cpp\\$tag\summary.jsonl 2>nul & findstr /i \"error terminate what\" E:\bench\results\cpp\\$tag\stderr.log" | python3 -c "
import sys, json
K = ['cameras','sla_ok','sla_seconds_within_1s_pct','sec_done_p50_ms','sec_done_p95_ms','sec_done_p99_ms','sec_done_max_ms','name_p95_ms','name_max_ms','name_wait_max_ms','named_box_pct','faces_left_unnamed_per_cam_s','rec_slots_used_pct','rec_denied_per_s','out_p95_ms','out_max_ms','tick_ms_avg','gpu_util_pct','gpu_mem_max_mib','buffer_overflow']
for l in sys.stdin:
    l = l.strip()
    if l.startswith('{'):
        r = json.loads(l); print('$tag', {k: r.get(k) for k in K})
    elif l: print('  log:', l[:200])"
