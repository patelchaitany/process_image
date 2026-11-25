#!/bin/bash
# (WSL) Dump every face's box and embedding for 450 frames of the two test clips, once per recogniser.
# Then, on Windows:  python fastpipe-dev\tools\check\track_scores.py E:\bench\results\cpp\ts_<scene>_<rec>.bin
cd /root/fastpipe
for scene in normal:/mnt/e/bench/cam720_265_rot0.mp4 dense:/mnt/e/bench/dense45_hevc_rot0.mp4; do
  name=${scene%%:*}; v=${scene#*:}
  for rec in ${RECS:-r50 edgeface_s edgeface_base}; do
    ./build/fastpipe --det engines/det_r384_fp16.plan --rec engines/rec_${rec}_fp16.plan --video "$v" \
      --dump ${FRAMES:-450} --dump-path /mnt/e/bench/results/cpp/ts_${name}_${rec}.bin 2>&1 | grep "dump:"
  done
done
