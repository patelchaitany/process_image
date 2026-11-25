cd /root/fastpipe && ./build/fastpipe --det engines/det_r384_fp16.plan --rec engines/rec_${REC:-edgeface_s}_fp16.plan --video "/mnt/e/bench/cam720_265_rot{r}.mp4" \
  --dump 60 --dump-path /mnt/e/bench/results/cpp/dump720.bin 2>&1 | grep dump
