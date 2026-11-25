cd /root/fastpipe && ./build/fastpipe --det engines/det_r384_fp16.plan --rec engines/rec_${REC:-edgeface_s}_fp16.plan --video "/mnt/e/bench/dense45_hevc_rot{r}.mp4" \
  --max-faces 64 --kfeval 540 --counts 2,3,6 2>&1 | grep keyframe | tee /mnt/e/bench/results/cpp/kfeval_dense.jsonl
