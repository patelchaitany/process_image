cd /root/fastpipe
for acc in 1 4 16 64; do for meas in 0.05 0.02; do
  ./build/fastpipe --det engines/det_r384_fp16.plan --rec engines/rec_${REC:-edgeface_s}_fp16.plan --video "/mnt/e/bench/cam30_265_rot{r}.mp4" --kfeval 540 --counts 3,6 --kf-acc $acc --kf-meas $meas 2>/dev/null | grep keyframe | sed "s/^{/{\"acc\": $acc, \"meas\": $meas, /"
done; done
