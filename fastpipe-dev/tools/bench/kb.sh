cd /root/fastpipe && ./build/fastpipe --det engines/det_r384_fp16.plan --rec engines/rec_${REC:-edgeface_s}_fp16.plan --kbench 1 --graphs 1 2>&1 | grep -v "TRT\]\|engine\]"
