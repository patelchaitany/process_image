#!/bin/bash
cd /mnt/e/bench
for r in 0 1 2 3; do
  a=$r; b=$(( (r+1)%4 )); c=$(( (r+2)%4 )); d=$(( (r+3)%4 ))
  for codec in hevc h264; do
    out=dense45_${codec}_rot$r.mp4
    [ -f $out ] && continue
    enc="-c:v ${codec}_nvenc -preset p5 -bf 0 -g 60 -b:v 3M"
    [ $codec = h264 ] && enc="$enc -profile:v high"
    ffmpeg -v error -y -i cam_rot$a.mp4 -i cam_rot$b.mp4 -i cam_rot$c.mp4 -i cam_rot$d.mp4 -filter_complex \
      "[0:v]crop=1920:240:0:240,scale=1280:160[a];[1:v]crop=1920:240:0:240,scale=1280:160[b];[2:v]crop=1920:120:0:330,scale=1280:80[c];[3:v]crop=1920:480:0:560,scale=1280:320[d];[a][d][b][c]vstack=inputs=4,setpts=N/30/TB[v]" \
      -map "[v]" -r 30 $enc -color_primaries bt709 -color_trc bt709 -colorspace bt709 -an -shortest $out
  done
done
cd /root/fastpipe && for r in 0 1; do ./build/fastpipe --det engines/det_r384_fp16.plan --rec engines/rec_${REC:-edgeface_s}_fp16.plan --video "/mnt/e/bench/dense45_hevc_rot$r.mp4" \
  --max-faces 64 --dump 150 --dump-path /mnt/e/bench/results/cpp/dense45_dump$r.bin 2>&1 | grep -c dump >/dev/null; done
python3 - <<'P'
import struct
for r in (0, 1):
    per = []
    with open(f"/mnt/e/bench/results/cpp/dense45_dump{r}.bin", "rb") as f:
        while True:
            h = f.read(8)
            if len(h) < 8: break
            k, n = struct.unpack("ii", h); f.read(20 * n + 2048 * n); per.append(n)
    print(f"rot{r}: faces per frame mean {sum(per)/len(per):.1f}, min {min(per)}, max {max(per)} over {len(per)} frames")
P
