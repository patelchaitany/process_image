#!/bin/bash
# Dense 720p30 test clips: three face bands (rows 240-480 of the 1080p crowd clip, ~17 faces each) from three
# time-shifted copies + one floor band, stacked into 1280x720. Real faces at 720p scale; the frame is a mosaic.
cd /mnt/e/bench
for r in 0 1 2 3; do
  a=$r; b=$(( (r+1)%4 )); c=$(( (r+2)%4 )); d=$(( (r+3)%4 ))
  for codec in hevc h264; do
    out=dense720_${codec}_rot$r.mp4
    [ -f $out ] && continue
    enc="-c:v ${codec}_nvenc -preset p5 -bf 0 -g 60 -b:v 3M"
    [ $codec = h264 ] && enc="$enc -profile:v high"
    ffmpeg -v error -y -i cam_rot$a.mp4 -i cam_rot$b.mp4 -i cam_rot$c.mp4 -i cam_rot$d.mp4 -filter_complex \
      "[0:v]crop=1920:240:0:240,scale=1280:160[a];[1:v]crop=1920:240:0:240,scale=1280:160[b];[3:v]crop=1920:360:0:600,scale=1280:240[d];[2:v]crop=1920:240:0:240,scale=1280:160[c];[a][b][d][c]vstack=inputs=4,setpts=N/30/TB[v]" \
      -map "[v]" -r 30 $enc -color_primaries bt709 -color_trc bt709 -colorspace bt709 -an -shortest $out
  done
done
for f in dense720_hevc_rot0.mp4 dense720_h264_rot0.mp4; do echo "$f $(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height,r_frame_rate,nb_frames -of csv=p=0 $f)"; done
cd /root/fastpipe && ./build/fastpipe --det engines/det_r384_fp16.plan --rec engines/rec_${REC:-edgeface_s}_fp16.plan --video "/mnt/e/bench/dense720_hevc_rot{r}.mp4" \
  --max-faces 64 --dump 90 --dump-path /mnt/e/bench/results/cpp/dense_dump.bin 2>&1 | grep dump
python3 - <<'P'
import struct
n_frames = faces = 0; per = []
with open("/mnt/e/bench/results/cpp/dense_dump.bin", "rb") as f:
    while True:
        h = f.read(8)
        if len(h) < 8: break
        k, n = struct.unpack("ii", h); f.read(20 * n + 2048 * n); per.append(n)
print(f"faces per frame: mean {sum(per)/len(per):.1f}, min {min(per)}, max {max(per)} over {len(per)} frames")
P
