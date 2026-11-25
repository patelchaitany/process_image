#!/bin/bash
# 1080p test clips as CCTV cameras commonly send them: 10 fps H.264 without B-frames, and 25 fps H.265 (NVENC)
cd /mnt/e/bench
for r in 0 1 2 3; do
  [ -f cam10_rot$r.mp4 ] || ffmpeg -v error -y -i cam_rot$r.mp4 -vf fps=10 -c:v h264_nvenc -preset p5 -profile:v high -bf 0 -g 20 -b:v 2M -color_primaries bt709 -color_trc bt709 -colorspace bt709 -an cam10_rot$r.mp4
  [ -f cam265_rot$r.mp4 ] || ffmpeg -v error -y -i cam_rot$r.mp4 -c:v hevc_nvenc -preset p5 -bf 0 -g 50 -b:v 3M -color_primaries bt709 -color_trc bt709 -colorspace bt709 -an cam265_rot$r.mp4
done
for f in cam10_rot0.mp4 cam265_rot0.mp4; do echo "$f $(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height,r_frame_rate,has_b_frames -of csv=p=0 $f)"; done
