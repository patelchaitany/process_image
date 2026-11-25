#!/bin/bash
# 1080p30 clips from the real 25 fps frames re-timed to 30 fps (no duplicated or synthetic frames; motion is 1.2x faster)
cd /mnt/e/bench
for r in 0 1 2 3; do
  [ -f cam30_264_rot$r.mp4 ] || ffmpeg -v error -y -i cam_rot$r.mp4 -vf "setpts=N/30/TB" -r 30 -c:v h264_nvenc -preset p5 -profile:v high -bf 0 -g 60 -b:v 4M -color_primaries bt709 -color_trc bt709 -colorspace bt709 -an cam30_264_rot$r.mp4
  [ -f cam30_265_rot$r.mp4 ] || ffmpeg -v error -y -i cam_rot$r.mp4 -vf "setpts=N/30/TB" -r 30 -c:v hevc_nvenc -preset p5 -bf 0 -g 60 -b:v 3M -color_primaries bt709 -color_trc bt709 -colorspace bt709 -an cam30_265_rot$r.mp4
done
for f in cam30_264_rot0.mp4 cam30_265_rot0.mp4; do echo "$f $(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height,r_frame_rate,nb_frames -of csv=p=0 $f)"; done
