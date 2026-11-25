#!/bin/bash
cd /mnt/e/bench
for r in 0 1 2 3; do
  [ -f cam720_264_rot$r.mp4 ] || ffmpeg -v error -y -i cam_rot$r.mp4 -vf "setpts=N/30/TB,scale=1280:720:flags=bicubic" -r 30 -c:v h264_nvenc -preset p5 -profile:v high -bf 0 -g 60 -b:v 2500k -color_primaries bt709 -color_trc bt709 -colorspace bt709 -an cam720_264_rot$r.mp4
  [ -f cam720_265_rot$r.mp4 ] || ffmpeg -v error -y -i cam_rot$r.mp4 -vf "setpts=N/30/TB,scale=1280:720:flags=bicubic" -r 30 -c:v hevc_nvenc -preset p5 -bf 0 -g 60 -b:v 1800k -color_primaries bt709 -color_trc bt709 -colorspace bt709 -an cam720_265_rot$r.mp4
done
for f in cam720_264_rot0.mp4 cam720_265_rot0.mp4; do echo "$f $(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height,r_frame_rate,nb_frames -of csv=p=0 $f)"; done
