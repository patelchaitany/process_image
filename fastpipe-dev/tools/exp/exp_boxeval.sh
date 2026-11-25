#!/bin/bash
# usage: exp_boxeval.sh <fp dir name> <tag> <scene: dense|normal> [fastpipe args...]
# Skipped-box accuracy on 540 frames: prints one JSON line per detection interval (every frame / 2nd / 3rd).
. "$(dirname "$0")/lock.sh"
fp=$1; tag=$2; scene=$3; shift 3
case $scene in dense) v='/mnt/e/bench/dense45_hevc_rot{r}.mp4';; normal) v='/mnt/e/bench/cam720_265_rot{r}.mp4';; *) echo "scene must be dense or normal"; exit 2;; esac
lock_gpu
$SCP "$(dirname "$0")"/wsl/*.sh laptop:E:/bench/wsl/
timeout 420 $SSH laptop "wsl -d Ubuntu-24.04 -u root -- bash /root/sync.sh exp_boxeval_wsl.sh /root/$fp $tag $v $*"
