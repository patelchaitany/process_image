#!/bin/bash
# usage: exp_build.sh <local fastpipe copy (contains src/ and CMakeLists.txt)> <name>
# Builds the copy on the laptop as /root/fp-<name>/build/fastpipe (use "fp-<name>" as the fp dir name afterwards).
. "$(dirname "$0")/lock.sh"
src=$1; name=$2
[ -d "$src/src" ] || { echo "no src/ in $src"; exit 2; }
lock_gpu
tar -czf $S/exp-$name.tgz -C "$src" src CMakeLists.txt
$SCP $S/exp-$name.tgz laptop:E:/bench/exp-$name.tgz
$SCP "$(dirname "$0")"/wsl/*.sh laptop:E:/bench/wsl/
timeout 900 $SSH laptop "(if not exist E:\bench\exp\\$name mkdir E:\bench\exp\\$name) & tar -xzf E:\bench\exp-$name.tgz -C E:\bench\exp\\$name && del E:\bench\exp-$name.tgz && wsl -d Ubuntu-24.04 -u root -- bash /root/sync.sh exp_build_wsl.sh $name"
