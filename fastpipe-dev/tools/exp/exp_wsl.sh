#!/bin/bash
# usage: exp_wsl.sh <local bash script> [args...]   runs the script inside WSL (as root) while holding the GPU lock
. "$(dirname "$0")/lock.sh"
f=$1; shift
lock_gpu
$SCP "$f" laptop:E:/bench/wsl/$(basename "$f")
timeout 1500 $SSH laptop "wsl -d Ubuntu-24.04 -u root -- bash /root/sync.sh $(basename "$f") $*"
