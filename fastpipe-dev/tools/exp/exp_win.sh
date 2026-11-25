#!/bin/bash
# usage: exp_win.sh "<Windows cmd.exe command>"   runs it on the laptop while holding the GPU lock
. "$(dirname "$0")/lock.sh"
lock_gpu
timeout 900 $SSH laptop "$1"
