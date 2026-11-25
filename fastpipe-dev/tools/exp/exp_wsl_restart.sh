#!/bin/bash
# Restart WSL after an out-of-memory crash ("The specified network name is no longer available").
. "$(dirname "$0")/lock.sh"
lock_gpu
timeout 180 $SSH laptop "wsl --shutdown & wsl -d Ubuntu-24.04 -u root -- bash -c 'uptime; nvidia-smi --query-gpu=memory.used --format=csv,noheader'" 2>&1 | grep -v "^ERROR: Input"
