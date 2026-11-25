#!/bin/bash
# (WSL, on the laptop) stop any earlier test session, then run the bridge. Started by start.sh over ssh.
pkill -f "live/bridge.py" 2>/dev/null
pkill -f "fastpipe .*--live" 2>/dev/null
sleep 0.3
export FP_WATCH_STDIN=1
exec python3 /mnt/e/bench/fastpipe/live/bridge.py
