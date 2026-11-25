#!/bin/bash
# Start the camera test page.
#   1. copies the bridge and the page to the laptop
#   2. starts the bridge there, next to the GPU pipeline (inside WSL)
#   3. tunnels it to this Mac and opens http://localhost:8765
# Ctrl-C stops the bridge and the pipeline. Nothing is opened on the laptop's network: the page is reached only
# through the ssh connection.
#   LAPTOP=<address of the GPU machine>  LAPTOP_USER=<ssh user>  PORT=8765  NO_OPEN=1 (do not open the browser)
#   FP_REC=rec_r50_fp16.plan (another recogniser engine; default EdgeFace-S)
# Put your own values in live/local.env (not in git) to avoid typing them:  LAPTOP=...  LAPTOP_USER=...
DIR="$(cd "$(dirname "$0")" && pwd)"
[ -f "$DIR/local.env" ] && . "$DIR/local.env"
LAPTOP="${LAPTOP:?set LAPTOP to the address of the GPU machine, or put it in live/local.env}"
LUSER="${LAPTOP_USER:-$USER}"
PORT="${PORT:-8765}"
OPTS=(-i "${LAPTOP_KEY:-$HOME/.ssh/id_ed25519}" -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o LogLevel=ERROR)
# A pinned host key beside this script (live/laptop_known_hosts, not in git) is used when present, so the machine's
# key never has to go into ~/.ssh/known_hosts.
[ -f "$DIR/laptop_known_hosts" ] && OPTS+=(-o "UserKnownHostsFile=$DIR/laptop_known_hosts" -o StrictHostKeyChecking=yes)
set -e
ssh "${OPTS[@]}" "$LUSER@$LAPTOP" "if not exist E:\\bench\\fastpipe-dev\\live mkdir E:\\bench\\fastpipe-dev\\live"
scp -q "${OPTS[@]}" "$DIR/bridge.py" "$DIR/index.html" "$DIR/wsl_run.sh" "$LUSER@$LAPTOP:E:/bench/fastpipe-dev/live/"
echo "Page: http://localhost:$PORT   (Ctrl-C to stop)"
# the page needs Chrome's video encoder; fall back to the default browser if Chrome is not installed
if [ -z "$NO_OPEN" ]; then (sleep 2; open -a "Google Chrome" "http://localhost:$PORT" 2>/dev/null || open "http://localhost:$PORT") & fi
exec ssh "${OPTS[@]}" -o ExitOnForwardFailure=yes -L "$PORT:127.0.0.1:$PORT" "$LUSER@$LAPTOP" \
  "wsl -d Ubuntu-24.04 -u root -e env FP_PORT=$PORT ${FP_REC:+FP_REC=$FP_REC} bash /mnt/e/bench/fastpipe-dev/live/wsl_run.sh"
