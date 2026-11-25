# Source this. One laptop, one GPU: every interaction with the laptop happens while holding this lock.
# The laptop is reached through an ssh config holding "Host laptop" (address, user, key, known-hosts file).
# FP_SSH_DIR is the folder with that ssh_config; it also holds the GPU lock.
S=${FP_SSH_DIR:?set FP_SSH_DIR to the folder that holds ssh_config with a "Host laptop" entry}
SSH="ssh -F $S/ssh_config"
SCP="scp -q -F $S/ssh_config"
LOCK=$S/gpu.lock
lock_gpu() {
  while ! mkdir "$LOCK" 2>/dev/null; do
    p=$(cat "$LOCK/pid" 2>/dev/null)
    if [ -n "$p" ] && ! kill -0 "$p" 2>/dev/null; then rm -rf "$LOCK"; continue; fi   # holder died: take over
    sleep 3
  done
  echo $$ > "$LOCK/pid"
  trap 'rm -rf "$LOCK"' EXIT
}
