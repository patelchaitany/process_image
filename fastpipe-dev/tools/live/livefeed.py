#!/usr/bin/env python3
"""(WSL) feed one Annex-B file to fastpipe --live directly (no bridge) and print what comes back.
usage: livefeed.py <file.h264> <det-zoom> [seconds]"""
import os, struct, subprocess, sys, threading, time, json
f, zoom, secs = sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 3
buf = open(f, 'rb').read()
cuts = []
i = 0
while i + 4 < len(buf):
    if buf[i] == 0 and buf[i+1] == 0 and (buf[i+2] == 1 or (buf[i+2] == 0 and buf[i+3] == 1)):
        off = i + 3 if buf[i+2] == 1 else i + 4
        if buf[off] & 0x1f == 9: cuts.append(i)
        i = off
    else:
        i += 1
aus = [buf[s:(cuts[k+1] if k + 1 < len(cuts) else len(buf))] for k, s in enumerate(cuts)]
os.makedirs('/tmp/fpdbg', exist_ok=True)
p = '/tmp/fpdbg/cam0.h264'
if os.path.exists(p): os.unlink(p)
os.mkfifo(p)
env = dict(os.environ, FP_LIVE_DEBUG='1')
proc = subprocess.Popen(['/root/fastpipe/build/fastpipe', '--det', '/root/fastpipe/engines/det_r384_fp16.plan', '--rec',
    '/root/fastpipe/engines/' + os.environ.get('FP_REC', 'rec_edgeface_s_fp16.plan'), '--live', '/tmp/fpdbg/cam{r}.h264', '--counts', '1', '--analyze-fps', '10',
    '--min-batch', '1', '--max-wait-ms', '0', '--recheck-s', '1', '--det-zoom', zoom, '--out', '/tmp/fpdbg/out'],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, cwd='/root/fastpipe')
lines, errs = [], []
def rd_err():
    for l in proc.stderr:
        l = l.decode('utf-8', 'replace').rstrip()
        if '[dbg]' in l or '[live]' in l or 'rror' in l: errs.append(l)
threading.Thread(target=rd_err, daemon=True).start()
ready = threading.Event()
def rd_out():
    for l in proc.stdout:
        l = l.decode().strip()
        if l.startswith('{"event":"ready"'): ready.set()
        elif l.startswith('{"cam"'): lines.append(json.loads(l))
threading.Thread(target=rd_out, daemon=True).start()
ready.wait(60)
with open(p, 'wb', buffering=0) as w:
    t0 = time.time()
    n = int(secs * 30)
    for s in range(n):
        d = aus[s % len(aus)]
        w.write(b'FPAU' + struct.pack('<IQ', len(d), s) + d)
        dt = t0 + (s + 1) / 30 - time.time()
        if dt > 0: time.sleep(dt)
    time.sleep(0.5)
    proc.stdin.write(b'Q\n'); proc.stdin.flush()
proc.wait(timeout=20)
nb = [len(l['b']) for l in lines]
print(os.path.basename(f), 'zoom', zoom, ': result frames', len(lines), 'boxes/frame', round(sum(nb) / max(1, len(nb)), 2), 'exit', proc.returncode)
for e in errs[:6]: print('   ', e)
