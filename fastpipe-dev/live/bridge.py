#!/usr/bin/env python3
"""Bridge between the browser test page and the GPU pipeline (runs in WSL next to fastpipe; standard library only).

Browser --WebSocket--> bridge --one pipe per camera--> fastpipe --stdout JSON lines--> bridge --WebSocket--> browser

  GET /            the test page (index.html)
  GET /clips       JSON list of test clips on the laptop
  GET /clip/<name> one test clip (byte ranges supported)
  GET /ws          WebSocket
      text   {"op":"start","cams":N,"zoom":0.5} | {"op":"zones","cam":i,"zones":[{"id":1,"pts":[[x,y],...]}]}
             {"op":"enroll","cam":i,"track":t,"label":l} | {"op":"stop"}
      binary [u8 cam][u8 key][u16 0][u32 frame number] + one H.264 access unit (Annex B)
      back   fastpipe's result and event lines, unchanged, plus {"event":"log"|"stopped"|"error",...}
"""
import base64
import hashlib
import json
import os
import queue
import re
import struct
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("FP_PORT", "8765"))
FASTPIPE = os.environ.get("FP_BIN", "/root/fastpipe/build/fastpipe")
ENGINES = os.environ.get("FP_ENGINES", "/root/fastpipe/engines")
REC = os.environ.get("FP_REC", "rec_edgeface_s_fp16.plan")      # recogniser engine; its .thr file sets the match threshold
CLIPS = os.environ.get("FP_CLIPS", "/mnt/e/bench")
PIPES = "/tmp/fplive"
MAX_CAMS = 4
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


class Session:
    """One running fastpipe process and its camera pipes. Only one exists at a time (one GPU)."""

    def __init__(self, cams, send, zoom=1.0):
        self.cams = cams
        self.send = send            # send(text) to the browser
        self.alive = True
        self.ready = False
        self.queues = [queue.Queue() for _ in range(cams)]
        os.makedirs(PIPES, exist_ok=True)
        self.paths = []
        for i in range(cams):
            p = f"{PIPES}/cam{i}.h264"
            if os.path.exists(p):
                os.unlink(p)
            os.mkfifo(p)
            self.paths.append(p)
        args = [FASTPIPE,
                "--det", f"{ENGINES}/det_r384_fp16.plan", "--rec", f"{ENGINES}/{REC}",
                "--live", f"{PIPES}/cam{{r}}.h264", "--live-size", "1280x720", "--live-fps", "30",
                "--counts", str(cams), "--analyze-fps", "10",
                # two cameras never fill a batch: launch a tick as soon as a frame is ready
                "--min-batch", "1", "--max-wait-ms", "0",
                # the GPU is nearly idle here, so re-check names every second (enrolment shows up fast)
                "--recheck-s", "1",
                # close-up cameras: shrink the picture for the detector (it misses faces that fill the frame)
                "--det-zoom", f"{zoom:.2f}",
                "--out", "/tmp/fplive/out", "--tag", "live"]
        log("starting", " ".join(args))
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     cwd=os.path.dirname(os.path.dirname(FASTPIPE)))
        self.stdin_lock = threading.Lock()
        threading.Thread(target=self._stdout, daemon=True).start()
        threading.Thread(target=self._stderr, daemon=True).start()

    def _stdout(self):
        for raw in self.proc.stdout:
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            if not self.ready and line.startswith('{"event":"ready"'):
                # fastpipe has opened its ends of the pipes: connect ours, then tell the browser to start sending
                for i in range(self.cams):
                    threading.Thread(target=self._writer, args=(i,), daemon=True).start()
                self.ready = True
            self.send(line)
        code = self.proc.wait()
        log("fastpipe exited", code)
        if self.alive:
            self.alive = False
            self.send(json.dumps({"event": "stopped", "code": code}))

    def _stderr(self):
        for raw in self.proc.stderr:
            line = raw.decode("utf-8", "replace").rstrip()
            if line:
                log("[fastpipe]", line)
                self.send(json.dumps({"event": "log", "line": line[:300]}))

    def _writer(self, i):
        try:
            with open(self.paths[i], "wb", buffering=0) as f:
                while self.alive:
                    try:
                        seq, data = self.queues[i].get(timeout=0.5)
                    except queue.Empty:
                        continue
                    f.write(b"FPAU" + struct.pack("<IQ", len(data), seq) + data)
        except OSError as e:
            if self.alive:
                log("camera pipe", i, "closed:", e)

    def frame(self, cam, seq, data):
        if self.ready and 0 <= cam < self.cams:
            self.queues[cam].put((seq, data))

    def control(self, line):
        if not self.alive:
            return
        try:
            with self.stdin_lock:
                self.proc.stdin.write((line + "\n").encode())
                self.proc.stdin.flush()
        except (OSError, ValueError):
            pass

    def stop(self):
        if not self.alive:
            return
        self.control("Q")
        self.alive = False
        try:
            self.proc.stdin.close()
        except (OSError, ValueError):
            pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        log("session stopped")


session = None
session_lock = threading.Lock()


def stop_session(reason=None):
    global session
    with session_lock:
        s, session = session, None
    if s:
        if reason and s.alive:      # tell the page that owned it why its boxes stop
            s.send(json.dumps({"event": "stopped", "code": 0, "reason": reason}))
        s.stop()


class WS:
    """Minimal RFC 6455 server side on an accepted connection."""

    def __init__(self, rfile, conn):
        self.rfile = rfile
        self.conn = conn
        self.lock = threading.Lock()
        self.open = True

    def _read(self, n):
        b = self.rfile.read(n)
        if b is None or len(b) < n:
            raise ConnectionError("closed")
        return b

    def recv(self):
        """Returns (opcode, payload) of the next complete message: 1 = text, 2 = binary. None when closed."""
        op0, parts = None, []
        while True:
            h = self._read(2)
            fin, op = h[0] & 0x80, h[0] & 0x0F
            masked, n = h[1] & 0x80, h[1] & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._read(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._read(8))[0]
            if n > (64 << 20):
                raise ConnectionError("frame too large")
            key = self._read(4) if masked else None
            data = self._read(n) if n else b""
            if key and n:
                k = (key * (n // 4 + 1))[:n]
                data = (int.from_bytes(data, "little") ^ int.from_bytes(k, "little")).to_bytes(n, "little")
            if op == 8:
                return None
            if op == 9:
                self._send(10, data)
                continue
            if op == 10:
                continue
            if op:
                op0 = op
            parts.append(data)
            if fin:
                return op0, b"".join(parts) if len(parts) > 1 else parts[0]

    def _send(self, op, data):
        n = len(data)
        if n < 126:
            h = bytes([0x80 | op, n])
        elif n < 65536:
            h = bytes([0x80 | op, 126]) + struct.pack(">H", n)
        else:
            h = bytes([0x80 | op, 127]) + struct.pack(">Q", n)
        with self.lock:
            if self.open:
                try:
                    self.conn.sendall(h + data)
                except OSError:
                    self.open = False

    def send_text(self, s):
        self._send(1, s.encode())


def zones_line(msg):
    cam = int(msg["cam"])
    zs = [z for z in msg.get("zones", []) if len(z.get("pts", [])) >= 3][:255]
    out = ["Z", str(cam), str(len(zs))]
    for z in zs:
        pts = z["pts"][:256]
        out += [str(int(z["id"])), str(len(pts))]
        for x, y in pts:
            out += [f"{min(max(float(x), 0.0), 1.0):.5f}", f"{min(max(float(y), 0.0), 1.0):.5f}"]
    return " ".join(out)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _file(self, path, ctype):
        try:
            size = os.path.getsize(path)
            f = open(path, "rb")
        except OSError:
            self.send_error(404)
            return
        with f:
            start, end = 0, size - 1
            m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1))
                    end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
                else:
                    start = max(0, size - int(m.group(2)))
                if start > end:
                    self.send_error(416)
                    return
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            f.seek(start)
            left = end - start + 1
            try:
                while left > 0:
                    b = f.read(min(left, 1 << 18))
                    if not b:
                        break
                    self.wfile.write(b)
                    left -= len(b)
            except OSError:
                pass

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/ws":
            return self._ws()
        if path in ("/", "/index.html"):
            return self._file(os.path.join(HERE, "index.html"), "text/html; charset=utf-8")
        if path == "/clips":
            try:
                names = sorted(n for n in os.listdir(CLIPS) if re.fullmatch(r"[A-Za-z0-9_.-]+\.mp4", n))
            except OSError:
                names = []
            body = json.dumps(names).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        m = re.fullmatch(r"/clip/([A-Za-z0-9_.-]+\.mp4)", path)
        if m:
            return self._file(os.path.join(CLIPS, m.group(1)), "video/mp4")
        self.send_error(404)

    def _ws(self):
        global session
        key = self.headers.get("Sec-WebSocket-Key")
        if not key or "websocket" not in self.headers.get("Upgrade", "").lower():
            self.send_error(400)
            return
        # only pages served by this bridge may drive the pipeline
        origin = self.headers.get("Origin", "")
        host = self.headers.get("Host", "")
        if origin and origin.split("://")[-1] != host:
            self.send_error(403)
            return
        accept = base64.b64encode(hashlib.sha1((key + WS_GUID).encode()).digest()).decode()
        self.send_response(101)
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept)
        self.end_headers()
        self.close_connection = True
        ws = WS(self.rfile, self.connection)
        log("browser connected")
        mine = None
        try:
            while True:
                m = ws.recv()
                if m is None:
                    break
                op, data = m
                if op == 2:
                    if mine and mine.alive and len(data) > 8:
                        cam, _key, _r, seq = struct.unpack_from("<BBHI", data)
                        mine.frame(cam, seq, data[8:])
                    continue
                msg = json.loads(data.decode())
                what = msg.get("op")
                if what == "start":
                    stop_session("Another page started the pipeline, so this one was stopped.")
                    cams = max(1, min(MAX_CAMS, int(msg.get("cams", 1))))
                    zoom = max(0.2, min(1.0, float(msg.get("zoom", 1.0))))
                    try:
                        mine = Session(cams, ws.send_text, zoom)
                    except OSError as e:
                        ws.send_text(json.dumps({"event": "error", "message": str(e)}))
                        continue
                    with session_lock:
                        session = mine
                elif what == "stop":
                    stop_session()
                    mine = None
                    ws.send_text(json.dumps({"event": "stopped", "code": 0}))
                elif what == "zones" and mine:
                    mine.control(zones_line(msg))
                elif what == "enroll" and mine:
                    mine.control(f"E {int(msg['cam'])} {int(msg['track'])} {int(msg['label'])}")
        except (ConnectionError, OSError, ValueError, KeyError) as e:
            log("browser connection ended:", e)
        finally:
            ws.open = False
            if mine:
                with session_lock:
                    cur = session is mine
                if cur:
                    stop_session()
            log("browser disconnected")


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    srv.daemon_threads = True
    log(f"bridge listening on http://127.0.0.1:{PORT}  (fastpipe: {FASTPIPE})")
    if os.environ.get("FP_WATCH_STDIN"):
        # started over ssh: when the connection goes away (stdin closes), stop the pipeline and exit
        def watch():
            try:
                while sys.stdin.buffer.read(4096):
                    pass
            except OSError:
                pass
            log("launcher went away, stopping")
            stop_session()
            os._exit(0)
        threading.Thread(target=watch, daemon=True).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_session()


if __name__ == "__main__":
    main()
