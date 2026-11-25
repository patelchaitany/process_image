// Headless end-to-end check of the live path, without a browser or a camera.
// Sends one Annex-B H.264 file (with access unit delimiters) as N cameras at 30 fps and checks what comes back.
//   node test_client.mjs <file.h264> [seconds=12] [cams=2] [ws://localhost:8765/ws]
import fs from 'node:fs';

const [file, secsArg, camsArg, urlArg] = process.argv.slice(2);
const SECS = +(secsArg || 12), CAMS = +(camsArg || 2), URL_ = urlArg || 'ws://localhost:8765/ws';
const W = 1280, H = 720;
const buf = fs.readFileSync(file);

// split at access unit delimiters (NAL type 9)
const cuts = [];
for (let i = 0; i + 4 < buf.length; i++) {
  if (buf[i] === 0 && buf[i + 1] === 0 && (buf[i + 2] === 1 || (buf[i + 2] === 0 && buf[i + 3] === 1))) {
    const off = buf[i + 2] === 1 ? i + 3 : i + 4;
    if ((buf[off] & 0x1f) === 9) cuts.push(i);
    i = off;
  }
}
const aus = cuts.map((s, k) => buf.subarray(s, k + 1 < cuts.length ? cuts[k + 1] : buf.length));
console.error(`${aus.length} access units, ${(buf.length / aus.length / 1024).toFixed(1)} KB each on average`);

const zones = [
  { id: 1, pts: [[0, 0], [0.5, 0], [0.5, 1], [0, 1]] },      // left half
  { id: 2, pts: [[0.5, 0], [1, 0], [1, 1], [0.5, 1]] },      // right half
  { id: 3, pts: [[0.4, 0.3], [0.6, 0.3], [0.6, 0.7], [0.4, 0.7]] },   // box in the middle, drawn last = on top
];
const expectZone = (cx, cy) => {
  const x = (Math.floor(cx) + 0.5) / W, y = (Math.floor(cy) + 0.5) / H;
  if (x > 0.4 && x < 0.6 && y > 0.3 && y < 0.7) return 3;
  return x < 0.5 ? 1 : 2;
};

const sent = Array.from({ length: CAMS }, () => new Map());
const st = Array.from({ length: CAMS }, () => ({ frames: 0, boxes: 0, lat: [], gpu: [], lastSeq: -1, gaps: 0, zoneOk: 0, zoneBad: 0,
  zoneCount: {}, named: 0, known: 0, tracks: new Set() }));
const events = [];
let ready = false, t0 = 0, seq = 0, enrollTrack = null, enrolledAt = null, namedAfter = [0, 0], readyAt = 0;
const tStart = performance.now();

const ws = new WebSocket(URL_);
ws.binaryType = 'arraybuffer';
ws.onopen = () => ws.send(JSON.stringify({ op: 'start', cams: CAMS, zoom: +(process.env.ZOOM || 1) }));   // ZOOM=0.5: close-up faces
ws.onerror = (e) => { console.error('websocket error', e.message || e); process.exit(2); };
ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  const now = performance.now();
  if (m.event) {
    if (m.event === 'log') { console.error('  [fastpipe]', m.line); return; }
    events.push(m);
    console.error('event', JSON.stringify(m));
    if (m.event === 'ready') {
      ready = true; readyAt = now; t0 = now;
      for (let c = 0; c < CAMS; c++) ws.send(JSON.stringify({ op: 'zones', cam: c, zones }));
      pump();
      setTimeout(finish, SECS * 1000);
    }
    if (m.event === 'enrolled') enrolledAt = now;
    return;
  }
  const s = st[m.cam];
  s.frames++;
  if (s.lastSeq >= 0 && m.seq !== s.lastSeq + 1) s.gaps++;
  s.lastSeq = m.seq;
  const ts = sent[m.cam].get(m.seq);
  if (ts !== undefined) { s.lat.push(now - ts); sent[m.cam].delete(m.seq); }
  if (m.det) s.gpu.push(m.ms);
  for (const [x1, y1, x2, y2, tid, known, label, zone] of m.b) {
    s.boxes++;
    s.tracks.add(tid);
    s.known += known;
    if (label >= 0) { s.named++; if (enrolledAt) namedAfter[m.cam]++; }
    s.zoneCount[zone] = (s.zoneCount[zone] || 0) + 1;
    // zones take a moment to reach the GPU: check from 1 s after start
    if (now - readyAt > 1000) (expectZone((x1 + x2) / 2, (y1 + y2) / 2) === zone ? s.zoneOk++ : s.zoneBad++);
    if (m.cam === 0 && enrollTrack === null && now - readyAt > 3000 && x2 - x1 > 40) {
      enrollTrack = tid;
      ws.send(JSON.stringify({ op: 'enroll', cam: 0, track: tid, label: 0 }));
      console.error('enrol request: camera 0 track', tid);
    }
  }
};

function pump() {
  if (!ready) return;
  const due = t0 + seq * (1000 / 30);
  const wait = due - performance.now();
  if (wait > 1) { setTimeout(pump, wait - 0.5); return; }
  const au = aus[seq % aus.length];
  for (let c = 0; c < CAMS; c++) {
    const msg = new Uint8Array(8 + au.length);
    const dv = new DataView(msg.buffer);
    dv.setUint8(0, c);
    dv.setUint8(1, seq % aus.length === 0 ? 1 : 0);
    dv.setUint32(4, seq, true);
    msg.set(au, 8);
    sent[c].set(seq, performance.now());
    ws.send(msg);
  }
  seq++;
  setImmediate(pump);
}

const pct = (v, p) => { if (!v.length) return null; const a = [...v].sort((x, y) => x - y); return +a[Math.min(a.length - 1, Math.floor(p / 100 * a.length))].toFixed(1); };

function finish() {
  ready = false;
  ws.send(JSON.stringify({ op: 'stop' }));
  const out = {
    start_s: +((readyAt - tStart) / 1000).toFixed(1), sent_frames_per_cam: seq, enrol_track: enrollTrack,
    enrolled_after_ms: enrolledAt ? +(enrolledAt - readyAt).toFixed(0) : null,
    cams: st.map((s, c) => ({
      cam: c, result_frames: s.frames, missing: seq - s.frames, seq_gaps: s.gaps, boxes_per_frame: +(s.boxes / Math.max(1, s.frames)).toFixed(2),
      tracks: s.tracks.size, latency_ms_p50: pct(s.lat, 50), latency_ms_p95: pct(s.lat, 95), latency_ms_max: pct(s.lat, 100),
      pipeline_ms_p50: pct(s.gpu, 50), pipeline_ms_p95: pct(s.gpu, 95),
      zone_ok: s.zoneOk, zone_wrong: s.zoneBad, zone_counts: s.zoneCount,
      boxes_recognised_pct: +(100 * s.known / Math.max(1, s.boxes)).toFixed(1), boxes_named: s.named, named_after_enrol: namedAfter[c],
    })),
    events: events.map((e) => e.event),
  };
  console.log(JSON.stringify(out, null, 1));
  setTimeout(() => process.exit(0), 800);
}
