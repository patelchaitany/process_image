# Phase 13: Per-detection bbox CSV logging and modular result writers

This document covers the modular `ResultWriter` system, the non-blocking `CsvResultWriter` for per-detection bbox logging, the `ConsoleResultWriter` for stdout output, and the `draw_bboxes.py` visualization tool.

## What was implemented

### Files, classes, and functions

| File | Role |
|------|------|
| `src/output/result_writer.h` | Abstract `ResultWriter` interface (`open`, `write`, `close`) and `FrameResult` struct |
| `src/output/csv_result_writer.h/.cpp` | `CsvResultWriter` — non-blocking async CSV writer with ring buffer + background thread |
| `src/output/console_result_writer.h/.cpp` | `ConsoleResultWriter` — stdout printing (extracted from old `ResultHandler`) |
| `src/pipeline/result_handler.h/.cpp` | `ResultHandler` — dispatcher that fans out to all registered `ResultWriter` instances |
| `tools/draw_bboxes.py` | Python tool: reads CSV + MP4, draws bboxes, re-encodes to H.264 via ffmpeg |

### FrameResult struct (result_writer.h)

```cpp
struct FrameResult {
    uint64_t frame_id = 0;
    std::string timestamp_utc;
    std::string source_id;
    std::vector<Detection> detections;
    std::vector<MatchResult> matches;
};
```

Previously `FrameResult` was defined in `result_handler.h` without timestamp or source fields. It was moved to `result_writer.h` and enriched so every writer has full context.

### CSV schema (CsvResultWriter)

One row per detection per frame:

```
frame_id,timestamp_utc,source_id,det_idx,x1,y1,x2,y2,bbox_width,bbox_height,det_confidence,identity,face_id,match_confidence
42,2026-05-03T10:00:01.234Z,rtsp://cam1,0,120.5,80.3,250.2,310.7,129.7,230.4,0.9312,john_doe,7,0.8451
42,2026-05-03T10:00:01.234Z,rtsp://cam1,1,500.0,100.0,620.0,280.0,120.0,180.0,0.8100,unknown,-1,0.0000
```

Frames with no detections still emit one row with `identity=no_detection` so downstream consumers can distinguish "no faces" from "frame was dropped".

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--bbox-csv <path>` | `./output/detections.csv` | Per-detection CSV output path |
| `--no-console` | off | Suppress per-frame stdout output |

## Architecture decisions

### Plugin-based writer pattern

```
Pipeline
  └── ResultHandler (dispatcher)
        ├── ConsoleResultWriter   (always registered)
        ├── CsvResultWriter       (registered if --bbox-csv is set)
        └── (future writers: JsonResultWriter, WebSocketWriter, DatabaseWriter, ...)
```

The `ResultWriter` interface is intentionally minimal — three virtual methods:

- `open()` — one-time initialization (create directories, open files, start threads)
- `write(const FrameResult&)` — called once per frame from the pipeline thread
- `close()` — flush and release resources

This was chosen over alternatives:

1. **Callback/lambda approach**: Less discoverable, harder to unit-test, no lifecycle management.
2. **Event bus / observer**: Overkill for a linear pipeline with 2-3 sinks. Adds indirection with no benefit.
3. **Single monolithic handler**: The original design. Broke the open-closed principle — adding CSV output would have required modifying `ResultHandler` directly.

The plugin pattern means adding a new output format (e.g., JSON-lines, gRPC stream, WebSocket push) requires only:
1. Create a new `.h/.cpp` implementing `ResultWriter`
2. Register it in `Pipeline::init()` with `result_handler_->addWriter(...)`

Zero changes to the pipeline, handler, or any existing writer.

### Non-blocking async writes (CsvResultWriter)

File I/O on the pipeline's hot path would add 0.5-2ms per frame (SSD) or 10ms+ (NFS/network). The `CsvResultWriter` follows the same pattern as `MetricsLogger`:

```
Pipeline thread                    Writer thread
     │                                  │
     │  write(FrameResult)              │
     │  ──push to RingBuffer──►         │
     │  (returns immediately)           │
     │                                  │  pop batch from buffer
     │                                  │  writeEntry() × N
     │                                  │  flush()
     │                                  │  rotateIfNeeded()
     │                                  │  sleep(flushIntervalMs)
     │                                  │
```

- **Ring buffer**: `RingBuffer<FrameResult, 4096>` — lock-free SPSC, same template as `MetricsLogger`. Capacity 4096 covers ~136 seconds at 30fps before drops.
- **Background thread**: Wakes every 500ms, drains all buffered entries, flushes once, checks rotation.
- **Shutdown drain**: `close()` sets `running_=false`, joins the thread, then drains any remaining entries so no data is lost.
- **File rotation**: When file size exceeds `rotateSizeMb`, the current file is renamed with a timestamp suffix and a fresh file with a new header is opened.

### ConsoleResultWriter

Extracted from the original `ResultHandler::handle()` implementation. Supports a `verbose` flag:
- `verbose=true` (default): prints per-detection bbox details
- `verbose=false`: prints only the frame summary line

This is always registered and cannot be disabled (only quieted with `--no-console`).

### FrameResult stamping

Every `FrameResult` is created via `Pipeline::make_result(frame_id)` which stamps `timestamp_utc` and `source_id` from the running config. This ensures all writers get consistent metadata without each writer having to know about the pipeline config.

## draw_bboxes.py visualization tool

### Workflow

1. **Parse CSV**: Groups detection rows by `frame_id` into a dict
2. **Read video frame-by-frame**: Maps each frame index to its detections
3. **Draw overlays**: Bounding box rectangle, filled label background with identity + confidence
4. **Write to temp MJPEG AVI** via OpenCV (fast, reliable across platforms)
5. **Re-encode to H.264 MP4** via ffmpeg (`libx264`, `yuv420p`, `+faststart`)
6. **Cleanup**: Remove temp file

### Why the two-stage encode

OpenCV's `VideoWriter` with `mp4v` fourcc produces MPEG-4 Part 2 files that macOS QuickTime and many web players refuse to play. OpenCV's H.264 support depends on the build (many pip installs lack it). Writing MJPEG first and piping through ffmpeg guarantees:

- Universal playback (QuickTime, VLC, browsers, mobile)
- `+faststart` moov atom placement for streaming
- Graceful fallback: if ffmpeg is missing, the MJPEG AVI is preserved with a warning

### Color scheme

- Known identities get a consistent color from a 10-color palette, keyed on `face_id % 10`
- Unknown / unmatched faces are drawn in red
- Label text color auto-adapts (black on bright backgrounds, white on dark) for readability

## Verification

- `draw_bboxes.py` successfully processed a 419-frame video with 297 detection frames
- Output H.264 MP4 plays correctly in QuickTime, VLC, and browser `<video>` tags
- Non-blocking `write()` adds ~0.001ms to the pipeline frame loop (ring buffer push only)
