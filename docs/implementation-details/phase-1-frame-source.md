# Phase 1: Frame Source (FFmpeg) — Implementation Details

This document describes **Phase 1**: decoding MP4 files and RTSP streams into contiguous **BGR24** frames on the CPU, exposed through a small abstract **`FrameSource`** API. It aligns with the design spec: [Face Detection & Recognition Pipeline Design](../superpowers/specs/2026-05-02-face-detection-recognition-pipeline-design.md).

---

## What Was Implemented

### Files, classes, and functions

| File | Role |
|------|------|
| `src/frame_source/frame_source.h` | Defines **`Frame`** (pixel payload + geometry + monotonic **`frame_index`**) and abstract **`FrameSource`** (`open`, `read`, `close`, geometry, FPS, `is_open`). |
| `src/frame_source/ffmpeg_source.h` | Concrete **`FFmpegSource`** holding FFmpeg contexts and decode state. |
| `src/frame_source/ffmpeg_source.cpp` | Implements **`FFmpegSource::open`**, **`read`**, **`decode_next_frame`**, **`close`**, accessors, and destructor. |
| `tests/test_frame_source.cpp` | GoogleTest coverage for invalid path, idempotent **`close`**, **`read`** before **`open`**, and a **disabled** integration test for a sample video. |

**`FFmpegSource` private members** (resources): `AVFormatContext*`, `AVCodecContext*`, `SwsContext*`, two `AVFrame*` (decoded and BGR scratch), `AVPacket*`, video stream index, frame counter, **`opened_`** flag.

**Public surface**: same as **`FrameSource`** — callers depend on the interface; **`FFmpegSource`** is the FFmpeg-backed implementation for both file paths and RTSP URLs (FFmpeg treats them uniformly at `avformat_open_input`).

---

## Architecture Decisions

### 1. FFmpeg API: send/receive (new) vs legacy decode

**Decision:** Use **`av_read_frame` → `avcodec_send_packet` → `avcodec_receive_frame`** (and **`av_packet_unref`** after use).

**Reasoning:**

- This is the **current** libavcodec decode loop. The older one-shot **`avcodec_decode_video2`**-style API is deprecated; send/receive correctly handles **buffering**, **B-frames**, and **EAGAIN** (decoder needs more packets before it can output a frame).
- **`decode_next_frame`** loops: read packets until one belongs to the video stream; send it; **`receive_frame`** until **`AVERROR(EAGAIN)`** causes another packet read — matching FFmpeg’s documented pattern.

### 2. `Frame` struct: `std::vector<uint8_t>` for `data`

**Decision:** Pixel storage is **`std::vector<uint8_t> data`** with **`width`**, **`height`**, **`channels`** (3), and **`frame_index`**.

**Reasoning:**

- **Ownership and lifetime** are obvious: the vector owns the buffer; no manual `new`/`delete` at call sites.
- **Tight packing**: output rows are copied so **`data`** is **`width × height × 3`** bytes with **no line padding** — simplifies later **`cudaMemcpy`** size calculation and CPU post-processing.
- **Trade-off**: per-frame heap allocation on **`resize`** can be optimized later (reuse buffer, **`reserve`**) if profiling shows allocator pressure; clarity and safety were prioritized for Phase 1.

### 3. Resource cleanup: RAII via destructor + `close()`

**Decision:** **`~FFmpegSource()`** calls **`close()`**. **`close()`** frees in dependency order: BGR buffer behind **`bgr_frame_`**, frames, packet, **`sws`**, codec context, format context; nulls pointers and clears **`opened_`**.

**Reasoning:**

- Guarantees no leak if the caller forgets **`close()`** after **`open()`**.
- **`open()`** starts with **`close()`** so repeated **`open`** is safe and failure paths can **`close()`** partially initialized state.

### 4. RTSP vs MP4: unified `FrameSource` abstraction

**Decision:** No separate RTSP type — **`FFmpegSource::open`** takes **`std::string source`** and passes it to **`avformat_open_input`**.

**Reasoning:**

- FFmpeg’s demuxer layer **abstracts protocol** (file, RTSP, etc.); the decode pipeline (**`AVCodecContext`**, **`sws_scale`**) is identical.
- The rest of the pipeline (GPU upload, Triton) depends only on **`Frame`**, not on transport.

### 5. Color conversion: `sws_scale` to BGR24

**Decision:** After decode, **`sws_getContext`** is created from codec pixel format to **`AV_PIX_FMT_BGR24`** with **`SWS_BILINEAR`**. **`sws_scale`** converts **`av_frame_`** into **`bgr_frame_`**.

**Reasoning:**

- Decoders typically output **YUV** (e.g. YUV420). The design targets **BGR** for CUDA preprocessing and consistency with common CV stacks.
- **`SWS_BILINEAR`** is a reasonable quality/performance tradeoff for chroma upsampling vs. nearest neighbor.

### 6. Line-by-line copy: stride vs width

**Decision:** After **`sws_scale`**, **`read`** copies row-by-row with **`memcpy`**, using **`bgr_frame_->linesize[0]`** as the source stride and **`w * 3`** as contiguous row length.

**Reasoning:**

- FFmpeg **may pad** rows for SIMD alignment; **`linesize[0]`** can be **greater than** `width * 3`.
- Writing directly into a flat **`std::vector`** without this loop would **ignore padding** and misalign rows. The loop produces **tightly packed** BGR for downstream code.

---

## Reasoning Summary

| Topic | Choice | Why |
|--------|--------|-----|
| Decode API | send/receive | Correct behavior, maintained API |
| Pixel container | `std::vector` | RAII, contiguous output after per-row copy |
| Cleanup | Destructor → `close` | Exception-safe, hard to misuse |
| Transport | Single `FFmpegSource` | FFmpeg demuxer hides RTSP vs file |
| YUV → BGR | `sws_scale` + bilinear flag | Standard, keeps decode path on CPU per spec |
| Stride | Per-row `memcpy` | Guarantees `width*3*height` layout |

---

## Key Code Snippets

**Abstract source and `Frame`:**

```7:28:src/frame_source/frame_source.h
struct Frame {
    std::vector<uint8_t> data;  // BGR24 interleaved
    int width = 0;
    int height = 0;
    int channels = 3;
    uint64_t frame_index = 0;

    size_t size_bytes() const { return data.size(); }
    bool empty() const { return data.empty(); }
};

class FrameSource {
public:
    virtual ~FrameSource() = default;
    virtual bool open(const std::string& source) = 0;
    virtual bool read(Frame& frame) = 0;
    virtual void close() = 0;
    virtual int width() const = 0;
    virtual int height() const = 0;
    virtual double fps() const = 0;
    virtual bool is_open() const = 0;
};
```

**Destructor and `sws_scale` + packed copy:**

```4:6:src/frame_source/ffmpeg_source.cpp
FFmpegSource::~FFmpegSource() {
    close();
}
```

```81:106:src/frame_source/ffmpeg_source.cpp
bool FFmpegSource::read(Frame& frame) {
    if (!opened_) return false;

    if (!decode_next_frame()) return false;

    sws_scale(sws_ctx_,
        av_frame_->data, av_frame_->linesize, 0, codec_ctx_->height,
        bgr_frame_->data, bgr_frame_->linesize);

    int w = codec_ctx_->width;
    int h = codec_ctx_->height;
    size_t row_bytes = static_cast<size_t>(w) * 3;

    frame.width = w;
    frame.height = h;
    frame.channels = 3;
    frame.frame_index = frame_count_++;
    frame.data.resize(static_cast<size_t>(w) * h * 3);

    for (int y = 0; y < h; ++y) {
        memcpy(frame.data.data() + y * row_bytes,
               bgr_frame_->data[0] + y * bgr_frame_->linesize[0],
               row_bytes);
    }

    return true;
}
```

**Send/receive decode loop:**

```109:128:src/frame_source/ffmpeg_source.cpp
bool FFmpegSource::decode_next_frame() {
    while (true) {
        int ret = av_read_frame(fmt_ctx_, pkt_);
        if (ret < 0) return false;  // EOF or error

        if (pkt_->stream_index != video_stream_idx_) {
            av_packet_unref(pkt_);
            continue;
        }

        ret = avcodec_send_packet(codec_ctx_, pkt_);
        av_packet_unref(pkt_);
        if (ret < 0) return false;

        ret = avcodec_receive_frame(codec_ctx_, av_frame_);
        if (ret == AVERROR(EAGAIN)) continue;
        if (ret < 0) return false;

        return true;
    }
}
```

---

## Verification Approach

1. **Unit tests** (`tests/test_frame_source.cpp`):
   - **`OpenInvalidPath`**: **`open`** fails and **`is_open()`** is false.
   - **`CloseWithoutOpen`**: **`close`** does not crash.
   - **`ReadWithoutOpen`**: **`read`** returns false.
2. **Integration (optional):** Enable **`FrameSourceTest.OpenAndReadVideo`** when `test_data/sample.mp4` is present; asserts dimensions, FPS, first frame index `0`, and expected **`data.size()`**.
3. **Manual / CI:** Point **`open`** at an RTSP URL and verify **`read`** returns frames (not covered by default tests due to environment dependence).
4. **Spec alignment:** Decode substage timings in the design doc (**`av_read_frame`**, **`avcodec_*`**, **`sws_scale`**) map directly to this implementation for Stage 1 of the pipeline.
