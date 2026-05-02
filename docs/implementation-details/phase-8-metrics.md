# Phase 8: Per-frame metrics, ring buffer, and async CSV writer

This document describes the metrics types, the lock-free single-producer/single-consumer (SPSC) queue between the pipeline and writer thread, CSV output and rotation, flush behavior, and how this design aligns with the pipeline design spec’s overhead budget.

## What was implemented

### Files, classes, and functions

| File | Role |
|------|------|
| `src/metrics/frame_metrics.h` | `FrameMetrics` — all per-frame timing and metadata fields (aligned with design spec naming) |
| `src/metrics/ring_buffer.h` | Template `RingBuffer<T, Capacity>` — lock-free SPSC queue |
| `src/metrics/metrics_logger.h` / `metrics_logger.cpp` | `MetricsLogger` — `init`, `shutdown`, `log_frame`, background `writer_thread_func`, `rotate_file_if_needed`, `generate_filename` |

**`FrameMetrics`** holds the full logical schema the product spec calls for: decode and color conversion, H2D, preprocess breakdown fields, both gRPC/YOLO/ArcFace intervals, Triton queue/compute placeholders, NMS breakdown, FAISS stages, result copy, total pipeline time, and `gpu_memory_used_mb`.

**`RingBuffer<FrameMetrics, 128>`** is the bounded queue from the pipeline thread (producer) to the metrics writer (consumer).

**`MetricsLogger`:**

- `init(output_dir, flush_interval_ms = 1000, rotate_size_mb = 100)` — creates `output_dir`, opens the first CSV, starts the writer thread.
- `log_frame(const FrameMetrics&)` — `buffer_.push(metrics)` only (non-blocking if space is available).
- `writer_thread_func` — drains the ring buffer, writes CSV rows, calls `flush` after a drain batch, optionally rotates, then sleeps `flush_interval_ms_`.
- `rotate_file_if_needed` — when `tellp()` meets `rotate_size_bytes_`, close file, open a new timestamped path, reset header flag.

Supporting infrastructure for GPU event timing exists in `src/utils/timer.h` (`CudaEventTimer` with `record_start`/`record_stop` on a stream and `elapsed_ms_async()` that reads elapsed time without synchronizing the stop event). The pipeline currently emphasizes **`CpuTimer`** for many stages; populating every `FrameMetrics` field from CUDA events and Triton stats is a natural follow-on that still uses the same async writer path.

## Architecture decisions

### SPSC lock-free ring buffer (atomic head/tail)

`RingBuffer` uses a fixed `std::array<T, Capacity>` and two `std::atomic<size_t>` indices:

- **`push`:** load `head` relaxed; compute `next = (head + 1) % Capacity`; if `next == tail` (acquire load), return false (full). Write slot, then store `head` with **release**.
- **`pop`:** load `tail` relaxed; if `tail == head` (acquire load), return false (empty). Read slot, advance `tail` with **release**.
- **`empty` / `size`:** acquire loads on both indices for a consistent snapshot.

**Reasoning:** The pipeline thread must not block on mutexes or disk I/O. One writer thread and one producer match SPSC exactly; correctness relies on the “one slot waste” convention (buffer capacity is `Capacity - 1` usable slots). Acquire/release pairs publish the producer/consumer visibility required without a mutex.

### CUDA events and async read strategy (design alignment)

The design spec requires:

- Record `cudaEventRecord` into the **same CUDA stream** as kernel work so timing reflects GPU ordering without per-frame `cudaDeviceSynchronize` in the hot path.
- **Read intervals on a separate thread** after the frame completes (or via `cudaEventQuery` / async-friendly elapsed read) so the pipeline thread never blocks on timing.

`CudaEventTimer::elapsed_ms_async()` calls `cudaEventElapsedTime` without first synchronizing the stop event; that is only valid once the stop event has actually completed—typically after a later point in the pipeline or on the monitor thread after `cudaStreamSynchronize` or `cudaEventSynchronize(stop)` once per frame or batch. The **metrics writer thread** is the right place to drain completed `FrameMetrics` rows: the producer pushes already-filled structs, so extending the producer to fill GPU-backed sub-intervals stays compatible with “no file I/O on the hot thread.”

**Reasoning:** Separating **record** (cheap, on stream) from **consume** (can wait) preserves the &lt; 0.04 ms instrumentation budget described in the spec.

### CSV schema: spec fields vs. emitted columns

`FrameMetrics` in code includes **all** atomic fields from the design document (including `preprocess_resize_ms`, `yolo_output_copy_ms`, `nms_sort_ms`, `faiss_normalize_ms`, etc.). The **current CSV header** written by `MetricsLogger` is a **compact subset** that aggregates some stages for readability and shorter rows—for example a single `preprocess_total_ms` column instead of four preprocess sub-columns, and `nms_total_ms` without separate sort/IoU columns. Empty/unused fields in the struct default to zero.

**Reasoning:** The struct is the long-term contract for rich telemetry; the writer can evolve to emit the full spec line verbatim by expanding the header and row formatting without changing the ring buffer API.

### File rotation (100 MB, timestamp in name)

When the open file’s position from `tellp()` reaches `rotate_size_mb * 1024 * 1024`, the implementation closes the file, generates a new name, opens it fresh, and clears `header_written_` so the next row writes a new header.

`generate_filename()` builds:

`{output_dir}/metrics_{YYYY-MM-DD_HHMMSS}.csv`

using `localtime_r` on the system clock—matching the spec’s `metrics_YYYY-MM-DD_HHMMSS.csv` pattern (local wall clock).

**Reasoning:** Size-based rotation avoids huge single files on long captures; timestamp names make sortable, unique files without a global counter.

### Flush interval (default 1 s)

After draining all currently queued items in a tight `while (buffer_.pop)` loop, the writer calls `file_.flush()` if anything was written, then sleeps `flush_interval_ms_` (constructor default **1000** ms). Tuning this trades latency of on-disk visibility against syscall overhead.

**Reasoning:** Matches `pipeline.yaml` example in the design spec (`flush_interval_ms: 1000`) and keeps the pipeline thread decoupled from flush timing.

### Overhead on the pipeline thread

`log_frame` is effectively a **lock-free `push` of `FrameMetrics`** (trivial copy of POD-like struct). The design spec allocates **~0.011 ms** per frame for “CSV row format + buffer push”; this implementation pushes the struct and defers **all** string formatting and `operator<<` work to the writer thread, so producer overhead is dominated by the ring buffer CAS-style updates—typically **at or below** the spec’s push budget.

**Reasoning:** Moving formatting off the hot path is the main lever to stay under the monitoring overhead target.

## Key code snippets

**Ring buffer core:**

```10:28:src/metrics/ring_buffer.h
    bool push(const T& item) {
        size_t head = head_.load(std::memory_order_relaxed);
        size_t next = (head + 1) % Capacity;
        if (next == tail_.load(std::memory_order_acquire)) {
            return false;  // full
        }
        buffer_[head] = item;
        head_.store(next, std::memory_order_release);
        return true;
    }

    bool pop(T& item) {
        size_t tail = tail_.load(std::memory_order_relaxed);
        if (tail == head_.load(std::memory_order_acquire)) {
            return false;  // empty
        }
        item = buffer_[tail];
        tail_.store((tail + 1) % Capacity, std::memory_order_release);
        return true;
    }
```

**Writer loop (header, row shape, flush, sleep):**

```54:96:src/metrics/metrics_logger.cpp
void MetricsLogger::writer_thread_func() {
    while (running_) {
        FrameMetrics m;
        bool wrote = false;

        while (buffer_.pop(m)) {
            if (!header_written_) {
                file_ << "frame_id,timestamp_utc,source_id,"
                      << "decode_ms,yuv_to_bgr_ms,cpu_to_gpu_ms,"
                      << "preprocess_total_ms,"
                      << "grpc1_overhead_ms,yolo_inference_ms,"
                      << "triton_yolo_queue_ms,triton_yolo_compute_ms,"
                      << "nms_total_ms,faces_detected,"
                      << "face_crop_ms,"
                      << "grpc2_overhead_ms,arcface_inference_ms,"
                      << "triton_arcface_queue_ms,triton_arcface_compute_ms,"
                      << "faiss_search_ms,"
                      << "result_copy_ms,total_pipeline_ms,gpu_memory_used_mb\n";
                header_written_ = true;
            }

            file_ << m.frame_id << "," << m.timestamp_utc << "," << m.source_id << ","
                  << m.decode_ms << "," << m.yuv_to_bgr_ms << "," << m.cpu_to_gpu_ms << ","
                  << m.preprocess_total_ms << ","
                  << m.grpc1_overhead_ms << "," << m.yolo_inference_ms << ","
                  << m.triton_yolo_queue_ms << "," << m.triton_yolo_compute_ms << ","
                  << m.nms_total_ms << "," << m.faces_detected << ","
                  << m.face_crop_ms << ","
                  << m.grpc2_overhead_ms << "," << m.arcface_inference_ms << ","
                  << m.triton_arcface_queue_ms << "," << m.triton_arcface_compute_ms << ","
                  << m.faiss_search_ms << ","
                  << m.result_copy_ms << "," << m.total_pipeline_ms << ","
                  << m.gpu_memory_used_mb << "\n";

            wrote = true;
            rotate_file_if_needed();
        }

        if (wrote) file_.flush();

        std::this_thread::sleep_for(std::chrono::milliseconds(flush_interval_ms_));
    }
}
```

**`CudaEventTimer` (async-friendly read path):**

```51:69:src/utils/timer.h
    void record_start(cudaStream_t stream = nullptr) {
        cudaEventRecord(start_, stream);
    }

    void record_stop(cudaStream_t stream = nullptr) {
        cudaEventRecord(stop_, stream);
    }

    float elapsed_ms() const {
        float ms = 0.0f;
        cudaEventSynchronize(stop_);
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }

    float elapsed_ms_async() const {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }
```

## Verification approach

1. **Functional:** Run the pipeline with `--output-csv ./metrics/run.csv`; confirm `metrics/` contains CSV files, headers appear once per file, and row counts track processed frames (including drops with `decode_ms == -1` or `total_pipeline_ms == -1` when those paths trigger).
2. **Rotation:** Lower `rotate_size_mb` in a test binary or temporarily patch `init` to call with `rotate_size_mb=1`, generate load, and confirm multiple `metrics_*.csv` files.
3. **Flush:** Set a small `flush_interval_ms`, watch file growth with `tail -f` or `ls -l` to see periodic updates after interval.
4. **Producer overhead:** Microbenchmark `MetricsLogger::log_frame` in isolation (repeated `push` of a default `FrameMetrics`) or profile the pipeline with metrics enabled vs. disabled; expect hot path dominated by decode/inference, not `log_frame`.
5. **Full schema:** When extending rows to match the design spec’s single-line column list exactly, diff the new header against the spec and validate with a spreadsheet or `awk` that column counts stay consistent.
