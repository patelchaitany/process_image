# Phase 0: Project Scaffolding — Implementation Details

This document describes what was implemented for **Phase 0 (Project Scaffolding)** of the real-time face detection and recognition pipeline: the CMake/CUDA build, shared utilities (`PipelineConfig`, timers), static YAML configuration stub, Python tool dependencies, and a compilable **hello-world** `main` that validates CLI parsing and graceful shutdown hooks. It aligns with the design spec: [Face Detection & Recognition Pipeline Design](../superpowers/specs/2026-05-02-face-detection-recognition-pipeline-design.md).

---

## 1. What Was Implemented

### 1.1 Root build system

| Artifact | Purpose |
|----------|---------|
| `CMakeLists.txt` | Declares `process_image` as a **CXX + CUDA** project (C++17), exports compile commands for IDE/clang tooling, pins default GPU architecture, locates FFmpeg (pkg-config), CUDA, SQLite3, yaml-cpp, Triton C++ client, FAISS (+ GPU), optional OpenCV and spdlog, aggregates all pipeline sources into one executable, and registers **GoogleTest**-based unit tests via `FetchContent`. |

Phase 0’s *functional* deliverable is a **build that succeeds** and a **`main`** that runs; the CMake file already lists source files for later phases so the graph of link dependencies is established early.

### 1.2 C++ utilities (headers)

| File | Purpose |
|------|---------|
| `src/utils/config.h` | Single `PipelineConfig` struct holding defaults for URLs, paths, thresholds, model names, frame/tensor geometry, NMS, and FAISS tuning; **`parse_args(int argc, char* argv[])`** implements the CLI described in the spec (subset of flags wired in Phase 0). |
| `src/utils/timer.h` | **CPU** timing (`CpuTimer`, `ScopedCpuTimer`) using `std::chrono::high_resolution_clock`; **GPU** timing (`CudaEventTimer`) using CUDA events with optional stream association and sync vs. async elapsed-time queries. |

These match the spec’s monitoring strategy: chrono on CPU paths, CUDA events on GPU paths.

### 1.3 Configuration and tooling

| File | Purpose |
|------|---------|
| `config/pipeline.yaml` | Declarative knobs for **metrics** (output dir, CSV format, flush/rotate policy) and **pipeline/detection/matching** sections that mirror fields in `PipelineConfig`. Loaded by later phases (e.g. metrics logger); Phase 0 establishes the schema alongside code defaults. |
| `tools/requirements.txt` | Python dependencies for the enrollment CLI and related scripts: `tritonclient[grpc]`, `numpy`, `opencv-python` — per spec. |

### 1.4 Application entry point

| File | Purpose |
|------|---------|
| `src/main.cpp` | Installs **SIGINT/SIGTERM** handlers (`std::atomic<bool> g_running`), calls `parse_args`, prints a **v0.1** banner with resolved config, and exits with a placeholder message that the full pipeline arrives in a later phase. **No FFmpeg/Triton/FAISS work** runs yet. |

### 1.5 Tests (wired in CMake, relevant to scaffolding)

CMake defines `test_nms`, `test_frame_source`, and `test_face_database` and uses `gtest_discover_tests` so CTest can enumerate cases. Their presence is part of the **test harness scaffolding** even when individual test bodies evolve in later phases.

---

## 2. Architecture Decisions

### 2.1 CMake structure

- **Project language mix**: `project(process_image LANGUAGES CXX CUDA)` ensures a single target can compile `.cu` translation units and link `CUDA::cudart` without a separate “nvcc-only” library, which keeps the real-time pipeline’s GPU code in-repo and linkable with the same flags as the rest of the app.

- **`CMAKE_EXPORT_COMPILE_COMMANDS ON`**: Produces `compile_commands.json` for clangd, include analysis, and static tooling — valuable for a mixed C++/CUDA tree.

- **`find_package(CUDAToolkit REQUIRED)`**: Uses CMake’s first-class CUDA package. Targets like `CUDA::cudart` are modern, cache-friendly, and avoid hard-coded `NVCC` / library paths when the toolkit is installed in standard layouts.

- **`find_package(PkgConfig REQUIRED)` + `pkg_check_modules(LIBAV REQUIRED …)`**: FFmpeg’s **libav\*** stack is exposed to the ecosystem primarily via `.pc` files on Linux/macOS. `pkg_check_modules` captures `CFLAGS`/`LDFLAGS`, include dirs, and library lists in one step. A plain `find_library` for `libavcodec` is fragile across distros and Homebrew prefixes.

- **`find_package(SQLite3 REQUIRED)`**: SQLite ships CMake config or Find modules on most platforms; linking `SQLite::SQLite3` is cleaner than manual `-lsqlite3`.

- **`find_path` / `find_library` for yaml-cpp, Triton client, FAISS**: Upstream does not always install CMake package configs in predictable locations. **Known search roots** (`/opt/tritonserver`, `/usr/local`, env vars `TRITON_CLIENT_DIR`, `FAISS_DIR`) make CI and bare-metal servers reproducible without requiring every dependency to be CMake-native.

- **`find_package(OpenCV QUIET …)`** and **`find_package(spdlog QUIET)`**: **Optional** dependencies gate preprocessor defines (`HAS_OPENCV`, `HAS_SPDLOG`) so CPU fallback and structured logging can be added without breaking minimal builds.

- **Google Test via `FetchContent`**: Declares `googletest` at a **pinned tag** (`v1.14.0`), sets `gtest_force_shared_crt` for MSVC compatibility, and calls `FetchContent_MakeAvailable`. Tests link `GTest::gtest_main` — no system `libgtest` required.

### 2.2 Config struct design

- **Single aggregate (`PipelineConfig`)**: All tunables visible in one place, easy to pass into a future `Pipeline` constructor and to diff against `pipeline.yaml`.

- **Sensible defaults in the struct**: CLI overrides only what the user specifies; the program runs with spec-consistent defaults (e.g. `triton_url = "localhost:8001"`, YOLO/ArcFace model name defaults).

- **CLI parsing via `parse_args`**: A **free function** returning `PipelineConfig` keeps `main` thin and avoids a global singleton. Parsing is **linear scan over `argv`** with explicit `--flag value` pairs; `--help` prints to stderr and `exit(0)`; missing `--input` is a hard error.

**Scope note (Phase 0):** The spec lists `--input` and `--triton` as required. The current parser **requires only `--input`**; `triton_url` falls back to the default if omitted. Tightening validation to require `--triton` is a one-line check if product owners want strict parity with the spec.

### 2.3 Timer design

- **CPU path**: `CpuTimer` exposes `start`, `stop`, and `elapsed_ms()` in float milliseconds. `ScopedCpuTimer` writes duration into a caller-provided `float&` in its destructor — useful for RAII-style section timing without manual `stop()` calls.

- **GPU path**: `CudaEventTimer` owns two `cudaEvent_t` handles (non-copyable). `record_start` / `record_stop` accept an optional `cudaStream_t` so timings track the same stream as kernel work. `elapsed_ms()` synchronizes on the **stop** event before `cudaEventElapsedTime` (safe when the consumer needs a completed interval). `elapsed_ms_async()` omits that synchronize — for the spec’s pattern of reading timings on a **monitoring thread** after the frame completes.

This dual design matches the design doc’s table: negligible overhead for chrono, CUDA events recorded into the stream without blocking the critical path until results are consumed.

### 2.4 Default CUDA architecture (`CMAKE_CUDA_ARCHITECTURES 75`)

The design spec targets **NVIDIA T4** (Turing, compute capability **7.5**). Setting `CMAKE_CUDA_ARCHITECTURES` to **75** ensures `nvcc` **JIT-compiles** (or **embeds** matching SASS) for that SM without relying on implicit “native” detection, which can differ between build and deploy hosts. For other GPUs, developers override this cache variable (e.g. `-DCMAKE_CUDA_ARCHITECTURES=80` for A100).

---

## 3. Reasoning (Design Rationale)

### 3.1 Manual `argc` / `argv` parsing instead of Boost.Program_options, cxxopts, etc.

- **Zero extra C++ dependency** for the executable’s surface area; the flag set is small and stable.
- **Header-only friendly**: `config.h` stays a single include with no link-time parser library.
- **Determinism and boot time**: No registration DSL or exception-heavy parsing paths — important for a latency-sensitive binary.
- **Trade-off**: Adding subcommands or complex syntax later may justify migrating to a library; for the documented CLI, a loop is sufficient and auditable.

### 3.2 `FetchContent` for Google Test

- **Reproducible version**: `GIT_TAG v1.14.0` pins behavior across machines.
- **No package manager mandate**: Works on minimal Docker images and servers where `apt install libgtest-dev` is undesirable or version-skewed.
- **Integrates with `GoogleTest` + `gtest_discover_tests`**: Automatic test registration in CTest without maintaining explicit test lists.

### 3.3 `pkg_check_modules` for FFmpeg

- FFmpeg’s ABI and **include paths** vary; pkg-config encodes **`-I`** order and transitive libs (e.g. dependency on `libavutil`) correctly.
- Matches how most **RTSP/MP4** C++ projects link FFmpeg on Linux and macOS.

### 3.4 CUDA SM 75 for T4

- **Performance**: Native SASS for 7.5 avoids running exclusively in PTX JIT fallback if fatbin is misconfigured.
- **Operational clarity**: The build explicitly documents “this artifact is meant for T4-class GPUs,” reducing “works on my workstation GPU” surprises.

---

## 4. Key Code Snippets

### 4.1 `PipelineConfig` and `parse_args`

From `src/utils/config.h`:

```cpp
struct PipelineConfig {
    std::string input_source;
    std::string triton_url = "localhost:8001";
    std::string output_csv = "./metrics/metrics.csv";
    std::string db_path = "./faces.db";
    float confidence_threshold = 0.5f;
    float match_threshold = 0.6f;
    std::string device_mode = "auto";  // "auto", "gpu", "cpu"
    std::string yolo_model = "yolo26_face";
    std::string arcface_model = "arcface";

    int frame_width = 1920;
    int frame_height = 1080;
    int yolo_input_size = 640;
    int arcface_input_size = 112;
    int max_faces_per_frame = 32;
    int embedding_dim = 512;

    float nms_iou_threshold = 0.45f;

    // FAISS
    int faiss_nprobe = 16;
    int faiss_ivf_threshold = 1000;  // switch to IVF above this many faces
};

inline PipelineConfig parse_args(int argc, char* argv[]) {
    PipelineConfig config;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--input" && i + 1 < argc) {
            config.input_source = argv[++i];
        } else if (arg == "--triton" && i + 1 < argc) {
            config.triton_url = argv[++i];
        }
        // ... additional flags and --help ...
    }

    if (config.input_source.empty()) {
        fprintf(stderr, "Error: --input is required\n");
        exit(1);
    }

    return config;
}
```

(The full file includes all flag branches and the help text; truncated here for readability.)

### 4.2 `CudaEventTimer`

From `src/utils/timer.h`:

```cpp
class CudaEventTimer {
public:
    CudaEventTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    ~CudaEventTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    CudaEventTimer(const CudaEventTimer&) = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;

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

    cudaEvent_t start_event() const { return start_; }
    cudaEvent_t stop_event() const { return stop_; }

private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
};
```

---

## 5. Verification

### 5.1 Compile-time / link-time

With dependencies installed per `CMakeLists.txt` (CUDA Toolkit, FFmpeg dev packages, SQLite3, yaml-cpp, Triton C++ client, FAISS, etc.), configuring and building the project should produce:

- `process_image` — linking CUDA runtime, FFmpeg, SQLite, yaml-cpp, Triton gRPC client, FAISS (+ GPU), pthread, and optional OpenCV/spdlog.

Phase 0’s **smoke criterion** is that this target **builds cleanly** on the reference stack (T4-oriented CUDA 7.5, C++17).

### 5.2 Runtime (hello-world)

Running the binary with required arguments, for example:

```bash
./process_image --input /path/to/video.mp4 --triton localhost:8001
```

should print:

- Application version line (`process_image v0.1`)
- Echo of **input**, **Triton URL**, **device mode**, **DB path**, **CSV path**
- The placeholder line: **“Pipeline not yet implemented. Scaffolding complete.”**

`SIGINT`/`SIGTERM` set `g_running` to false for future integration with the decode/inference loop (Phase 7 per `main.cpp` comment).

### 5.3 Relationship to the design spec

Phase 0 implements the **skeleton** named in the spec’s project layout: CMake, `config.h`, `timer.h`, `pipeline.yaml`, `tools/requirements.txt`, and CLI-driven `main.cpp`. It does **not** yet satisfy latency, Triton shared memory, or full CSV metrics — those are tracked in subsequent phases against the same spec.

---

## 6. File Index (Phase 0 Touchpoints)

| Path | Role |
|------|------|
| `CMakeLists.txt` | Build, deps, CUDA arch, tests |
| `src/main.cpp` | Signals, CLI, banner |
| `src/utils/config.h` | `PipelineConfig`, `parse_args` |
| `src/utils/timer.h` | CPU + CUDA timers |
| `config/pipeline.yaml` | Metrics & pipeline YAML |
| `tools/requirements.txt` | Python deps for enrollment tooling |

---

*Document generated to capture Phase 0 scaffolding as implemented in the repository; update subsequent sections when later phases replace the hello-world `main` body and wire `pipeline.yaml` into runtime loading.*
