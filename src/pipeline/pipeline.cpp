#include "pipeline/pipeline.h"
#include "output/console_result_writer.h"
#include "output/csv_result_writer.h"
#include <cstdio>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cuda_runtime.h>

Pipeline::~Pipeline() {
    shutdown();
}

bool Pipeline::detect_device() {
    if (config_.device_mode == "cpu") {
        use_gpu_ = false;
        return true;
    }

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (config_.device_mode == "gpu") {
        if (err != cudaSuccess || device_count == 0) {
            fprintf(stderr, "Error: --device gpu specified but no CUDA device found\n");
            return false;
        }
        use_gpu_ = true;
        return true;
    }

    use_gpu_ = (err == cudaSuccess && device_count > 0);
    return true;
}

std::string Pipeline::utc_timestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    std::tm tm{};
    gmtime_r(&time, &tm);
    std::ostringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S") << '.' << std::setfill('0')
       << std::setw(3) << ms.count() << 'Z';
    return ss.str();
}

FrameResult Pipeline::make_result(uint64_t frame_id) const {
    FrameResult r;
    r.frame_id = frame_id;
    r.timestamp_utc = utc_timestamp();
    r.source_id = config_.input_source;
    return r;
}

bool Pipeline::init(const PipelineConfig& config) {
    config_ = config;

    if (!detect_device()) return false;
    printf("Device mode: %s\n", use_gpu_ ? "GPU" : "CPU");

    // Try NVDEC hardware decode first (GPU path, no CPU threads)
    if (use_gpu_) {
        auto nvdec = std::make_unique<NvdecSource>();
        if (nvdec->open(config_.input_source)) {
            frame_source_ = std::move(nvdec);
            printf("Frame source: NVDEC hardware decoder (GPU)\n");
        } else {
            fprintf(stderr, "NVDEC unavailable, falling back to CPU decoder\n");
        }
    }

    // CPU fallback: single-threaded FFmpeg + PrefetchSource to hide latency
    if (!frame_source_) {
        auto cpu_source = std::make_unique<FFmpegSource>();
        auto prefetched = std::make_unique<PrefetchSource>(std::move(cpu_source), 5);
        if (!prefetched->open(config_.input_source)) {
            fprintf(stderr, "Failed to open input source: %s\n", config_.input_source.c_str());
            return false;
        }
        frame_source_ = std::move(prefetched);
        printf("Frame source: CPU decoder + prefetch\n");
    }
    printf("Input opened: %dx%d @ %.1f fps\n",
           frame_source_->width(), frame_source_->height(), frame_source_->fps());

    triton_client_ = std::make_unique<TritonClient>();
    if (!triton_client_->connect(config_.triton_url, 3, 1000)) {
        fprintf(stderr, "FATAL: Cannot reach Triton at %s after 3 retries. Exiting.\n",
                config_.triton_url.c_str());
        return false;
    }

    if (use_gpu_) {
        memory_pool_ = std::make_unique<GPUMemoryPool>();
        if (!memory_pool_->init(frame_source_->width(), frame_source_->height(),
                                 config_.max_faces_per_frame)) {
            fprintf(stderr, "Warning: GPU memory allocation failed, attempting with reduced pool\n");
            if (!memory_pool_->init(frame_source_->width(), frame_source_->height(),
                                     config_.max_faces_per_frame / 2)) {
                fprintf(stderr, "Warning: GPU still OOM, falling back to CPU mode\n");
                use_gpu_ = false;
                memory_pool_.reset();
            }
        }
        if (memory_pool_) {
            if (!memory_pool_->register_triton_shm(*triton_client_)) {
                fprintf(stderr, "Warning: SHM registration failed, falling back to CPU mode\n");
                memory_pool_->release();
                memory_pool_.reset();
                use_gpu_ = false;
            }
        }
        if (use_gpu_) {
            preprocessor_ = std::make_unique<Preprocessor>();
        }
    }

    if (!use_gpu_) {
        cpu_memory_pool_ = std::make_unique<CPUMemoryPool>();
        cpu_memory_pool_->init(frame_source_->width(), frame_source_->height(),
                               config_.max_faces_per_frame);
        cpu_preprocessor_ = std::make_unique<CPUPreprocessor>();
    }

    face_detector_ = std::make_unique<FaceDetector>(*triton_client_, config_.yolo_model);
    face_recognizer_ = std::make_unique<FaceRecognizer>(*triton_client_, config_.arcface_model);

    face_database_ = std::make_unique<FaceDatabase>();
    if (!face_database_->open(config_.db_path)) {
        fprintf(stderr, "Failed to open face database: %s\n", config_.db_path.c_str());
        return false;
    }
    printf("Face database: %d faces loaded\n", face_database_->count());

    if (use_gpu_) {
        face_matcher_ = std::make_unique<FaceMatcher>();
        if (!face_matcher_->init(*face_database_, config_.match_threshold, true)) {
            fprintf(stderr, "Failed to initialize GPU face matcher\n");
            return false;
        }
    } else {
        cpu_face_matcher_ = std::make_unique<CPUFaceMatcher>();
        if (!cpu_face_matcher_->init(*face_database_, config_.match_threshold)) {
            fprintf(stderr, "Failed to initialize CPU face matcher\n");
            return false;
        }
    }

    face_cropper_ = std::make_unique<FaceCropper>();

    // --- Result output writers ---
    result_handler_ = std::make_unique<ResultHandler>();
    result_handler_->addWriter(
        std::make_unique<ConsoleResultWriter>(config_.console_verbose));

    if (!config_.bbox_csv.empty()) {
        result_handler_->addWriter(
            std::make_unique<CsvResultWriter>(config_.bbox_csv, config_.bbox_csv_rotate_mb));
    }

    if (!result_handler_->openAll()) {
        fprintf(stderr, "Failed to open one or more result writers\n");
        return false;
    }

    // --- Metrics ---
    metrics_logger_ = std::make_unique<MetricsLogger>();
    std::string metrics_dir = config_.output_csv;
    auto last_slash = metrics_dir.rfind('/');
    if (last_slash != std::string::npos) {
        metrics_dir = metrics_dir.substr(0, last_slash);
    } else {
        metrics_dir = "./metrics";
    }
    metrics_logger_->init(metrics_dir);

    initialized_ = true;
    printf("Pipeline initialized successfully (%s mode)\n", use_gpu_ ? "GPU" : "CPU");
    return true;
}

void Pipeline::run(std::atomic<bool>& running) {
    if (!initialized_) {
        fprintf(stderr, "Pipeline not initialized\n");
        return;
    }

    Frame frame;
    FrameMetrics metrics;
    CpuTimer total_timer;
    uint64_t frames_processed = 0;
    uint64_t frames_dropped = 0;

    while (running.load(std::memory_order_relaxed)) {
        total_timer.start();
        metrics = FrameMetrics{};
        metrics.timestamp_utc = utc_timestamp();
        metrics.source_id = config_.input_source;

        CpuTimer decode_timer;
        decode_timer.start();
        bool got_frame = frame_source_->read(frame);
        decode_timer.stop();

        if (!got_frame) {
            if (frame_source_->is_open()) {
                metrics.decode_ms = -1.0f;
                frames_dropped++;
                metrics_logger_->log_frame(metrics);
                continue;
            }
            break;
        }

        metrics.decode_ms = decode_timer.elapsed_ms();
        metrics.frame_id = frame.frame_index;

        bool success = false;
        try {
            if (use_gpu_) {
                process_frame_gpu(frame, metrics);
            } else {
                process_frame_cpu(frame, metrics);
            }
            success = true;
        } catch (...) {
            success = false;
        }

        if (!success) {
            frames_dropped++;
            metrics.total_pipeline_ms = -1.0f;
            metrics_logger_->log_frame(metrics);
            continue;
        }

        total_timer.stop();
        metrics.total_pipeline_ms = total_timer.elapsed_ms();
        frames_processed++;

        if (use_gpu_) {
            size_t free_mem = 0, total_mem = 0;
            if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
                metrics.gpu_memory_used_mb =
                    static_cast<float>(total_mem - free_mem) / (1024.0f * 1024.0f);
            }
        }

        metrics_logger_->log_frame(metrics);
    }

    printf("Pipeline finished. Processed: %lu, Dropped: %lu\n", frames_processed, frames_dropped);
}

std::vector<Detection> Pipeline::decode_raw_detections(
    const std::vector<RawDetection>& raw, float scale, int pad_x, int pad_y,
    int orig_width, int orig_height) const {

    std::vector<Detection> candidates;
    candidates.reserve(raw.size());
    for (const auto& rd : raw) {
        float x1 = (rd.cx - rd.w / 2.0f - pad_x) / scale;
        float y1 = (rd.cy - rd.h / 2.0f - pad_y) / scale;
        float x2 = (rd.cx + rd.w / 2.0f - pad_x) / scale;
        float y2 = (rd.cy + rd.h / 2.0f - pad_y) / scale;

        x1 = std::max(0.0f, std::min(x1, static_cast<float>(orig_width)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(orig_height)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(orig_width)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(orig_height)));

        if (x2 > x1 && y2 > y1) {
            candidates.push_back({x1, y1, x2, y2, rd.confidence});
        }
    }
    return candidates;
}

void Pipeline::process_frame_gpu(Frame& frame, FrameMetrics& metrics) {
    cudaStream_t stream = memory_pool_->stream();
    CudaEventTimer gpu_timer;

    gpu_timer.record_start(stream);
    if (frame.on_gpu) {
        memory_pool_->copy_gpu_frame(frame.gpu_data, frame.size_bytes(), stream);
    } else {
        memory_pool_->upload_frame(frame.data.data(), frame.size_bytes(), stream);
    }
    gpu_timer.record_stop(stream);
    metrics.cpu_to_gpu_ms = gpu_timer.elapsed_ms();

    gpu_timer.record_start(stream);
    preprocessor_->preprocess_yolo(
        memory_pool_->buffers().raw_frame,
        memory_pool_->buffers().yolo_input,
        frame.width, frame.height,
        config_.yolo_input_size, stream);
    gpu_timer.record_stop(stream);
    metrics.preprocess_total_ms = gpu_timer.elapsed_ms();

    CpuTimer yolo_timer;
    yolo_timer.start();
    std::vector<RawDetection> raw_detections;
    InferResult yoloStats;
    bool yolo_ok = face_detector_->detect("yolo_input_shm", "yolo_output_shm",
                                           raw_detections, config_.confidence_threshold,
                                           &yoloStats);
    yolo_timer.stop();
    metrics.yolo_inference_ms = yolo_timer.elapsed_ms();
    metrics.triton_yolo_queue_ms = yoloStats.queueTimeNs / 1e6f;
    metrics.triton_yolo_compute_ms = yoloStats.computeTimeNs / 1e6f;

    if (!yolo_ok) return;

    CpuTimer nms_timer;
    nms_timer.start();
    PreprocessParams params = preprocessor_->last_params();
    std::vector<Detection> candidates = decode_raw_detections(
        raw_detections, params.scale, params.pad_x, params.pad_y,
        frame.width, frame.height);
    std::vector<Detection> detections = apply_nms(candidates, config_.nms_iou_threshold);
    nms_timer.stop();
    metrics.nms_total_ms = nms_timer.elapsed_ms();
    metrics.faces_detected = static_cast<int>(detections.size());

    if (detections.empty()) {
        FrameResult result = make_result(frame.frame_index);
        result_handler_->handle(result);
        return;
    }

    gpu_timer.record_start(stream);
    face_cropper_->crop_and_preprocess(
        memory_pool_->buffers().raw_frame, detections,
        memory_pool_->buffers().arcface_input,
        frame.width, frame.height, stream);
    gpu_timer.record_stop(stream);
    metrics.face_crop_ms = gpu_timer.elapsed_ms();

    CpuTimer arcface_timer;
    arcface_timer.start();
    std::vector<std::vector<float>> embeddings;
    InferResult arcfaceStats;
    bool arcface_ok = face_recognizer_->recognize(
        "arcface_input_shm", "arcface_output_shm",
        static_cast<int>(detections.size()), embeddings, &arcfaceStats);
    arcface_timer.stop();
    metrics.arcface_inference_ms = arcface_timer.elapsed_ms();
    metrics.triton_arcface_queue_ms = arcfaceStats.queueTimeNs / 1e6f;
    metrics.triton_arcface_compute_ms = arcfaceStats.computeTimeNs / 1e6f;

    if (!arcface_ok) {
        FrameResult result = make_result(frame.frame_index);
        result.detections = detections;
        result_handler_->handle(result);
        return;
    }

    CpuTimer match_timer;
    match_timer.start();
    std::vector<MatchResult> matches = face_matcher_->match(embeddings);
    match_timer.stop();
    metrics.faiss_search_ms = match_timer.elapsed_ms();

    FrameResult result = make_result(frame.frame_index);
    result.detections = detections;
    result.matches = matches;
    result_handler_->handle(result);
}

void Pipeline::process_frame_cpu(Frame& frame, FrameMetrics& metrics) {
    {
        ScopedCpuTimer t(metrics.cpu_to_gpu_ms);
        cpu_memory_pool_->upload_frame(frame.data.data(), frame.size_bytes());
    }

    {
        ScopedCpuTimer t(metrics.preprocess_total_ms);
        cpu_preprocessor_->preprocess_yolo(
            cpu_memory_pool_->buffers().raw_frame.data(),
            cpu_memory_pool_->buffers().yolo_input.data(),
            frame.width, frame.height, config_.yolo_input_size);
    }

    CpuTimer yolo_timer;
    yolo_timer.start();
    std::vector<RawDetection> raw_detections;
    InferResult yoloStats;
    bool yolo_ok = face_detector_->detectDirect(
        cpu_memory_pool_->buffers().yolo_input.data(),
        raw_detections, config_.confidence_threshold, &yoloStats);
    yolo_timer.stop();
    metrics.yolo_inference_ms = yolo_timer.elapsed_ms();
    metrics.triton_yolo_queue_ms = yoloStats.queueTimeNs / 1e6f;
    metrics.triton_yolo_compute_ms = yoloStats.computeTimeNs / 1e6f;

    if (!yolo_ok) return;

    CpuTimer nms_timer;
    nms_timer.start();
    CPUPreprocessParams params = cpu_preprocessor_->last_params();
    std::vector<Detection> candidates = decode_raw_detections(
        raw_detections, params.scale, params.pad_x, params.pad_y,
        frame.width, frame.height);
    std::vector<Detection> detections = apply_nms(candidates, config_.nms_iou_threshold);
    nms_timer.stop();
    metrics.nms_total_ms = nms_timer.elapsed_ms();
    metrics.faces_detected = static_cast<int>(detections.size());

    if (detections.empty()) {
        FrameResult result = make_result(frame.frame_index);
        result_handler_->handle(result);
        return;
    }

    {
        ScopedCpuTimer t(metrics.face_crop_ms);
        std::vector<float> bboxes;
        bboxes.reserve(detections.size() * 4);
        for (const auto& d : detections) {
            bboxes.push_back(d.x1);
            bboxes.push_back(d.y1);
            bboxes.push_back(d.x2);
            bboxes.push_back(d.y2);
        }
        cpu_preprocessor_->preprocess_arcface(
            cpu_memory_pool_->buffers().raw_frame.data(),
            cpu_memory_pool_->buffers().arcface_input.data(),
            bboxes.data(), static_cast<int>(detections.size()),
            frame.width, frame.height);
    }

    CpuTimer arcface_timer;
    arcface_timer.start();
    std::vector<std::vector<float>> embeddings;
    InferResult arcfaceStats;
    bool arcface_ok = face_recognizer_->recognizeDirect(
        cpu_memory_pool_->buffers().arcface_input.data(),
        static_cast<int>(detections.size()), embeddings, &arcfaceStats);
    arcface_timer.stop();
    metrics.arcface_inference_ms = arcface_timer.elapsed_ms();
    metrics.triton_arcface_queue_ms = arcfaceStats.queueTimeNs / 1e6f;
    metrics.triton_arcface_compute_ms = arcfaceStats.computeTimeNs / 1e6f;

    if (!arcface_ok) {
        FrameResult result = make_result(frame.frame_index);
        result.detections = detections;
        result_handler_->handle(result);
        return;
    }

    CpuTimer match_timer;
    match_timer.start();
    std::vector<MatchResult> matches = cpu_face_matcher_->match(embeddings);
    match_timer.stop();
    metrics.faiss_search_ms = match_timer.elapsed_ms();

    FrameResult result = make_result(frame.frame_index);
    result.detections = detections;
    result.matches = matches;
    result_handler_->handle(result);
}

void Pipeline::shutdown() {
    if (result_handler_) {
        result_handler_->closeAll();
    }
    if (metrics_logger_) {
        metrics_logger_->shutdown();
        metrics_logger_.reset();
    }
    if (memory_pool_ && triton_client_) {
        memory_pool_->unregister_triton_shm(*triton_client_);
        memory_pool_->release();
        memory_pool_.reset();
    }
    if (cpu_memory_pool_) {
        cpu_memory_pool_->release();
        cpu_memory_pool_.reset();
    }
    if (triton_client_) {
        triton_client_->disconnect();
        triton_client_.reset();
    }
    if (face_database_) {
        face_database_->close();
        face_database_.reset();
    }
    if (frame_source_) {
        frame_source_->close();
        frame_source_.reset();
    }
    face_detector_.reset();
    face_recognizer_.reset();
    face_matcher_.reset();
    cpu_face_matcher_.reset();
    face_cropper_.reset();
    result_handler_.reset();
    preprocessor_.reset();
    cpu_preprocessor_.reset();
    initialized_ = false;
}
