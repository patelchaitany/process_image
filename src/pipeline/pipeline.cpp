#include "pipeline/pipeline.h"
#include <cstdio>
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

    // auto-detect
    use_gpu_ = (err == cudaSuccess && device_count > 0);
    return true;
}

bool Pipeline::init(const PipelineConfig& config) {
    config_ = config;

    if (!detect_device()) return false;

    printf("Device mode: %s\n", use_gpu_ ? "GPU" : "CPU");

    // Initialize frame source
    frame_source_ = std::make_unique<FFmpegSource>();
    if (!frame_source_->open(config_.input_source)) {
        fprintf(stderr, "Failed to open input source: %s\n", config_.input_source.c_str());
        return false;
    }
    printf("Input opened: %dx%d @ %.1f fps\n",
           frame_source_->width(), frame_source_->height(), frame_source_->fps());

    // Initialize Triton client with retry (3 attempts, 1s backoff)
    triton_client_ = std::make_unique<TritonClient>();
    if (!triton_client_->connect(config_.triton_url, 3, 1000)) {
        fprintf(stderr, "FATAL: Cannot reach Triton at %s after 3 retries. Exiting.\n",
                config_.triton_url.c_str());
        return false;
    }

    // Initialize GPU memory pool (with CUDA OOM recovery)
    if (use_gpu_) {
        memory_pool_ = std::make_unique<GPUMemoryPool>();
        if (!memory_pool_->init(frame_source_->width(), frame_source_->height(),
                                 config_.max_faces_per_frame)) {
            fprintf(stderr, "Warning: GPU memory allocation failed, attempting with reduced pool\n");
            // Retry with half the max faces
            if (!memory_pool_->init(frame_source_->width(), frame_source_->height(),
                                     config_.max_faces_per_frame / 2)) {
                fprintf(stderr, "Warning: GPU still OOM, falling back to CPU mode\n");
                use_gpu_ = false;
                memory_pool_.reset();
            }
        }

        if (memory_pool_) {
            memory_pool_->register_triton_shm(config_.triton_url);
        }
    }

    // Initialize preprocessor
    preprocessor_ = std::make_unique<Preprocessor>();

    // Initialize detectors/recognizers
    face_detector_ = std::make_unique<FaceDetector>(*triton_client_, config_.yolo_model);
    face_recognizer_ = std::make_unique<FaceRecognizer>(*triton_client_, config_.arcface_model);

    // Initialize face database
    face_database_ = std::make_unique<FaceDatabase>();
    if (!face_database_->open(config_.db_path)) {
        fprintf(stderr, "Failed to open face database: %s\n", config_.db_path.c_str());
        return false;
    }
    printf("Face database: %d faces loaded\n", face_database_->count());

    // Initialize face matcher
    face_matcher_ = std::make_unique<FaceMatcher>();
    if (!face_matcher_->init(*face_database_, config_.match_threshold, use_gpu_)) {
        fprintf(stderr, "Failed to initialize face matcher\n");
        return false;
    }

    // Initialize face cropper
    face_cropper_ = std::make_unique<FaceCropper>();

    // Initialize result handler
    result_handler_ = std::make_unique<ResultHandler>();

    // Initialize metrics logger
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
    printf("Pipeline initialized successfully\n");
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

        // Stage 1: Decode frame
        CpuTimer decode_timer;
        decode_timer.start();
        bool got_frame = frame_source_->read(frame);
        decode_timer.stop();

        if (!got_frame) {
            // EOF or decode error
            if (frame_source_->is_open()) {
                // Decode error on a still-open stream: skip frame
                metrics.decode_ms = -1.0f;
                frames_dropped++;
                metrics_logger_->log_frame(metrics);
                continue;
            }
            break;  // EOF
        }

        metrics.decode_ms = decode_timer.elapsed_ms();
        metrics.frame_id = frame.frame_index;

        // Process the frame -- catch gRPC failures gracefully
        bool success = false;
        try {
            process_frame(frame, metrics);
            success = true;
        } catch (...) {
            // gRPC or other unrecoverable error during frame processing
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

        metrics_logger_->log_frame(metrics);
    }

    printf("Pipeline finished. Processed: %lu, Dropped: %lu\n", frames_processed, frames_dropped);
}

void Pipeline::process_frame(Frame& frame, FrameMetrics& metrics) {
    cudaStream_t stream = memory_pool_ ? memory_pool_->stream() : nullptr;

    // Stage 2: Upload to GPU
    CpuTimer upload_timer;
    upload_timer.start();
    if (memory_pool_) {
        memory_pool_->upload_frame(frame.data.data(), frame.size_bytes(), stream);
    }
    upload_timer.stop();
    metrics.cpu_to_gpu_ms = upload_timer.elapsed_ms();

    // Stage 3: YOLO preprocessing
    CpuTimer preprocess_timer;
    preprocess_timer.start();
    if (memory_pool_ && preprocessor_) {
        preprocessor_->preprocess_yolo(
            memory_pool_->buffers().raw_frame,
            memory_pool_->buffers().yolo_input,
            frame.width, frame.height,
            config_.yolo_input_size, stream);
    }
    preprocess_timer.stop();
    metrics.preprocess_total_ms = preprocess_timer.elapsed_ms();

    // Stage 4-5: YOLO inference via Triton
    CpuTimer yolo_timer;
    yolo_timer.start();
    std::vector<RawDetection> raw_detections;
    bool yolo_ok = face_detector_->detect("yolo_input_shm", "yolo_output_shm",
                                           raw_detections, config_.confidence_threshold);
    yolo_timer.stop();
    metrics.yolo_inference_ms = yolo_timer.elapsed_ms();

    if (!yolo_ok) {
        // Triton gRPC call failed mid-pipeline: skip frame
        return;
    }

    // Stage 6-7: Post-process (NMS)
    CpuTimer nms_timer;
    nms_timer.start();

    PreprocessParams params = preprocessor_->last_params();
    std::vector<Detection> candidates = filter_and_decode(
        nullptr, 0,
        config_.confidence_threshold, config_.yolo_input_size,
        frame.width, frame.height,
        params.scale, params.pad_x, params.pad_y);

    std::vector<Detection> detections = apply_nms(candidates, config_.nms_iou_threshold);
    nms_timer.stop();
    metrics.nms_total_ms = nms_timer.elapsed_ms();
    metrics.faces_detected = static_cast<int>(detections.size());

    if (detections.empty()) {
        // No faces - still a successful frame, just nothing to recognize
        FrameResult result;
        result.frame_id = frame.frame_index;
        result_handler_->handle(result);
        return;
    }

    // Stage 8: Face cropping + ArcFace preprocessing
    CpuTimer crop_timer;
    crop_timer.start();
    if (memory_pool_) {
        face_cropper_->crop_and_preprocess(
            memory_pool_->buffers().raw_frame, detections,
            memory_pool_->buffers().arcface_input,
            frame.width, frame.height, stream);
    }
    crop_timer.stop();
    metrics.face_crop_ms = crop_timer.elapsed_ms();

    // Stage 9: ArcFace inference
    CpuTimer arcface_timer;
    arcface_timer.start();
    std::vector<std::vector<float>> embeddings;
    bool arcface_ok = face_recognizer_->recognize(
        "arcface_input_shm", "arcface_output_shm",
        static_cast<int>(detections.size()), embeddings);
    arcface_timer.stop();
    metrics.arcface_inference_ms = arcface_timer.elapsed_ms();

    if (!arcface_ok) {
        // ArcFace gRPC failed - report detections without matches
        FrameResult result;
        result.frame_id = frame.frame_index;
        result.detections = detections;
        result_handler_->handle(result);
        return;
    }

    // Stage 10: FAISS matching
    CpuTimer match_timer;
    match_timer.start();
    std::vector<MatchResult> matches = face_matcher_->match(embeddings);
    match_timer.stop();
    metrics.faiss_search_ms = match_timer.elapsed_ms();

    // Stage 11: Result handling
    FrameResult result;
    result.frame_id = frame.frame_index;
    result.detections = detections;
    result.matches = matches;
    result_handler_->handle(result);
}

void Pipeline::shutdown() {
    if (metrics_logger_) {
        metrics_logger_->shutdown();
        metrics_logger_.reset();
    }
    if (memory_pool_) {
        memory_pool_->unregister_triton_shm();
        memory_pool_->release();
        memory_pool_.reset();
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
    face_cropper_.reset();
    result_handler_.reset();
    preprocessor_.reset();
    initialized_ = false;
}
