#include "service/pipeline_session.h"

#include "frame_source/ffmpeg_source.h"
#include "frame_source/nvdec_source.h"
#include "frame_source/prefetch_source.h"
#include "gpu/cpu_memory_pool.h"
#include "gpu/cpu_preprocessor.h"
#include "gpu/memory_pool.h"
#include "gpu/preprocessor.h"
#include "matching/cpu_face_matcher.h"
#include "output/webhook_result_writer.h"
#include "pipeline/result_handler.h"
#include "postprocess/nms.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sstream>

PipelineSession::PipelineSession(const std::string& session_id,
                               const std::string& session_name,
                               const std::string& source_uri,
                               const std::string& callback_url,
                               const SessionConfig& config,
                               SharedResources& shared)
    : sessionId_(session_id),
      sessionName_(session_name),
      sourceUri_(source_uri),
      callbackUrl_(callback_url),
      config_(config),
      state_("starting"),
      shared_(shared) {}

PipelineSession::~PipelineSession() {
    stop();
}

bool PipelineSession::start() {
    {
        std::lock_guard<std::mutex> lock(stateMutex_);
        startTime_ = std::chrono::steady_clock::now();
    }
    useGpu_ = shared_.use_gpu;
    if (!initFrameSource()) {
        std::lock_guard<std::mutex> lock(stateMutex_);
        errorMessage_ = "failed to open frame source";
        state_ = "error";
        cleanupResources();
        return false;
    }
    bool pipeline_ok = useGpu_ ? initGpuResources() : initCpuResources();
    if (!pipeline_ok) {
        std::lock_guard<std::mutex> lock(stateMutex_);
        errorMessage_ = useGpu_ ? "GPU pool or SHM registration failed"
                               : "CPU pool or matcher initialization failed";
        state_ = "error";
        cleanupResources();
        return false;
    }
    if (!initResultHandler()) {
        std::lock_guard<std::mutex> lock(stateMutex_);
        errorMessage_ = "result handler initialization failed";
        state_ = "error";
        cleanupResources();
        return false;
    }
    running_.store(true);
    {
        std::lock_guard<std::mutex> lock(stateMutex_);
        state_ = "running";
    }
    workerThread_ = std::thread([this]() { runLoop(); });
    return true;
}

void PipelineSession::stop() {
    running_.store(false);
    if (workerThread_.joinable()) {
        workerThread_.join();
    }
    cleanupResources();
    setState("stopped");
}

SessionStatus PipelineSession::status() const {
    SessionStatus s;
    s.session_id = sessionId_;
    s.session_name = sessionName_;
    s.source_uri = sourceUri_;
    s.frames_processed = framesProcessed_.load(std::memory_order_relaxed);
    s.frames_dropped = framesDropped_.load(std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(stateMutex_);
    s.state = state_;
    s.error_message = errorMessage_;
    if (state_ != "starting") {
        auto elapsed = std::chrono::steady_clock::now() - startTime_;
        s.uptime_seconds = std::chrono::duration<double>(elapsed).count();
    }
    return s;
}

void PipelineSession::runLoop() {
    try {
        Frame frame;
        while (running_.load(std::memory_order_relaxed)) {
            auto tick = std::chrono::steady_clock::now();
            bool got_frame = frameSource_->read(frame);
            if (!got_frame) {
                if (frameSource_->is_open()) {
                    framesDropped_.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                break;
            }
            bool ok = false;
            try {
                if (useGpu_) {
                    processFrameGpu(frame);
                } else {
                    processFrameCpu(frame);
                }
                ok = true;
            } catch (...) {
                ok = false;
            }
            if (!ok) {
                framesDropped_.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            framesProcessed_.fetch_add(1, std::memory_order_relaxed);
            if (config_.max_fps > 0) {
                using namespace std::chrono;
                const auto period = duration<double>(1.0 / static_cast<double>(config_.max_fps));
                const auto elapsed = steady_clock::now() - tick;
                if (elapsed < period) {
                    std::this_thread::sleep_for(period - elapsed);
                }
            }
        }
        setState("finished");
    } catch (const std::exception& ex) {
        {
            std::lock_guard<std::mutex> lock(stateMutex_);
            errorMessage_ = ex.what();
        }
        setState("error");
    } catch (...) {
        {
            std::lock_guard<std::mutex> lock(stateMutex_);
            errorMessage_ = "unknown error in pipeline session";
        }
        setState("error");
    }
}

void PipelineSession::setState(const std::string& state) {
    std::lock_guard<std::mutex> lock(stateMutex_);
    state_ = state;
}

bool PipelineSession::initFrameSource() {
    if (shared_.use_gpu) {
        auto nvdec = std::make_unique<NvdecSource>();
        if (nvdec->open(sourceUri_)) {
            frameSource_ = std::move(nvdec);
            return true;
        }
    }
    auto cpu_src = std::make_unique<FFmpegSource>();
    auto prefetched =
        std::make_unique<PrefetchSource>(std::move(cpu_src), 5);
    if (!prefetched->open(sourceUri_)) {
        return false;
    }
    frameSource_ = std::move(prefetched);
    return true;
}

bool PipelineSession::initGpuResources() {
    if (!shared_.triton_client) {
        return false;
    }
    memoryPool_ = std::make_unique<GPUMemoryPool>();
    const int w = frameSource_->width();
    const int h = frameSource_->height();
    if (!memoryPool_->init(w, h, shared_.max_faces_per_frame)) {
        return false;
    }
    std::string prefix = sessionId_;
    auto& bufs = memoryPool_->buffers();
    auto& tc = *shared_.triton_client;
    const size_t yolo_in_bytes =
        static_cast<size_t>(1) * 3 * shared_.yolo_input_size *
        shared_.yolo_input_size * sizeof(float);
    const size_t yolo_out_bytes = static_cast<size_t>(8400) * 5 * sizeof(float);
    const int mf = shared_.max_faces_per_frame;
    const size_t arc_in_bytes =
        static_cast<size_t>(mf) * 3 * 112 * 112 * sizeof(float);
    const size_t arc_out_bytes =
        static_cast<size_t>(mf) * 512 * sizeof(float);
    shmYoloIn_ = "yolo_input_shm_" + prefix;
    shmYoloOut_ = "yolo_output_shm_" + prefix;
    shmArcIn_ = "arcface_input_shm_" + prefix;
    shmArcOut_ = "arcface_output_shm_" + prefix;
    if (!tc.registerCudaShm(shmYoloIn_, bufs.yolo_input, yolo_in_bytes)) {
        return false;
    }
    if (!tc.registerCudaShm(shmYoloOut_, bufs.yolo_output, yolo_out_bytes)) {
        return false;
    }
    if (!tc.registerCudaShm(shmArcIn_, bufs.arcface_input, arc_in_bytes)) {
        return false;
    }
    if (!tc.registerCudaShm(shmArcOut_, bufs.arcface_output, arc_out_bytes)) {
        return false;
    }
    preprocessor_ = std::make_unique<Preprocessor>();
    return true;
}

bool PipelineSession::initCpuResources() {
    cpuMemoryPool_ = std::make_unique<CPUMemoryPool>();
    const int w = frameSource_->width();
    const int h = frameSource_->height();
    if (!cpuMemoryPool_->init(w, h, shared_.max_faces_per_frame)) {
        return false;
    }
    cpuPreprocessor_ = std::make_unique<CPUPreprocessor>();
    cpuFaceMatcher_ = std::make_unique<CPUFaceMatcher>();
    const float mt =
        config_.match_threshold > 0.0f ? config_.match_threshold
                                         : shared_.match_threshold;
    if (!cpuFaceMatcher_->init(*shared_.face_database, mt)) {
        return false;
    }
    return true;
}

bool PipelineSession::initResultHandler() {
    resultHandler_ = std::make_unique<ResultHandler>();
    resultHandler_->addWriter(
        std::make_unique<WebhookResultWriter>(callbackUrl_, sessionId_));
    return resultHandler_->openAll();
}

void PipelineSession::processFrameGpu(Frame& frame) {
    cudaStream_t stream = memoryPool_->stream();
    if (frame.on_gpu) {
        memoryPool_->copy_gpu_frame(frame.gpu_data, frame.size_bytes(), stream);
    } else {
        memoryPool_->upload_frame(frame.data.data(), frame.size_bytes(), stream);
    }
    preprocessor_->preprocess_yolo(memoryPool_->buffers().raw_frame,
                                   memoryPool_->buffers().yolo_input,
                                   frame.width, frame.height,
                                   shared_.yolo_input_size, stream);
    std::vector<RawDetection> raw_detections;
    const float conf = effectiveConfidenceThreshold();
    bool yolo_ok =
        shared_.face_detector->detect(shmYoloIn_, shmYoloOut_, raw_detections,
                                      conf, nullptr);
    if (!yolo_ok) {
        return;
    }
    PreprocessParams params = preprocessor_->last_params();
    std::vector<Detection> candidates =
        decodeRawDetections(raw_detections, params.scale, params.pad_x,
                            params.pad_y, frame.width, frame.height);
    std::vector<Detection> detections =
        apply_nms(candidates, shared_.nms_iou_threshold);
    if (detections.empty()) {
        FrameResult result = makeResult(frame.frame_index);
        resultHandler_->handle(result);
        return;
    }
    shared_.face_cropper->crop_and_preprocess(
        memoryPool_->buffers().raw_frame, detections,
        memoryPool_->buffers().arcface_input, frame.width, frame.height,
        stream);
    std::vector<std::vector<float>> embeddings;
    bool arc_ok =
        shared_.face_recognizer->recognize(shmArcIn_, shmArcOut_,
                                           static_cast<int>(detections.size()),
                                           embeddings, nullptr);
    if (!arc_ok) {
        FrameResult result = makeResult(frame.frame_index);
        result.detections = detections;
        resultHandler_->handle(result);
        return;
    }
    std::vector<MatchResult> matches =
        shared_.face_matcher->match(embeddings);
    FrameResult result = makeResult(frame.frame_index);
    result.detections = detections;
    result.matches = matches;
    resultHandler_->handle(result);
}

void PipelineSession::processFrameCpu(Frame& frame) {
    cpuMemoryPool_->upload_frame(frame.data.data(), frame.size_bytes());
    cpuPreprocessor_->preprocess_yolo(cpuMemoryPool_->buffers().raw_frame.data(),
                                      cpuMemoryPool_->buffers().yolo_input.data(),
                                      frame.width, frame.height,
                                      shared_.yolo_input_size);
    std::vector<RawDetection> raw_detections;
    const float conf = effectiveConfidenceThreshold();
    bool yolo_ok =
        shared_.face_detector->detectDirect(
            cpuMemoryPool_->buffers().yolo_input.data(), raw_detections,
            conf, nullptr);
    if (!yolo_ok) {
        return;
    }
    CPUPreprocessParams params = cpuPreprocessor_->last_params();
    std::vector<Detection> candidates =
        decodeRawDetections(raw_detections, params.scale, params.pad_x,
                            params.pad_y, frame.width, frame.height);
    std::vector<Detection> detections =
        apply_nms(candidates, shared_.nms_iou_threshold);
    if (detections.empty()) {
        FrameResult result = makeResult(frame.frame_index);
        resultHandler_->handle(result);
        return;
    }
    std::vector<float> bboxes;
    bboxes.reserve(detections.size() * 4);
    for (const auto& d : detections) {
        bboxes.push_back(d.x1);
        bboxes.push_back(d.y1);
        bboxes.push_back(d.x2);
        bboxes.push_back(d.y2);
    }
    cpuPreprocessor_->preprocess_arcface(
        cpuMemoryPool_->buffers().raw_frame.data(),
        cpuMemoryPool_->buffers().arcface_input.data(),
        bboxes.data(), static_cast<int>(detections.size()),
        frame.width, frame.height);
    std::vector<std::vector<float>> embeddings;
    bool arc_ok =
        shared_.face_recognizer->recognizeDirect(
            cpuMemoryPool_->buffers().arcface_input.data(),
            static_cast<int>(detections.size()), embeddings, nullptr);
    if (!arc_ok) {
        FrameResult result = makeResult(frame.frame_index);
        result.detections = detections;
        resultHandler_->handle(result);
        return;
    }
    std::vector<MatchResult> matches = cpuFaceMatcher_->match(embeddings);
    FrameResult result = makeResult(frame.frame_index);
    result.detections = detections;
    result.matches = matches;
    resultHandler_->handle(result);
}

FrameResult PipelineSession::makeResult(uint64_t frame_id) const {
    FrameResult r;
    r.frame_id = frame_id;
    r.timestamp_utc = utcTimestamp();
    r.source_id = sourceUri_;
    return r;
}

std::string PipelineSession::utcTimestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  now.time_since_epoch()) %
              1000;
    std::tm tm{};
    gmtime_r(&time, &tm);
    std::ostringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S") << '.' << std::setfill('0')
       << std::setw(3) << ms.count() << 'Z';
    return ss.str();
}

float PipelineSession::effectiveConfidenceThreshold() const {
    return config_.confidence_threshold > 0.0f ? config_.confidence_threshold
                                             : shared_.confidence_threshold;
}

std::vector<Detection> PipelineSession::decodeRawDetections(
    const std::vector<RawDetection>& raw, float scale, int pad_x, int pad_y,
    int orig_width, int orig_height) const {
    std::vector<Detection> candidates;
    candidates.reserve(raw.size());
    for (const auto& rd : raw) {
        float x1 = (rd.cx - rd.w / 2.0f - static_cast<float>(pad_x)) / scale;
        float y1 = (rd.cy - rd.h / 2.0f - static_cast<float>(pad_y)) / scale;
        float x2 = (rd.cx + rd.w / 2.0f - static_cast<float>(pad_x)) / scale;
        float y2 = (rd.cy + rd.h / 2.0f - static_cast<float>(pad_y)) / scale;
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

void PipelineSession::cleanupResources() {
    if (resultHandler_) {
        resultHandler_->closeAll();
    }
    if (shared_.triton_client && memoryPool_) {
        if (!shmYoloIn_.empty()) {
            shared_.triton_client->unregisterCudaShm(shmYoloIn_);
        }
        if (!shmYoloOut_.empty()) {
            shared_.triton_client->unregisterCudaShm(shmYoloOut_);
        }
        if (!shmArcIn_.empty()) {
            shared_.triton_client->unregisterCudaShm(shmArcIn_);
        }
        if (!shmArcOut_.empty()) {
            shared_.triton_client->unregisterCudaShm(shmArcOut_);
        }
    }
    shmYoloIn_.clear();
    shmYoloOut_.clear();
    shmArcIn_.clear();
    shmArcOut_.clear();
    if (memoryPool_) {
        memoryPool_->release();
    }
    if (cpuMemoryPool_) {
        cpuMemoryPool_->release();
    }
    if (frameSource_) {
        frameSource_->close();
    }
    preprocessor_.reset();
    cpuPreprocessor_.reset();
    cpuFaceMatcher_.reset();
    memoryPool_.reset();
    cpuMemoryPool_.reset();
    frameSource_.reset();
    resultHandler_.reset();
}
