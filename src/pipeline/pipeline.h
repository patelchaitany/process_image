#pragma once

#include "utils/config.h"
#include "frame_source/ffmpeg_source.h"
#include "gpu/memory_pool.h"
#include "gpu/cpu_memory_pool.h"
#include "gpu/preprocessor.h"
#include "gpu/cpu_preprocessor.h"
#include "inference/triton_client.h"
#include "inference/face_detector.h"
#include "inference/face_recognizer.h"
#include "matching/face_database.h"
#include "matching/face_matcher.h"
#include "matching/cpu_face_matcher.h"
#include "postprocess/nms.h"
#include "postprocess/face_cropper.h"
#include "pipeline/result_handler.h"
#include "metrics/metrics_logger.h"
#include "metrics/frame_metrics.h"
#include "utils/timer.h"

#include <atomic>
#include <memory>

class Pipeline {
public:
    Pipeline() = default;
    ~Pipeline();

    bool init(const PipelineConfig& config);
    void run(std::atomic<bool>& running);
    void shutdown();

private:
    PipelineConfig config_;
    bool initialized_ = false;
    bool use_gpu_ = true;

    // Shared components
    std::unique_ptr<FFmpegSource> frame_source_;
    std::unique_ptr<TritonClient> triton_client_;
    std::unique_ptr<FaceDetector> face_detector_;
    std::unique_ptr<FaceRecognizer> face_recognizer_;
    std::unique_ptr<FaceDatabase> face_database_;
    std::unique_ptr<FaceCropper> face_cropper_;
    std::unique_ptr<ResultHandler> result_handler_;
    std::unique_ptr<MetricsLogger> metrics_logger_;

    // GPU-specific
    std::unique_ptr<GPUMemoryPool> memory_pool_;
    std::unique_ptr<Preprocessor> preprocessor_;
    std::unique_ptr<FaceMatcher> face_matcher_;

    // CPU-specific
    std::unique_ptr<CPUMemoryPool> cpu_memory_pool_;
    std::unique_ptr<CPUPreprocessor> cpu_preprocessor_;
    std::unique_ptr<CPUFaceMatcher> cpu_face_matcher_;

    bool detect_device();
    void process_frame_gpu(Frame& frame, FrameMetrics& metrics);
    void process_frame_cpu(Frame& frame, FrameMetrics& metrics);
    std::string utc_timestamp() const;

    std::vector<Detection> decode_raw_detections(
        const std::vector<RawDetection>& raw, float scale, int pad_x, int pad_y,
        int orig_width, int orig_height) const;
};
