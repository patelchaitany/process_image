#pragma once

#include "utils/config.h"
#include "frame_source/ffmpeg_source.h"
#include "gpu/memory_pool.h"
#include "gpu/preprocessor.h"
#include "inference/triton_client.h"
#include "inference/face_detector.h"
#include "inference/face_recognizer.h"
#include "matching/face_database.h"
#include "matching/face_matcher.h"
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

    std::unique_ptr<FFmpegSource> frame_source_;
    std::unique_ptr<GPUMemoryPool> memory_pool_;
    std::unique_ptr<Preprocessor> preprocessor_;
    std::unique_ptr<TritonClient> triton_client_;
    std::unique_ptr<FaceDetector> face_detector_;
    std::unique_ptr<FaceRecognizer> face_recognizer_;
    std::unique_ptr<FaceDatabase> face_database_;
    std::unique_ptr<FaceMatcher> face_matcher_;
    std::unique_ptr<FaceCropper> face_cropper_;
    std::unique_ptr<ResultHandler> result_handler_;
    std::unique_ptr<MetricsLogger> metrics_logger_;

    bool detect_device();
    void process_frame(Frame& frame, FrameMetrics& metrics);
};
