#pragma once

#include <cstdint>
#include <string>

struct FrameMetrics {
    uint64_t frame_id = 0;
    std::string timestamp_utc;
    std::string source_id;

    float decode_ms = 0;
    float yuv_to_bgr_ms = 0;
    float cpu_to_gpu_ms = 0;

    float preprocess_resize_ms = 0;
    float preprocess_color_ms = 0;
    float preprocess_transpose_ms = 0;
    float preprocess_normalize_ms = 0;
    float preprocess_total_ms = 0;

    float grpc1_overhead_ms = 0;
    float yolo_inference_ms = 0;
    float triton_yolo_queue_ms = 0;
    float triton_yolo_compute_ms = 0;

    float yolo_output_copy_ms = 0;
    float confidence_filter_ms = 0;
    float nms_sort_ms = 0;
    float nms_iou_ms = 0;
    float nms_total_ms = 0;
    int faces_detected = 0;

    float face_crop_ms = 0;
    float face_arcface_preprocess_ms = 0;

    float grpc2_overhead_ms = 0;
    float arcface_inference_ms = 0;
    float triton_arcface_queue_ms = 0;
    float triton_arcface_compute_ms = 0;

    float faiss_normalize_ms = 0;
    float faiss_search_ms = 0;

    float result_copy_ms = 0;
    float total_pipeline_ms = 0;
    float gpu_memory_used_mb = 0;
};
