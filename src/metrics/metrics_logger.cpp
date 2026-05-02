#include "metrics/metrics_logger.h"
#include <cstdio>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <sstream>

MetricsLogger::~MetricsLogger() {
    shutdown();
}

bool MetricsLogger::init(const std::string& output_dir, int flush_interval_ms,
                          int rotate_size_mb) {
    output_dir_ = output_dir;
    flush_interval_ms_ = flush_interval_ms;
    rotate_size_bytes_ = static_cast<size_t>(rotate_size_mb) * 1024 * 1024;

    std::filesystem::create_directories(output_dir_);

    std::string filename = generate_filename();
    file_.open(filename, std::ios::out | std::ios::app);
    if (!file_.is_open()) {
        fprintf(stderr, "MetricsLogger: cannot open %s\n", filename.c_str());
        return false;
    }

    running_ = true;
    writer_thread_ = std::thread(&MetricsLogger::writer_thread_func, this);
    return true;
}

void MetricsLogger::shutdown() {
    running_ = false;
    if (writer_thread_.joinable()) {
        writer_thread_.join();
    }

    FrameMetrics m;
    while (buffer_.pop(m)) {
        if (file_.is_open()) {
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
        }
    }

    if (file_.is_open()) {
        file_.flush();
        file_.close();
    }
}

void MetricsLogger::log_frame(const FrameMetrics& metrics) {
    buffer_.push(metrics);
}

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

void MetricsLogger::rotate_file_if_needed() {
    current_size_ = static_cast<size_t>(file_.tellp());
    if (current_size_ >= rotate_size_bytes_) {
        file_.close();
        std::string filename = generate_filename();
        file_.open(filename, std::ios::out);
        header_written_ = false;
        current_size_ = 0;
    }
}

std::string MetricsLogger::generate_filename() const {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&time, &tm);

    std::ostringstream ss;
    ss << output_dir_ << "/metrics_"
       << std::put_time(&tm, "%Y-%m-%d_%H%M%S") << ".csv";
    return ss.str();
}
