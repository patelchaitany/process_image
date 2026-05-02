#pragma once

#include <string>

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
        } else if (arg == "--output-csv" && i + 1 < argc) {
            config.output_csv = argv[++i];
        } else if (arg == "--db" && i + 1 < argc) {
            config.db_path = argv[++i];
        } else if (arg == "--confidence" && i + 1 < argc) {
            config.confidence_threshold = std::stof(argv[++i]);
        } else if (arg == "--match-threshold" && i + 1 < argc) {
            config.match_threshold = std::stof(argv[++i]);
        } else if (arg == "--device" && i + 1 < argc) {
            config.device_mode = argv[++i];
        } else if (arg == "--yolo-model" && i + 1 < argc) {
            config.yolo_model = argv[++i];
        } else if (arg == "--arcface-model" && i + 1 < argc) {
            config.arcface_model = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            fprintf(stderr,
                "Usage: process_image --input <source> --triton <url> [options]\n\n"
                "Required:\n"
                "  --input <path|url>        Input source: MP4 file path or RTSP URL\n"
                "  --triton <url>            Triton server URL (e.g. localhost:8001)\n\n"
                "Optional:\n"
                "  --output-csv <path>       Metrics CSV output path (default: ./metrics/metrics.csv)\n"
                "  --db <path>              SQLite face database path (default: ./faces.db)\n"
                "  --confidence <float>      YOLO detection confidence threshold (default: 0.5)\n"
                "  --match-threshold <float> Face match similarity threshold (default: 0.6)\n"
                "  --device <gpu|cpu>        Force device mode (default: auto-detect)\n"
                "  --yolo-model <name>       Triton model name for detection (default: yolo26_face)\n"
                "  --arcface-model <name>    Triton model name for recognition (default: arcface)\n"
            );
            exit(0);
        }
    }

    if (config.input_source.empty()) {
        fprintf(stderr, "Error: --input is required\n");
        exit(1);
    }

    return config;
}
