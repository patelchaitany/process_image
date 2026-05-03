#pragma once

#include <string>
#include <cstdio>

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

    int faiss_nprobe = 16;
    int faiss_ivf_threshold = 1000;

    std::string bbox_csv = "./output/detections.csv";
    int bbox_csv_rotate_mb = 100;
    bool console_verbose = true;

    std::string config_file;
};

#ifdef HAS_YAML_CPP
#include <yaml-cpp/yaml.h>

/// @brief Load pipeline.yaml and apply values to config (does not override
/// fields already set by CLI arguments).
inline void load_config_yaml(const std::string& path, PipelineConfig& config) {
    YAML::Node root;
    try {
        root = YAML::LoadFile(path);
    } catch (const YAML::Exception& e) {
        fprintf(stderr, "Warning: could not load config '%s': %s\n", path.c_str(), e.what());
        return;
    }

    if (auto m = root["metrics"]) {
        if (m["output_dir"] && config.output_csv == "./metrics/metrics.csv")
            config.output_csv = m["output_dir"].as<std::string>() + "/metrics.csv";
    }
    if (auto p = root["pipeline"]) {
        if (p["frame_width"])          config.frame_width          = p["frame_width"].as<int>();
        if (p["frame_height"])         config.frame_height         = p["frame_height"].as<int>();
        if (p["yolo_input_size"])      config.yolo_input_size      = p["yolo_input_size"].as<int>();
        if (p["arcface_input_size"])   config.arcface_input_size   = p["arcface_input_size"].as<int>();
        if (p["max_faces_per_frame"])  config.max_faces_per_frame  = p["max_faces_per_frame"].as<int>();
    }
    if (auto d = root["detection"]) {
        if (d["confidence_threshold"]) config.confidence_threshold = d["confidence_threshold"].as<float>();
        if (d["nms_iou_threshold"])    config.nms_iou_threshold    = d["nms_iou_threshold"].as<float>();
    }
    if (auto mt = root["matching"]) {
        if (mt["similarity_threshold"]) config.match_threshold      = mt["similarity_threshold"].as<float>();
        if (mt["faiss_ivf_threshold"])  config.faiss_ivf_threshold  = mt["faiss_ivf_threshold"].as<int>();
        if (mt["faiss_nprobe"])         config.faiss_nprobe         = mt["faiss_nprobe"].as<int>();
    }

    fprintf(stderr, "Config loaded from %s\n", path.c_str());
}
#else
inline void load_config_yaml(const std::string& path, PipelineConfig& /*config*/) {
    fprintf(stderr, "Warning: yaml-cpp not available, ignoring config file '%s'\n", path.c_str());
}
#endif

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
        } else if (arg == "--bbox-csv" && i + 1 < argc) {
            config.bbox_csv = argv[++i];
        } else if (arg == "--no-console") {
            config.console_verbose = false;
        } else if (arg == "--config" && i + 1 < argc) {
            config.config_file = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            fprintf(stderr,
                "Usage: process_image --input <source> --triton <url> [options]\n\n"
                "Required:\n"
                "  --input <path|url>        Input source: MP4 file path or RTSP URL\n"
                "  --triton <url>            Triton server URL (e.g. localhost:8001)\n\n"
                "Optional:\n"
                "  --config <path>           YAML config file (default: config/pipeline.yaml)\n"
                "  --output-csv <path>       Metrics CSV output path (default: ./metrics/metrics.csv)\n"
                "  --db <path>              SQLite face database path (default: ./faces.db)\n"
                "  --confidence <float>      YOLO detection confidence threshold (default: 0.5)\n"
                "  --match-threshold <float> Face match similarity threshold (default: 0.6)\n"
                "  --bbox-csv <path>         Per-detection bbox CSV path (default: ./output/detections.csv)\n"
                "  --no-console              Suppress per-frame console output\n"
                "  --device <gpu|cpu|auto>   Force device mode (default: auto-detect)\n"
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

    if (!config.config_file.empty()) {
        load_config_yaml(config.config_file, config);
    }

    return config;
}
