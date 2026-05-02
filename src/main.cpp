#include <cstdio>
#include <csignal>
#include <atomic>

#include "utils/config.h"
#include "pipeline/pipeline.h"

static std::atomic<bool> g_running{true};

static void signal_handler(int sig) {
    (void)sig;
    g_running.store(false, std::memory_order_relaxed);
    fprintf(stderr, "\nInterrupted. Shutting down...\n");
}

int main(int argc, char* argv[]) {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    PipelineConfig config = parse_args(argc, argv);

    printf("process_image v0.1\n");
    printf("  Input:       %s\n", config.input_source.c_str());
    printf("  Triton:      %s\n", config.triton_url.c_str());
    printf("  Device:      %s\n", config.device_mode.c_str());
    printf("  DB:          %s\n", config.db_path.c_str());
    printf("  Metrics CSV: %s\n", config.output_csv.c_str());
    printf("  YOLO model:  %s\n", config.yolo_model.c_str());
    printf("  ArcFace:     %s\n", config.arcface_model.c_str());
    printf("  Confidence:  %.2f\n", config.confidence_threshold);
    printf("  Match thr:   %.2f\n", config.match_threshold);
    printf("\n");

    Pipeline pipeline;
    if (!pipeline.init(config)) {
        fprintf(stderr, "Pipeline initialization failed\n");
        return 1;
    }

    pipeline.run(g_running);
    pipeline.shutdown();

    printf("Done.\n");
    return 0;
}
