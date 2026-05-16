#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "service/session_manager.h"
#include "service/grpc_server.h"

static GrpcServer* g_server = nullptr;

static void signalHandler(int sig) {
    (void)sig;
    fprintf(stderr, "\nInterrupted. Shutting down...\n");
    if (g_server) g_server->shutdown();
}

static ServiceConfig parseServiceArgs(int argc, char* argv[]) {
    ServiceConfig config;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--triton" && i + 1 < argc) {
            config.triton_url = argv[++i];
        } else if (arg == "--db" && i + 1 < argc) {
            config.db_path = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            config.grpc_port = std::atoi(argv[++i]);
        } else if (arg == "--max-sessions" && i + 1 < argc) {
            config.max_sessions = std::atoi(argv[++i]);
        } else if (arg == "--device" && i + 1 < argc) {
            config.device_mode = argv[++i];
        } else if (arg == "--yolo-model" && i + 1 < argc) {
            config.yolo_model = argv[++i];
        } else if (arg == "--arcface-model" && i + 1 < argc) {
            config.arcface_model = argv[++i];
        } else if (arg == "--confidence" && i + 1 < argc) {
            config.confidence_threshold = std::stof(argv[++i]);
        } else if (arg == "--match-threshold" && i + 1 < argc) {
            config.match_threshold = std::stof(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            fprintf(stderr,
                "Usage: video_analytics_service [options]\n\n"
                "Options:\n"
                "  --triton <url>            Triton server URL (default: localhost:8001)\n"
                "  --db <path>               Face database path (default: ./faces.db)\n"
                "  --port <int>              gRPC listen port (default: 50051)\n"
                "  --max-sessions <int>      Max concurrent sessions (default: 30)\n"
                "  --device <gpu|cpu|auto>   Device mode (default: auto)\n"
                "  --yolo-model <name>       Triton YOLO model name (default: yolo26_face)\n"
                "  --arcface-model <name>    Triton ArcFace model name (default: arcface)\n"
                "  --confidence <float>      Default detection threshold (default: 0.5)\n"
                "  --match-threshold <float> Default match threshold (default: 0.6)\n"
            );
            std::exit(0);
        }
    }
    return config;
}

int main(int argc, char* argv[]) {
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);
    ServiceConfig config = parseServiceArgs(argc, argv);
    printf("video_analytics_service v0.1\n");
    printf("  Triton:        %s\n", config.triton_url.c_str());
    printf("  DB:            %s\n", config.db_path.c_str());
    printf("  gRPC port:     %d\n", config.grpc_port);
    printf("  Max sessions:  %d\n", config.max_sessions);
    printf("  Device:        %s\n", config.device_mode.c_str());
    printf("\n");
    SessionManager manager;
    if (!manager.init(config)) {
        fprintf(stderr, "Failed to initialize session manager\n");
        return 1;
    }
    GrpcServer server(manager);
    g_server = &server;
    server.run(config.grpc_port);
    g_server = nullptr;
    manager.shutdownAll();
    printf("Done.\n");
    return 0;
}
