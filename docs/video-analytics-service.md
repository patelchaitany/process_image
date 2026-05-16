# Video Analytics Service

gRPC service for multi-camera face detection and recognition. Accepts RTSP streams or MP4 files, runs YOLO face detection + ArcFace recognition via Triton, and delivers results (frame_id, bounding boxes, person identities) to client-provided webhook URLs.

## Architecture

```
Clients                          Service                              Triton
-------                          -------                              ------

Client 1 --StartSession-->  +------------------+
Client 2 --StopSession--->  |   gRPC Server    |
Client N --ListSessions-->  |   (port 50051)   |
                            +--------+---------+
                                     |
                            +--------v---------+
                            | SessionManager   |
                            | (up to 30 cams)  |
                            +--------+---------+
                                     |
                     +---------------+---------------+
                     |               |               |
              +------v------+ +------v------+ +------v------+
              | Session A   | | Session B   | | Session N   |
              | cam1 RTSP   | | cam2 RTSP   | | file.mp4    |
              | own decoder | | own decoder | | own decoder |
              | own GPU buf | | own GPU buf | | own GPU buf |
              | own thread  | | own thread  | | own thread  |
              +------+------+ +------+------+ +------+------+
                     |               |               |
                     +-------+-------+-------+-------+
                             |               |
                    +--------v--------+  +---v---------+
                    | SharedResources |  | Webhook     |
                    | TritonClient    |  | HTTP POST   |
                    | FaceDatabase    |  | per session |
                    | FaceMatcher     |  +------+------+
                    +---------+-------+         |
                              |                 v
                     +--------v--------+   Client webhook
                     | Triton Server   |   endpoint
                     | YOLO + ArcFace  |
                     +-----------------+
```

### Design decisions

- **Triton stays** as inference backend. Shared by all sessions. Dynamic batching groups requests from multiple cameras into efficient GPU batches.
- **Webhook delivery** instead of gRPC streaming. Each session POSTs JSON results to the client's callback URL.
- **Per-session isolation** for decode (FrameSource), GPU buffers (GPUMemoryPool), and CUDA streams. Each camera runs in its own thread.
- **Shared resources** for read-only components: TritonClient, FaceDatabase, FaceMatcher (FAISS index). Initialized once, referenced by all sessions.
- **No frame pixels** in responses. Only frame_id + detection metadata + person identities.
- **Separate binary** (`video_analytics_service`) alongside existing `process_image` CLI.

### Scaling characteristics (30 cameras)

| Resource | Shared/Per-session | Scaling concern |
|----------|-------------------|-----------------|
| NVDEC decode | Per-session (1 session per GPU) | ~30 concurrent sessions on A100 |
| GPU memory for buffers | Per-session (~20 MB each) | ~600 MB for 30 sessions |
| Triton inference | Shared (dynamic batching) | Triton batches across sessions |
| FAISS index | Shared (one copy in GPU RAM) | No duplication |
| CPU threads | Per-session (1 thread each) | 30 threads, mostly sleeping |

## API

### gRPC Service

Defined in `proto/analytics.proto`, package `videoanalytics.v1`.

#### StartSession

Creates a new pipeline session for a camera or video file.

```protobuf
rpc StartSession(StartSessionRequest) returns (StartSessionResponse);

message StartSessionRequest {
  string source_uri = 1;       // RTSP URL or MP4 file path
  string callback_url = 2;     // Webhook URL for result delivery
  string session_name = 3;     // Optional human-readable name
  SessionConfig config = 4;    // Optional per-session overrides
}

message SessionConfig {
  float confidence_threshold = 1;  // 0 = use service default (0.5)
  float match_threshold = 2;       // 0 = use service default (0.6)
  int32 max_fps = 3;               // 0 = no limit
}

message StartSessionResponse {
  string session_id = 1;       // UUID assigned by service
  string status = 2;           // "running" or "error"
  string error = 3;            // populated if status == "error"
}
```

#### StopSession

Stops and removes a running session.

```protobuf
rpc StopSession(StopSessionRequest) returns (StopSessionResponse);

message StopSessionRequest { string session_id = 1; }
message StopSessionResponse {
  bool success = 1;
  uint64 frames_processed = 2;
}
```

#### ListSessions / GetSessionStatus

```protobuf
rpc ListSessions(ListSessionsRequest) returns (ListSessionsResponse);
rpc GetSessionStatus(GetSessionStatusRequest) returns (SessionStatus);

message SessionStatus {
  string session_id = 1;
  string session_name = 2;
  string source_uri = 3;
  string state = 4;            // starting, running, stopped, error, finished
  uint64 frames_processed = 5;
  uint64 frames_dropped = 6;
  double uptime_seconds = 7;
}
```

### Webhook Result Format

Each processed frame triggers an HTTP POST to the session's `callback_url`:

```
POST <callback_url>
Content-Type: application/json

{
  "session_id": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d",
  "frame_id": 42,
  "timestamp_utc": "2026-05-04T12:30:45.123Z",
  "source_id": "rtsp://192.168.1.100:554/stream",
  "detections": [
    {
      "bbox": {"x1": 120.5, "y1": 80.2, "x2": 200.1, "y2": 220.8},
      "confidence": 0.93,
      "match": {
        "person_id": 17,
        "name": "John Doe",
        "similarity": 0.85
      }
    },
    {
      "bbox": {"x1": 400.0, "y1": 90.0, "x2": 480.0, "y2": 230.0},
      "confidence": 0.78,
      "match": null
    }
  ]
}
```

- `frame_id` is the sequential frame counter from the video decoder. Use it to correlate results with your own video display.
- `match` is `null` when the face is detected but not recognized (below similarity threshold or not in the gallery).
- Webhook delivery is async with retry (3 attempts with 100ms/500ms/1000ms backoff). Frames are never dropped due to webhook failures blocking the pipeline.

## Components

### Dependency graph

```
                  +-------------------+
                  | G: service_main   |
                  | + CMakeLists.txt  |
                  +--------+----------+
                           |
              +------------+------------+
              |                         |
     +--------v--------+      +--------v--------+
     | F: GrpcServer    |      | E: SessionManager|
     | (proto mapping)  |      | (lifecycle mgmt) |
     +--------+---------+      +--------+---------+
              |                         |
              |                +--------+--------+
              |                |                 |
     +--------v--------+  +---v---+       +-----v-----+
     | A: analytics     |  | D: PipelineSession       |
     |    .proto        |  | (per-camera worker)      |
     +------------------+  +---+-------+---+----------+
                               |       |
                          +----v--+ +--v-----------+
                          | C:    | | B: Webhook   |
                          |Shared | | ResultWriter |
                          |Res.   | +--------------+
                          +-------+
```

Build order: A, B, C (parallel, no deps) -> D -> E -> F -> G.

### Component A: Proto Definition

| | |
|---|---|
| **File** | `proto/analytics.proto` |
| **Dependencies** | None |
| **Produces** | Generated `analytics.pb.h/.cc`, `analytics.grpc.pb.h/.cc` |
| **Consumed by** | Component F (GrpcServer inherits from generated service base) |

### Component B: WebhookResultWriter

| | |
|---|---|
| **Files** | `src/output/webhook_result_writer.h`, `src/output/webhook_result_writer.cpp` |
| **Implements** | `ResultWriter` interface from `src/output/result_writer.h` |
| **External deps** | libcurl |
| **Consumed by** | Component D (one WebhookResultWriter per session) |

Async HTTP POST writer. `write()` is non-blocking: serializes FrameResult to JSON, pushes to internal queue. Background thread drains queue, POSTs to callback_url with retries.

**Input:** `write(const FrameResult&)` called per frame.

**Output:** HTTP POST JSON to `callback_url`.

**Thread safety:** Queue protected by mutex + condition_variable. Running flag is atomic.

### Component C: SharedResources

| | |
|---|---|
| **Files** | `src/service/shared_resources.h`, `src/service/shared_resources.cpp` |
| **Owns** | TritonClient, FaceDatabase, FaceMatcher, FaceDetector, FaceRecognizer, FaceCropper |
| **Consumed by** | Component D (by reference), Component E (owns instance) |

Initialized once at service startup. All members are read-only after init and thread-safe for concurrent access from multiple session threads.

**Init sequence:**
1. Detect GPU (cuda/cpu/auto)
2. Connect TritonClient
3. Open FaceDatabase, load gallery
4. Build FAISS index via FaceMatcher
5. Create FaceDetector, FaceRecognizer, FaceCropper wrappers

**Thread safety:** TritonClient gRPC is concurrent-safe. FaceMatcher FAISS search is read-only. FaceDetector/FaceRecognizer are stateless wrappers.

Also defines `ServiceConfig`:

```cpp
struct ServiceConfig {
    std::string triton_url = "localhost:8001";
    std::string db_path = "./faces.db";
    std::string device_mode = "auto";   // gpu, cpu, auto
    std::string yolo_model = "yolo26_face";
    std::string arcface_model = "arcface";
    float confidence_threshold = 0.5f;
    float match_threshold = 0.6f;
    float nms_iou_threshold = 0.45f;
    int yolo_input_size = 640;
    int max_faces_per_frame = 32;
    int max_sessions = 30;
    int grpc_port = 50051;
};
```

### Component D: PipelineSession

| | |
|---|---|
| **Files** | `src/service/pipeline_session.h`, `src/service/pipeline_session.cpp` |
| **Owns** | FrameSource, GPUMemoryPool, Preprocessor, ResultHandler, worker thread |
| **Uses** | SharedResources (by reference), WebhookResultWriter |
| **Consumed by** | Component E (SessionManager creates/destroys instances) |

One camera or video file. Runs decode-preprocess-detect-recognize-match in a dedicated thread using per-session GPU buffers and shared inference resources.

**Per-session SHM naming:** Each session registers GPU memory with Triton using unique names (`yolo_input_shm_{session_id}`, etc.) to avoid conflicts between concurrent sessions.

**Frame processing flow (GPU path):**
1. Decode frame via NvdecSource (or FFmpeg CPU fallback)
2. Upload/copy to session's GPUMemoryPool
3. Preprocess: letterbox + normalize to 640x640
4. YOLO detection via shared FaceDetector (Triton SHM)
5. NMS on detections
6. Crop faces via shared FaceCropper
7. ArcFace recognition via shared FaceRecognizer (Triton SHM)
8. Match embeddings against gallery via shared FaceMatcher (FAISS)
9. Build FrameResult, dispatch to WebhookResultWriter

**Lifecycle states:** starting -> running -> finished (MP4 EOF) or error.

Also defines `SessionConfig` and `SessionStatus` structs.

### Component E: SessionManager

| | |
|---|---|
| **Files** | `src/service/session_manager.h`, `src/service/session_manager.cpp` |
| **Owns** | SharedResources, map of PipelineSession instances |
| **Consumed by** | Component F (GrpcServer calls SessionManager methods) |

Manages lifecycle of up to 30 concurrent PipelineSession instances. Generates UUID v4 session IDs. Thread-safe: multiple gRPC threads may call methods concurrently.

**Public API:**
- `init(ServiceConfig)` -- initialize shared resources
- `startSession(source_uri, callback_url, name, config)` -> StartResult
- `stopSession(session_id)` -> StopResult
- `listSessions()` -> vector of SessionStatus
- `getSessionStatus(session_id)` -> SessionStatus
- `shutdownAll()` -- stop all sessions, release shared resources

Also defines `StartResult` and `StopResult` structs.

### Component F: GrpcServer

| | |
|---|---|
| **Files** | `src/service/grpc_server.h`, `src/service/grpc_server.cpp` |
| **Inherits** | `videoanalytics::v1::VideoAnalytics::Service` (generated) |
| **Uses** | SessionManager (by reference) |
| **Consumed by** | Component G (service_main.cpp) |

Thin gRPC adapter. Each RPC extracts request fields, calls the corresponding SessionManager method, fills the response. No business logic.

### Component G: Service Entry Point + Build

| | |
|---|---|
| **Files** | `src/service_main.cpp`, `CMakeLists.txt` |
| **Dependencies** | All components |

CLI entry point for the service binary. Parses `--triton`, `--db`, `--port`, `--max-sessions`, etc. Initializes SessionManager, starts GrpcServer, blocks until SIGINT/SIGTERM.

CMakeLists.txt adds a `video_analytics_service` target conditionally built when gRPC, Protobuf, and CURL are found. The existing `process_image` CLI target is unchanged.

## File Layout

```
proto/
  analytics.proto                     # gRPC service definition

src/
  main.cpp                            # existing CLI entry point (unchanged)
  service_main.cpp                    # service entry point

  service/
    shared_resources.h                # SharedResources + ServiceConfig
    shared_resources.cpp
    pipeline_session.h                # PipelineSession + SessionConfig + SessionStatus
    pipeline_session.cpp
    session_manager.h                 # SessionManager + StartResult + StopResult
    session_manager.cpp
    grpc_server.h                     # GrpcServer
    grpc_server.cpp

  output/
    webhook_result_writer.h           # WebhookResultWriter
    webhook_result_writer.cpp
    result_writer.h                   # existing ResultWriter interface (unchanged)
    csv_result_writer.h/.cpp          # existing (unchanged)
    console_result_writer.h/.cpp      # existing (unchanged)

  pipeline/
    pipeline.h/.cpp                   # existing Pipeline (unchanged, CLI only)
    result_handler.h/.cpp             # existing ResultHandler (shared)

  frame_source/                       # existing (unchanged, shared)
  gpu/                                # existing (unchanged, shared)
  inference/                          # existing (unchanged, shared)
  matching/                           # existing (unchanged, shared)
  postprocess/                        # existing (unchanged, shared)
  metrics/                            # existing (unchanged, shared)
```

## Build

### Prerequisites

```bash
# Required (already present for process_image)
# CUDA Toolkit, FFmpeg, SQLite3, Triton Client SDK, FAISS

# New dependencies for the service
sudo apt install libgrpc++-dev protobuf-compiler-grpc libcurl4-openssl-dev
```

### Compile

```bash
mkdir -p build && cd build
cmake ..

# CLI tool (same as before)
make process_image

# Service binary (requires gRPC + Protobuf + CURL)
make video_analytics_service
```

If gRPC/Protobuf/CURL are not found, `video_analytics_service` is silently skipped and `process_image` builds as before.

## Usage

### Start the service

```bash
./video_analytics_service \
  --triton localhost:8001 \
  --db ./faces.db \
  --port 50051 \
  --max-sessions 30 \
  --device auto
```

### CLI options

```
--triton <url>            Triton server URL (default: localhost:8001)
--db <path>               Face database path (default: ./faces.db)
--port <int>              gRPC listen port (default: 50051)
--max-sessions <int>      Max concurrent sessions (default: 30)
--device <gpu|cpu|auto>   Device mode (default: auto)
--yolo-model <name>       Triton YOLO model name (default: yolo26_face)
--arcface-model <name>    Triton ArcFace model name (default: arcface)
--confidence <float>      Default detection threshold (default: 0.5)
--match-threshold <float> Default match threshold (default: 0.6)
```

### Client usage (grpcurl examples)

```bash
# Start a session
grpcurl -plaintext -d '{
  "source_uri": "rtsp://192.168.1.100:554/stream",
  "callback_url": "http://myapp:8080/webhook",
  "session_name": "lobby-camera"
}' localhost:50051 videoanalytics.v1.VideoAnalytics/StartSession

# List all sessions
grpcurl -plaintext localhost:50051 videoanalytics.v1.VideoAnalytics/ListSessions

# Check session status
grpcurl -plaintext -d '{"session_id": "a1b2c3d4-..."}' \
  localhost:50051 videoanalytics.v1.VideoAnalytics/GetSessionStatus

# Stop a session
grpcurl -plaintext -d '{"session_id": "a1b2c3d4-..."}' \
  localhost:50051 videoanalytics.v1.VideoAnalytics/StopSession
```

### Receiving webhook results

Set up an HTTP endpoint to receive POST requests:

```python
from flask import Flask, request

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def receive_result():
    data = request.json
    frame_id = data['frame_id']
    for det in data['detections']:
        bbox = det['bbox']
        match = det.get('match')
        if match:
            print(f"Frame {frame_id}: {match['name']} at ({bbox['x1']:.0f},{bbox['y1']:.0f})")
        else:
            print(f"Frame {frame_id}: unknown face at ({bbox['x1']:.0f},{bbox['y1']:.0f})")
    return '', 200

app.run(host='0.0.0.0', port=8080)
```
