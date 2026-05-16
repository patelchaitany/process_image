# API Reference

Video Analytics Service exposes a **gRPC API** for managing face detection and recognition sessions over video streams or files. Results are delivered asynchronously to client-provided **webhook URLs** via HTTP POST.

- **Transport:** gRPC (HTTP/2)
- **Default port:** 50051
- **Proto package:** `videoanalytics.v1`
- **Proto file:** [`proto/analytics.proto`](../proto/analytics.proto)

---

## Table of Contents

- [Starting the Service](#starting-the-service)
- [RPC Endpoints](#rpc-endpoints)
  - [StartSession](#startsession)
  - [StopSession](#stopsession)
  - [ListSessions](#listsessions)
  - [GetSessionStatus](#getsessionstatus)
- [Webhook Result Delivery](#webhook-result-delivery)
- [Data Types](#data-types)
- [Error Handling](#error-handling)
- [Client Examples](#client-examples)
  - [grpcurl](#grpcurl)
  - [Python](#python)
  - [Receiving Webhooks](#receiving-webhooks)

---

## Starting the Service

```bash
./video_analytics_service [options]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--triton <url>` | string | `localhost:8001` | Triton Inference Server gRPC URL |
| `--db <path>` | string | `./faces.db` | SQLite face gallery database path |
| `--port <int>` | int | `50051` | gRPC listen port |
| `--max-sessions <int>` | int | `30` | Maximum concurrent sessions |
| `--device <mode>` | string | `auto` | Device mode: `gpu`, `cpu`, or `auto` |
| `--yolo-model <name>` | string | `yolo26_face` | Triton YOLO model name |
| `--arcface-model <name>` | string | `arcface` | Triton ArcFace model name |
| `--confidence <float>` | float | `0.5` | Default face detection confidence threshold |
| `--match-threshold <float>` | float | `0.6` | Default face match similarity threshold |

The service binds to `0.0.0.0:<port>` and blocks until `SIGINT` or `SIGTERM`.

---

## RPC Endpoints

### StartSession

Creates a new pipeline session for a camera stream or video file. The service begins decoding, detecting faces, recognizing identities, and posting results to the callback URL immediately.

| | |
|---|---|
| **Method** | `videoanalytics.v1.VideoAnalytics/StartSession` |
| **Request** | `StartSessionRequest` |
| **Response** | `StartSessionResponse` |

#### Request Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `source_uri` | string | Yes | RTSP URL (e.g. `rtsp://192.168.1.100:554/stream`) or local file path (e.g. `/data/video.mp4`) |
| `callback_url` | string | Yes | HTTP(S) URL where the service will POST detection results as JSON |
| `session_name` | string | No | Human-readable label for the session (e.g. `lobby-camera`) |
| `config` | `SessionConfig` | No | Per-session overrides for thresholds and frame rate |

#### SessionConfig Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `confidence_threshold` | float | `0` (use service default: 0.5) | Minimum YOLO detection confidence. Faces below this score are discarded. |
| `match_threshold` | float | `0` (use service default: 0.6) | Minimum cosine similarity for a face to be considered a match against the gallery. |
| `max_fps` | int32 | `0` (no limit) | Maximum frames per second to process. Excess frames are dropped. |

#### Response Fields

| Field | Type | Description |
|---|---|---|
| `session_id` | string | UUID v4 assigned by the service. Use this to stop or query the session. |
| `status` | string | `"running"` if the session started successfully, `"error"` otherwise. |
| `error` | string | Error message when `status` is `"error"`. Empty on success. |

#### Error Conditions

| Condition | `status` | `error` |
|---|---|---|
| Max sessions reached (default 30) | `"error"` | `"max sessions reached"` |
| Invalid or unreachable `source_uri` | `"error"` | `"failed to open source: <detail>"` |
| Empty `source_uri` or `callback_url` | `"error"` | `"source_uri and callback_url are required"` |

---

### StopSession

Stops and removes a running session. The pipeline thread is joined, per-session GPU buffers are freed, and the webhook writer is drained and closed.

| | |
|---|---|
| **Method** | `videoanalytics.v1.VideoAnalytics/StopSession` |
| **Request** | `StopSessionRequest` |
| **Response** | `StopSessionResponse` |

#### Request Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `session_id` | string | Yes | The UUID returned by `StartSession`. |

#### Response Fields

| Field | Type | Description |
|---|---|---|
| `success` | bool | `true` if the session was found and stopped. `false` if the session ID was not found. |
| `frames_processed` | uint64 | Total number of frames the session processed before stopping. `0` if session was not found. |

---

### ListSessions

Returns the status of all tracked sessions (running, finished, or errored).

| | |
|---|---|
| **Method** | `videoanalytics.v1.VideoAnalytics/ListSessions` |
| **Request** | `ListSessionsRequest` (empty) |
| **Response** | `ListSessionsResponse` |

#### Request Fields

None.

#### Response Fields

| Field | Type | Description |
|---|---|---|
| `sessions` | repeated `SessionStatus` | Array of status objects, one per tracked session. |

---

### GetSessionStatus

Returns the status of a single session by ID.

| | |
|---|---|
| **Method** | `videoanalytics.v1.VideoAnalytics/GetSessionStatus` |
| **Request** | `GetSessionStatusRequest` |
| **Response** | `SessionStatus` |

#### Request Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `session_id` | string | Yes | The UUID returned by `StartSession`. |

#### Response Fields

See [SessionStatus](#sessionstatus) below. If the session ID is not found, the response contains `state: "unknown"` with all other fields empty or zero.

---

## Webhook Result Delivery

Each processed frame triggers an asynchronous HTTP POST to the session's `callback_url`.

### Request

```
POST <callback_url>
Content-Type: application/json
```

### Payload Schema

```json
{
  "session_id": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d",
  "frame_id": 42,
  "timestamp_utc": "2026-05-04T12:30:45.123Z",
  "source_id": "rtsp://192.168.1.100:554/stream",
  "detections": [
    {
      "bbox": {
        "x1": 120.5,
        "y1": 80.2,
        "x2": 200.1,
        "y2": 220.8
      },
      "confidence": 0.93,
      "match": {
        "person_id": 17,
        "name": "John Doe",
        "similarity": 0.85
      }
    },
    {
      "bbox": {
        "x1": 400.0,
        "y1": 90.0,
        "x2": 480.0,
        "y2": 230.0
      },
      "confidence": 0.78,
      "match": null
    }
  ]
}
```

### Payload Fields

| Field | Type | Description |
|---|---|---|
| `session_id` | string | UUID of the session that produced this result. |
| `frame_id` | uint64 | Sequential frame counter from the video decoder. Use to correlate with your own video timeline. |
| `timestamp_utc` | string | ISO 8601 UTC timestamp of when the frame was processed. |
| `source_id` | string | The `source_uri` that was passed to `StartSession`. |
| `detections` | array | Array of detected faces in the frame. Empty array if no faces detected. |

### Detection Object

| Field | Type | Description |
|---|---|---|
| `bbox` | object | Bounding box in pixel coordinates: `x1`, `y1` (top-left), `x2`, `y2` (bottom-right). |
| `bbox.x1` | float | Left edge x-coordinate. |
| `bbox.y1` | float | Top edge y-coordinate. |
| `bbox.x2` | float | Right edge x-coordinate. |
| `bbox.y2` | float | Bottom edge y-coordinate. |
| `confidence` | float | YOLO detection confidence score (0.0 to 1.0). |
| `match` | object or null | Identity match result. `null` when the face is detected but not recognized (below similarity threshold or not in the gallery). |

### Match Object

Present only when a face matches a known identity in the gallery.

| Field | Type | Description |
|---|---|---|
| `person_id` | int | Internal person ID from the face database. |
| `name` | string | Name of the matched person as enrolled in the gallery. |
| `similarity` | float | Cosine similarity score (0.0 to 1.0). Higher is better. |

### Delivery Behavior

| Property | Detail |
|---|---|
| **Delivery** | Asynchronous. The pipeline never blocks on webhook delivery. |
| **Retries** | 4 attempts total (1 initial + 3 retries) with backoff delays of 100ms, 500ms, 1000ms. |
| **Failure** | After all retries fail, the payload is dropped and a warning is logged. The pipeline continues processing. |
| **Ordering** | Payloads are delivered in frame order per session. Concurrent sessions deliver independently. |
| **Expected response** | Any 2xx HTTP status code. Response body is ignored. |

---

## Data Types

### SessionStatus

Returned by `ListSessions` and `GetSessionStatus`.

| Field | Type | Description |
|---|---|---|
| `session_id` | string | UUID of the session. |
| `session_name` | string | Human-readable label provided at creation, or empty. |
| `source_uri` | string | The RTSP URL or file path being processed. |
| `state` | string | Lifecycle state (see table below). |
| `frames_processed` | uint64 | Number of frames successfully processed. |
| `frames_dropped` | uint64 | Number of frames dropped (e.g. due to `max_fps` throttling). |
| `uptime_seconds` | double | Wall-clock seconds since the session was created. |

### Session States

| State | Meaning |
|---|---|
| `starting` | Session is initializing (opening source, allocating GPU buffers). |
| `running` | Session is actively decoding and processing frames. |
| `finished` | Video file reached EOF. Session remains queryable until stopped. |
| `error` | An unrecoverable error occurred (decode failure, Triton disconnect, etc.). |
| `stopped` | Session was explicitly stopped via `StopSession`. |
| `unknown` | Returned by `GetSessionStatus` when the session ID is not found. |

### ServiceConfig (Server-side Defaults)

These defaults apply when `SessionConfig` fields are `0` or omitted.

| Parameter | Default | Description |
|---|---|---|
| Triton URL | `localhost:8001` | Triton Inference Server gRPC endpoint. |
| Face DB | `./faces.db` | SQLite database with enrolled face embeddings. |
| Device mode | `auto` | `gpu` uses CUDA/NVDEC, `cpu` uses FFmpeg/OpenCV, `auto` detects. |
| YOLO model | `yolo26_face` | Triton model name for face detection. |
| ArcFace model | `arcface` | Triton model name for face recognition. |
| Confidence threshold | `0.5` | Minimum YOLO detection score. |
| Match threshold | `0.6` | Minimum cosine similarity for identity match. |
| NMS IoU threshold | `0.45` | Intersection-over-union threshold for non-maximum suppression. |
| YOLO input size | `640` | Input resolution (640x640) for the YOLO model. |
| Max faces per frame | `32` | Upper bound on detections processed per frame. |
| Max sessions | `30` | Maximum concurrent pipeline sessions. |
| gRPC port | `50051` | Server listen port. |

---

## Error Handling

All four RPCs return `grpc::Status::OK` at the transport level. Application-level errors are encoded in the response messages:

| Endpoint | Error Indicator | Description |
|---|---|---|
| `StartSession` | `status == "error"`, `error` field populated | Source unreachable, max sessions exceeded, initialization failure. |
| `StopSession` | `success == false` | Session ID not found. |
| `GetSessionStatus` | `state == "unknown"` | Session ID not found. |
| `ListSessions` | N/A | Always succeeds. Returns empty list if no sessions exist. |

Network-level gRPC errors (UNAVAILABLE, DEADLINE_EXCEEDED, etc.) follow standard gRPC semantics and indicate transport failures, not application logic issues.

---

## Client Examples

### grpcurl

Requires [grpcurl](https://github.com/fullstorydev/grpcurl) and server reflection or the proto file.

**Start a session:**

```bash
grpcurl -plaintext -d '{
  "source_uri": "rtsp://192.168.1.100:554/stream",
  "callback_url": "http://myapp:8080/webhook",
  "session_name": "lobby-camera",
  "config": {
    "confidence_threshold": 0.6,
    "match_threshold": 0.7,
    "max_fps": 15
  }
}' localhost:50051 videoanalytics.v1.VideoAnalytics/StartSession
```

```json
{
  "sessionId": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d",
  "status": "running"
}
```

**List all sessions:**

```bash
grpcurl -plaintext localhost:50051 videoanalytics.v1.VideoAnalytics/ListSessions
```

```json
{
  "sessions": [
    {
      "sessionId": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d",
      "sessionName": "lobby-camera",
      "sourceUri": "rtsp://192.168.1.100:554/stream",
      "state": "running",
      "framesProcessed": "1542",
      "framesDropped": "0",
      "uptimeSeconds": 102.4
    }
  ]
}
```

**Get session status:**

```bash
grpcurl -plaintext -d '{
  "session_id": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"
}' localhost:50051 videoanalytics.v1.VideoAnalytics/GetSessionStatus
```

**Stop a session:**

```bash
grpcurl -plaintext -d '{
  "session_id": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"
}' localhost:50051 videoanalytics.v1.VideoAnalytics/StopSession
```

```json
{
  "success": true,
  "framesProcessed": "3087"
}
```

### Python

Using the `grpcio` and `grpcio-tools` packages with generated stubs.

```bash
pip install grpcio grpcio-tools
python -m grpc_tools.protoc -Iproto --python_out=. --grpc_python_out=. proto/analytics.proto
```

```python
import grpc
import analytics_pb2
import analytics_pb2_grpc

channel = grpc.insecure_channel("localhost:50051")
stub = analytics_pb2_grpc.VideoAnalyticsStub(channel)

# Start a session
response = stub.StartSession(analytics_pb2.StartSessionRequest(
    source_uri="rtsp://192.168.1.100:554/stream",
    callback_url="http://myapp:8080/webhook",
    session_name="lobby-camera",
    config=analytics_pb2.SessionConfig(
        confidence_threshold=0.6,
        match_threshold=0.7,
        max_fps=15,
    ),
))
print(f"Session {response.session_id}: {response.status}")

# List sessions
sessions = stub.ListSessions(analytics_pb2.ListSessionsRequest())
for s in sessions.sessions:
    print(f"  {s.session_id} [{s.state}] {s.frames_processed} frames")

# Get status
status = stub.GetSessionStatus(
    analytics_pb2.GetSessionStatusRequest(session_id=response.session_id)
)
print(f"State: {status.state}, Uptime: {status.uptime_seconds:.1f}s")

# Stop
result = stub.StopSession(
    analytics_pb2.StopSessionRequest(session_id=response.session_id)
)
print(f"Stopped: {result.success}, Frames: {result.frames_processed}")
```

### Receiving Webhooks

A minimal Flask server to receive detection results:

```python
from flask import Flask, request

app = Flask(__name__)

@app.route("/webhook", methods=["POST"])
def receive_result():
    data = request.json
    session = data["session_id"]
    frame = data["frame_id"]

    for det in data["detections"]:
        bbox = det["bbox"]
        match = det.get("match")
        if match:
            print(f"[{session[:8]}] Frame {frame}: "
                  f"{match['name']} (sim={match['similarity']:.2f}) "
                  f"at ({bbox['x1']:.0f},{bbox['y1']:.0f})-({bbox['x2']:.0f},{bbox['y2']:.0f})")
        else:
            print(f"[{session[:8]}] Frame {frame}: "
                  f"unknown face "
                  f"at ({bbox['x1']:.0f},{bbox['y1']:.0f})-({bbox['x2']:.0f},{bbox['y2']:.0f})")

    return "", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
```
