"""
Face Analytics Client — Streamlit app for face registration & video analytics.

Run:  streamlit run client/app.py
"""

import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import grpc
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Proto stub generation (runs once, generates analytics_pb2*.py next to app)
# ---------------------------------------------------------------------------
_STUB_DIR = os.path.dirname(os.path.abspath(__file__))
_PROTO_DIR = os.path.join(_STUB_DIR, "..", "proto")


def _ensure_stubs():
    if os.path.exists(os.path.join(_STUB_DIR, "analytics_pb2.py")):
        return True
    proto_file = os.path.join(_PROTO_DIR, "analytics.proto")
    if not os.path.exists(proto_file):
        st.error(
            f"Proto file not found at `{proto_file}`. "
            "Make sure you run from the project root."
        )
        return False
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"-I{_PROTO_DIR}",
            f"--python_out={_STUB_DIR}",
            f"--grpc_python_out={_STUB_DIR}",
            proto_file,
        ]
    )
    return True


if not _ensure_stubs():
    st.stop()

if _STUB_DIR not in sys.path:
    sys.path.insert(0, _STUB_DIR)

import analytics_pb2  # noqa: E402
import analytics_pb2_grpc  # noqa: E402

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Face Analytics", layout="wide")

# ---------------------------------------------------------------------------
# Webhook result collector (runs as a background HTTP server)
# ---------------------------------------------------------------------------


class WebhookCollector:
    """Tiny HTTP server that collects webhook POSTs from the analytics service."""

    def __init__(self, port: int):
        self.port = port
        self.results: list[dict] = []
        self.lock = threading.Lock()
        self._server: HTTPServer | None = None

    def start(self):
        parent = self

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                with parent.lock:
                    parent.results.append(body)
                self.send_response(200)
                self.end_headers()

            def log_message(self, *_args):
                pass

        self._server = HTTPServer(("0.0.0.0", self.port), _Handler)
        thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        thread.start()

    def get_results(self, session_id: str | None = None) -> list[dict]:
        with self.lock:
            if session_id:
                return [r for r in self.results if r.get("session_id") == session_id]
            return list(self.results)

    def clear(self):
        with self.lock:
            self.results.clear()


@st.cache_resource
def _start_webhook_collector(port: int) -> WebhookCollector:
    collector = WebhookCollector(port)
    collector.start()
    return collector


# ---------------------------------------------------------------------------
# gRPC helpers
# ---------------------------------------------------------------------------


def _parse_target(url: str) -> tuple[str, bool]:
    """Return (grpc_target, use_tls) from a user-supplied URL."""
    url = url.strip().rstrip("/")
    if url.startswith("https://"):
        host = url.removeprefix("https://")
        if ":" not in host:
            host += ":443"
        return host, True
    host = url.removeprefix("http://")
    return host, False


def _get_channel(url: str):
    target, tls = _parse_target(url)
    if tls:
        return grpc.secure_channel(target, grpc.ssl_channel_credentials())
    return grpc.insecure_channel(target)


def _get_stub(url: str):
    return analytics_pb2_grpc.VideoAnalyticsStub(_get_channel(url))


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def _draw_detections_pil(image, detections):
    """Draw bounding boxes on a PIL Image. Returns a new image."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for det in detections:
        b = det.bbox
        draw.rectangle([b.x1, b.y1, b.x2, b.y2], outline="lime", width=3)
        draw.text((b.x1 + 4, b.y1 - 16), f"{det.confidence:.0%}", fill="lime")
    return img


# ---------------------------------------------------------------------------
# Sidebar — server configuration
# ---------------------------------------------------------------------------
st.sidebar.title("Server")

server_url = st.sidebar.text_input(
    "gRPC endpoint",
    value=st.session_state.get("server_url", "localhost:50051"),
    placeholder="https://50051-xxx.cloudspaces.litng.ai",
)
st.session_state.server_url = server_url

if st.sidebar.button("Test connection"):
    try:
        stub = _get_stub(server_url)
        resp = stub.ListSessions(analytics_pb2.ListSessionsRequest(), timeout=10)
        n = len(resp.sessions)
        st.sidebar.success(f"Connected — {n} active session{'s' if n != 1 else ''}.")
    except grpc.RpcError as e:
        st.sidebar.error(f"gRPC error: {e.code().name} — {e.details()}")
    except Exception as e:
        st.sidebar.error(f"Connection failed: {e}")

st.sidebar.divider()
st.sidebar.caption(
    "Provide the gRPC URL of your Video Analytics Service. "
    "Use `https://` for TLS (e.g. Lightning AI cloudspaces)."
)

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------
st.title("Face Analytics")
tab_register, tab_video = st.tabs(["Register Face", "Video Analytics"])

# ===== Tab 1: Register Face ================================================
with tab_register:
    st.subheader("Register a New Face")
    st.caption("Take a photo → verify detection → enter name → register.")

    col_cam, col_result = st.columns(2)

    with col_cam:
        photo = st.camera_input("Capture your face", key="register_cam")

    if photo is not None:
        image_bytes = photo.getvalue()
        pil_image = Image.open(io.BytesIO(image_bytes))

        with col_result:
            with st.spinner("Running face detection…"):
                try:
                    stub = _get_stub(server_url)
                    detect_resp = stub.DetectFaces(
                        analytics_pb2.DetectFacesRequest(image_data=image_bytes),
                        timeout=15,
                    )
                except Exception as e:
                    st.error(f"Detection failed: {e}")
                    detect_resp = None

            if detect_resp is None:
                pass
            elif not detect_resp.success:
                st.error(f"Server error: {detect_resp.error}")
            else:
                n_faces = len(detect_resp.detections)
                annotated = _draw_detections_pil(pil_image, detect_resp.detections)
                st.image(
                    annotated,
                    caption=f"{n_faces} face{'s' if n_faces != 1 else ''} detected "
                    f"({detect_resp.image_width}x{detect_resp.image_height})",
                    use_column_width=True,
                )

                if n_faces == 0:
                    st.warning("No faces detected. Try better lighting or move closer.")
                elif n_faces > 1:
                    st.warning(
                        f"{n_faces} faces found — registration requires exactly 1. "
                        "Crop the image or move others out of frame."
                    )
                else:
                    st.success("Face detected — ready to register.")
                    name = st.text_input("Your name", key="reg_name")
                    if st.button("Register", type="primary", disabled=not name):
                        with st.spinner("Registering…"):
                            try:
                                reg_resp = stub.RegisterFace(
                                    analytics_pb2.RegisterFaceRequest(
                                        name=name,
                                        image_data=image_bytes,
                                    ),
                                    timeout=20,
                                )
                                if reg_resp.success:
                                    st.success(
                                        f"Registered **{name}** "
                                        f"(person ID: {reg_resp.person_id})"
                                    )
                                else:
                                    st.error(f"Registration failed: {reg_resp.error}")
                            except Exception as e:
                                st.error(f"Error: {e}")


# ===== Tab 2: Video Analytics ==============================================
with tab_video:
    st.subheader("Video Analytics Pipeline")
    st.caption(
        "Record or upload a video → it's saved on the server → "
        "StartSession processes it with the full YOLO + ArcFace pipeline → "
        "results are collected via webhook and displayed here."
    )

    # ── Step 1: Video source ───────────────────────────────────────────────
    st.markdown("#### 1. Video Source")
    source_mode = st.radio(
        "How to provide the video",
        ["Upload a file", "Record from webcam", "File already on server"],
        horizontal=True,
        key="va_source_mode",
    )

    video_server_path: str | None = None
    _UPLOAD_DIR = "/tmp/va_uploads"

    if source_mode == "Upload a file":
        uploaded = st.file_uploader(
            "Drop a video file",
            type=["mp4", "avi", "mkv", "mov", "webm"],
            key="va_upload",
        )
        if uploaded is not None:
            os.makedirs(_UPLOAD_DIR, exist_ok=True)
            save_path = os.path.join(_UPLOAD_DIR, uploaded.name)
            with open(save_path, "wb") as f:
                f.write(uploaded.getvalue())
            video_server_path = save_path
            st.video(uploaded)
            st.info(f"Saved to **{save_path}** on the server filesystem.")

    elif source_mode == "Record from webcam":
        rec_duration = st.slider(
            "Recording duration (seconds)", 3, 30, 10, key="va_rec_dur"
        )
        if st.button("Start recording", key="va_rec_btn"):
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Could not open webcam.")
            else:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

                os.makedirs(_UPLOAD_DIR, exist_ok=True)
                out_path = os.path.join(
                    _UPLOAD_DIR,
                    f"recording_{int(time.time())}.mp4",
                )
                fourcc = cv2.VideoWriter.fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

                preview = st.empty()
                progress = st.progress(0, text="Recording…")
                start_t = time.time()

                while time.time() - start_t < rec_duration:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    writer.write(frame)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    preview.image(rgb, channels="RGB", use_column_width=True)
                    elapsed = time.time() - start_t
                    progress.progress(
                        min(elapsed / rec_duration, 1.0),
                        text=f"Recording… {elapsed:.1f}s / {rec_duration}s",
                    )

                writer.release()
                cap.release()
                preview.empty()
                progress.empty()

                st.session_state.va_recorded_path = out_path
                st.success(f"Recorded to **{out_path}**")

        if "va_recorded_path" in st.session_state:
            video_server_path = st.session_state.va_recorded_path
            if os.path.exists(video_server_path):
                st.video(video_server_path)

    else:  # File already on server
        video_server_path = st.text_input(
            "Absolute file path on the server",
            value="/workspace/video.mp4",
            key="va_server_path",
        )

    # ── Step 2: Session configuration ──────────────────────────────────────
    st.markdown("#### 2. Session Configuration")

    col_cb, col_name = st.columns(2)
    with col_cb:
        webhook_port = st.number_input(
            "Webhook receiver port (on this machine)",
            value=8080,
            min_value=1024,
            max_value=65535,
            key="va_webhook_port",
        )
        callback_url = st.text_input(
            "Callback URL",
            value=f"http://localhost:{webhook_port}/webhook",
            key="va_callback_url",
            help="The analytics service POSTs detection results here. "
            "If this app runs on the same machine as the service, "
            "localhost works.",
        )
    with col_name:
        session_name = st.text_input(
            "Session name", value="streamlit-session", key="va_session_name"
        )

    with st.expander("Advanced — detection thresholds"):
        adv_col1, adv_col2, adv_col3 = st.columns(3)
        with adv_col1:
            conf_threshold = st.slider(
                "Confidence threshold",
                0.0, 1.0, 0.0,
                step=0.05,
                key="va_conf",
                help="0 = use server default (0.5)",
            )
        with adv_col2:
            match_threshold = st.slider(
                "Match threshold",
                0.0, 1.0, 0.0,
                step=0.05,
                key="va_match",
                help="0 = use server default (0.6)",
            )
        with adv_col3:
            max_fps = st.number_input(
                "Max FPS", value=0, min_value=0, max_value=60,
                key="va_max_fps",
                help="0 = no limit",
            )

    # ── Step 3: Start / Stop ───────────────────────────────────────────────
    st.markdown("#### 3. Run")

    col_start, col_stop, col_refresh = st.columns(3)

    with col_start:
        start_disabled = not video_server_path
        if st.button(
            "Start Session", type="primary", disabled=start_disabled, key="va_start"
        ):
            collector = _start_webhook_collector(int(webhook_port))
            collector.clear()
            try:
                stub = _get_stub(server_url)
                resp = stub.StartSession(
                    analytics_pb2.StartSessionRequest(
                        source_uri=video_server_path,
                        callback_url=callback_url,
                        session_name=session_name,
                        config=analytics_pb2.SessionConfig(
                            confidence_threshold=conf_threshold,
                            match_threshold=match_threshold,
                            max_fps=int(max_fps),
                        ),
                    ),
                    timeout=15,
                )
                if resp.status == "running":
                    st.session_state.va_session_id = resp.session_id
                    st.success(f"Session started: `{resp.session_id}`")
                else:
                    st.error(f"Failed to start: {resp.error}")
            except Exception as e:
                st.error(f"StartSession error: {e}")

    with col_stop:
        if st.button("Stop Session", key="va_stop"):
            sid = st.session_state.get("va_session_id")
            if not sid:
                st.warning("No active session.")
            else:
                try:
                    stub = _get_stub(server_url)
                    resp = stub.StopSession(
                        analytics_pb2.StopSessionRequest(session_id=sid),
                        timeout=10,
                    )
                    if resp.success:
                        st.info(
                            f"Stopped. Processed **{resp.frames_processed}** frames."
                        )
                    else:
                        st.warning("Session not found (may have already finished).")
                except Exception as e:
                    st.error(f"StopSession error: {e}")

    with col_refresh:
        st.button("Refresh", key="va_refresh")

    # ── Step 4: Status & Results ───────────────────────────────────────────
    sid = st.session_state.get("va_session_id")
    if sid:
        st.markdown("#### 4. Session Status")
        try:
            stub = _get_stub(server_url)
            status = stub.GetSessionStatus(
                analytics_pb2.GetSessionStatusRequest(session_id=sid),
                timeout=10,
            )

            state_colors = {
                "running": "🟢",
                "finished": "🔵",
                "stopped": "⚪",
                "error": "🔴",
                "starting": "🟡",
            }
            state_icon = state_colors.get(status.state, "❓")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("State", f"{state_icon} {status.state}")
            m2.metric("Frames Processed", f"{status.frames_processed:,}")
            m3.metric("Frames Dropped", f"{status.frames_dropped:,}")
            m4.metric("Uptime", f"{status.uptime_seconds:.1f}s")
        except Exception as e:
            st.warning(f"Could not fetch status: {e}")

        # -- Webhook results --
        st.markdown("#### 5. Detection Results")
        try:
            collector = _start_webhook_collector(int(webhook_port))
            results = collector.get_results(session_id=sid)
        except Exception:
            results = []

        if not results:
            st.info(
                "No results yet. Make sure the callback URL is reachable "
                "from the analytics service. Click **Refresh** to check again."
            )
        else:
            st.write(f"**{len(results)}** frames received via webhook.")

            for payload in results[-20:]:
                frame_id = payload.get("frame_id", "?")
                ts = payload.get("timestamp_utc", "")
                detections = payload.get("detections", [])

                if not detections:
                    continue

                with st.expander(
                    f"Frame {frame_id} — {len(detections)} face(s) — {ts}",
                    expanded=(payload == results[-1]),
                ):
                    for det in detections:
                        bbox = det.get("bbox", {})
                        conf = det.get("confidence", 0)
                        match = det.get("match")

                        box_str = (
                            f"({bbox.get('x1', 0):.0f}, {bbox.get('y1', 0):.0f}) → "
                            f"({bbox.get('x2', 0):.0f}, {bbox.get('y2', 0):.0f})"
                        )

                        if match:
                            st.markdown(
                                f"**{match['name']}** — "
                                f"similarity {match['similarity']:.2f} · "
                                f"conf {conf:.0%} · {box_str}"
                            )
                        else:
                            st.markdown(
                                f"*Unknown face* — conf {conf:.0%} · {box_str}"
                            )
