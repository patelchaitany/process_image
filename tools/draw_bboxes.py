#!/usr/bin/env python3
"""
Overlay bounding boxes from a detections CSV onto an MP4 video.

Reads the CSV produced by CsvResultWriter and draws each detection's
bounding box, identity label, and confidence on the corresponding
video frame, then writes the annotated video to an output file.

Usage:
    python tools/draw_bboxes.py --video input.mp4 --csv output/detections.csv
    python tools/draw_bboxes.py --video input.mp4 --csv detections.csv -o annotated.mp4
    python tools/draw_bboxes.py --video input.mp4 --csv detections.csv --show
"""

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


# ── Colour palette for distinct identities ──────────────────────────
PALETTE = [
    (0, 255, 0),    # green
    (255, 128, 0),  # orange
    (0, 128, 255),  # blue
    (255, 0, 255),  # magenta
    (0, 255, 255),  # cyan
    (255, 255, 0),  # yellow
    (128, 0, 255),  # purple
    (0, 255, 128),  # spring green
    (255, 0, 128),  # rose
    (128, 255, 0),  # lime
]
UNKNOWN_COLOR = (0, 0, 255)  # red for unmatched faces


def parse_csv(csv_path: str) -> dict[int, list[dict]]:
    """
    Parse the detections CSV into a dict keyed by frame_id.
    Each value is a list of detection dicts for that frame.
    """
    frames = defaultdict(list)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = int(row["frame_id"])
            det = {
                "det_idx": int(row["det_idx"]),
                "x1": float(row["x1"]),
                "y1": float(row["y1"]),
                "x2": float(row["x2"]),
                "y2": float(row["y2"]),
                "det_confidence": float(row["det_confidence"]),
                "identity": row["identity"],
                "face_id": int(row["face_id"]),
                "match_confidence": float(row["match_confidence"]),
            }
            if det["identity"] == "no_detection":
                continue
            frames[fid].append(det)

    return dict(frames)


def color_for_identity(identity: str, face_id: int) -> tuple[int, int, int]:
    if identity == "unknown" or face_id < 0:
        return UNKNOWN_COLOR
    return PALETTE[face_id % len(PALETTE)]


def draw_detections(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw bounding boxes and labels on a single frame."""
    for det in detections:
        x1, y1 = int(det["x1"]), int(det["y1"])
        x2, y2 = int(det["x2"]), int(det["y2"])
        color = color_for_identity(det["identity"], det["face_id"])

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = det["identity"]
        if det["face_id"] >= 0:
            label += f" ({det['match_confidence']:.2f})"
        else:
            label += f" ({det['det_confidence']:.2f})"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        label_y = max(y1 - 6, th + 4)
        cv2.rectangle(
            frame,
            (x1, label_y - th - 4),
            (x1 + tw + 4, label_y + baseline),
            color,
            cv2.FILLED,
        )

        brightness = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
        text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
        cv2.putText(frame, label, (x1 + 2, label_y), font, font_scale, text_color, thickness)

    return frame


def reencode_with_ffmpeg(raw_path: str, final_path: str, fps: float) -> bool:
    """Re-encode a raw AVI to H.264 MP4 using ffmpeg for universal playback."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False

    cmd = [
        ffmpeg, "-y",
        "-i", raw_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "20",
        "-r", str(fps),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        final_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def process_video(
    video_path: str,
    csv_path: str,
    output_path: str,
    show: bool = False,
):
    frames_data = parse_csv(csv_path)
    if not frames_data:
        print(f"Warning: no detections found in {csv_path}", file=sys.stderr)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video {video_path}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Write raw frames to a temp AVI (lossless, fast), then re-encode to H.264
    tmp_file = tempfile.NamedTemporaryFile(suffix=".avi", delete=False)
    tmp_path = tmp_file.name
    tmp_file.close()

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"Error: cannot create temp video writer", file=sys.stderr)
        sys.exit(1)

    frame_idx = 0
    annotated_count = 0

    print(f"Input:  {video_path} ({width}x{height} @ {fps:.1f} fps, {total_frames} frames)")
    print(f"CSV:    {csv_path} ({len(frames_data)} frames with detections)")
    print(f"Output: {output_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = frames_data.get(frame_idx, [])
        if detections:
            frame = draw_detections(frame, detections)
            annotated_count += 1

        cv2.putText(
            frame,
            f"Frame {frame_idx}",
            (10, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1,
        )

        writer.write(frame)

        if show:
            cv2.imshow("Detections", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"  processed {frame_idx}/{total_frames} frames...")

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()

    print(f"Encoding {frame_idx} frames to H.264...")
    if reencode_with_ffmpeg(tmp_path, output_path, fps):
        Path(tmp_path).unlink(missing_ok=True)
        print(f"Done. {frame_idx} frames written, {annotated_count} had detections.")
    else:
        # ffmpeg not available -- fall back to moving the raw MJPG AVI
        fallback = Path(output_path).with_suffix(".avi")
        shutil.move(tmp_path, str(fallback))
        print(
            f"Warning: ffmpeg not found, saved as MJPEG AVI: {fallback}\n"
            f"  Install ffmpeg for H.264 MP4 output: brew install ffmpeg",
            file=sys.stderr,
        )
        print(f"Done. {frame_idx} frames written, {annotated_count} had detections.")


def main():
    parser = argparse.ArgumentParser(
        description="Draw bounding boxes from a detections CSV onto an MP4 video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/draw_bboxes.py --video cam1.mp4 --csv output/detections.csv
  python tools/draw_bboxes.py --video cam1.mp4 --csv det.csv -o annotated.mp4 --show
        """,
    )
    parser.add_argument("--video", required=True, help="Input MP4 video file")
    parser.add_argument("--csv", required=True, help="Detections CSV from the pipeline")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output video path (default: <video>_annotated.mp4)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show a live preview window while processing",
    )

    args = parser.parse_args()

    if args.output is None:
        p = Path(args.video)
        args.output = str(p.parent / f"{p.stem}_annotated.mp4")

    process_video(
        video_path=args.video,
        csv_path=args.csv,
        output_path=args.output,
        show=args.show,
    )


if __name__ == "__main__":
    main()
