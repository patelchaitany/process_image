#!/usr/bin/env python3
"""Face enrollment tool - populates SQLite face database from photo folders.

Each subfolder name is the person's name, containing one or more photos.
Uses Triton Inference Server for YOLO face detection and ArcFace embedding extraction.
"""

import argparse
import os
import sys
import sqlite3
import struct

import cv2
import numpy as np
import tritonclient.grpc as grpcclient


def parse_args():
    parser = argparse.ArgumentParser(
        description="Enroll faces from a photo directory into the face database"
    )
    parser.add_argument("--faces-dir", required=True, help="Root folder with person subfolders")
    parser.add_argument("--db", required=True, help="SQLite database path (created if not exists)")
    parser.add_argument("--triton", required=True, help="Triton server URL (e.g. localhost:8001)")
    parser.add_argument("--yolo-model", default="yolo26_face", help="Triton model name for detection")
    parser.add_argument("--arcface-model", default="arcface", help="Triton model name for recognition")
    parser.add_argument("--confidence", type=float, default=0.5, help="Face detection confidence threshold")
    parser.add_argument("--replace", action="store_true", help="Drop existing faces table and re-enroll")
    parser.add_argument("--debug", action="store_true", help="Print raw YOLO output shape and stats")
    return parser.parse_args()


def connect_triton(url):
    """Connect to Triton and verify it's alive."""
    print(f"Connecting to Triton at {url}...", end=" ")
    try:
        client = grpcclient.InferenceServerClient(url=url)
        if not client.is_server_live():
            print("FAILED (server not live)")
            sys.exit(1)
        print("OK")
        return client
    except Exception as e:
        print(f"FAILED ({e})")
        sys.exit(1)


def preprocess_yolo(image, input_size=640):
    """Preprocess image for YOLO: resize with letterbox, normalize to [0,1], CHW layout."""
    h, w = image.shape[:2]
    scale = min(input_size / w, input_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    pad_x, pad_y = (input_size - new_w) // 2, (input_size - new_h) // 2

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    # BGR to RGB, HWC to CHW, normalize
    blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = np.expand_dims(blob, 0)  # [1, 3, 640, 640]

    return blob, scale, pad_x, pad_y


def preprocess_arcface(face_crop, input_size=112):
    """Preprocess face crop for ArcFace: resize to 112x112, normalize (x-127.5)/127.5, CHW."""
    resized = cv2.resize(face_crop, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    # BGR to RGB, normalize, HWC to CHW
    blob = resized[:, :, ::-1].astype(np.float32)
    blob = (blob - 127.5) / 127.5
    blob = blob.transpose(2, 0, 1)
    blob = np.expand_dims(blob, 0)  # [1, 3, 112, 112]
    return blob


def run_yolo(client, model_name, input_blob):
    """Run YOLO inference on Triton, return raw output."""
    inputs = [grpcclient.InferInput("images", input_blob.shape, "FP32")]
    inputs[0].set_data_from_numpy(input_blob)

    outputs = [grpcclient.InferRequestedOutput("output0")]

    result = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    return result.as_numpy("output0")


def run_arcface(client, model_name, input_blob):
    """Run ArcFace inference on Triton, return embedding."""
    inputs = [grpcclient.InferInput("input", input_blob.shape, "FP32")]
    inputs[0].set_data_from_numpy(input_blob)

    outputs = [grpcclient.InferRequestedOutput("output")]

    result = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    return result.as_numpy("output")


def normalize_yolo_output(output, debug=False):
    """Ensure YOLO output is in [N, 5] layout (cx, cy, w, h, conf).

    Handles both [batch, N, 5] and the transposed [batch, 5, N] format
    that many YOLO variants produce natively.
    """
    if output.ndim == 3:
        output = output[0]

    if debug:
        print(f"    [debug] raw output shape (after batch squeeze): {output.shape}")
        print(f"    [debug] value range: [{output.min():.4f}, {output.max():.4f}]")

    # [N, 5]: already in expected layout (N >> 5)
    # [5, N]: transposed — need to flip
    if output.ndim == 2 and output.shape[0] < output.shape[1]:
        if debug:
            print(f"    [debug] transposing from [{output.shape[0]}, {output.shape[1]}] "
                  f"to [{output.shape[1]}, {output.shape[0]}]")
        output = output.T

    return output


def decode_yolo_output(output, confidence_threshold, scale, pad_x, pad_y, orig_w, orig_h,
                       debug=False):
    """Decode YOLO output to bounding boxes in original image coordinates."""
    output = normalize_yolo_output(output, debug=debug)

    if debug:
        confs = output[:, 4] if output.shape[1] > 4 else output[:, -1]
        print(f"    [debug] {output.shape[0]} candidates, "
              f"max conf={confs.max():.4f}, "
              f">{confidence_threshold}: {(confs > confidence_threshold).sum()}")

    detections = []
    for det in output:
        conf = det[4]
        if conf < confidence_threshold:
            continue

        cx, cy, w, h = det[0], det[1], det[2], det[3]

        # Convert from model coords to original image coords
        x1 = (cx - w / 2 - pad_x) / scale
        y1 = (cy - h / 2 - pad_y) / scale
        x2 = (cx + w / 2 - pad_x) / scale
        y2 = (cy + h / 2 - pad_y) / scale

        # Clip
        x1 = max(0, min(x1, orig_w))
        y1 = max(0, min(y1, orig_h))
        x2 = max(0, min(x2, orig_w))
        y2 = max(0, min(y2, orig_h))

        if x2 > x1 and y2 > y1:
            detections.append((x1, y1, x2, y2, conf))

    return detections


def nms(detections, iou_threshold=0.45):
    """Simple greedy NMS."""
    if not detections:
        return []

    dets = sorted(detections, key=lambda d: d[4], reverse=True)
    keep = []

    while dets:
        best = dets.pop(0)
        keep.append(best)
        remaining = []
        for d in dets:
            iou = compute_iou(best, d)
            if iou <= iou_threshold:
                remaining.append(d)
        dets = remaining

    return keep


def compute_iou(a, b):
    """Compute IoU between two boxes (x1, y1, x2, y2, conf)."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0


def init_database(db_path, replace=False):
    """Initialize SQLite database with faces table."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if replace:
        cursor.execute("DROP TABLE IF EXISTS faces")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            embedding BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_faces_name ON faces(name)")
    conn.commit()
    return conn


def embedding_to_blob(embedding):
    """Convert float32 numpy array to bytes for SQLite BLOB storage."""
    return embedding.astype(np.float32).tobytes()


def main():
    args = parse_args()

    if not os.path.isdir(args.faces_dir):
        print(f"Error: {args.faces_dir} is not a directory")
        sys.exit(1)

    client = connect_triton(args.triton)
    conn = init_database(args.db, replace=args.replace)
    cursor = conn.cursor()

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

    # Gather person directories
    persons = sorted([
        d for d in os.listdir(args.faces_dir)
        if os.path.isdir(os.path.join(args.faces_dir, d))
    ])

    total_persons = len(persons)
    enrolled_count = 0
    total_images = 0
    used_images = 0
    skipped_images = 0

    for person_name in persons:
        person_dir = os.path.join(args.faces_dir, person_name)
        images = sorted([
            f for f in os.listdir(person_dir)
            if os.path.splitext(f)[1].lower() in image_extensions
        ])

        total_images += len(images)
        print(f"Processing {person_name}/ ({len(images)} images)...")

        person_embeddings = []

        for img_name in images:
            img_path = os.path.join(person_dir, img_name)
            image = cv2.imread(img_path)

            if image is None:
                print(f"  {img_name}: cannot read, skipping")
                skipped_images += 1
                continue

            h, w = image.shape[:2]

            # Run YOLO
            yolo_input, scale, pad_x, pad_y = preprocess_yolo(image)
            yolo_output = run_yolo(client, args.yolo_model, yolo_input)

            if args.debug:
                print(f"  [{img_name}] YOLO raw output shape: {yolo_output.shape}, "
                      f"dtype: {yolo_output.dtype}")

            # Decode and NMS
            detections = decode_yolo_output(
                yolo_output, args.confidence, scale, pad_x, pad_y, w, h,
                debug=args.debug,
            )
            detections = nms(detections)

            if len(detections) == 0:
                raw = normalize_yolo_output(yolo_output)
                max_conf = raw[:, 4].max() if raw.shape[1] > 4 else 0.0
                print(f"  {img_name}: 0 faces detected (max conf={max_conf:.4f}), skipping")
                skipped_images += 1
                continue
            elif len(detections) > 1:
                print(f"  {img_name}: {len(detections)} faces detected, skipping (ambiguous)")
                skipped_images += 1
                continue

            # Single face - extract embedding
            x1, y1, x2, y2, _ = detections[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            face_crop = image[y1:y2, x1:x2]

            if face_crop.size == 0:
                print(f"  {img_name}: invalid crop, skipping")
                skipped_images += 1
                continue

            arcface_input = preprocess_arcface(face_crop)
            embedding = run_arcface(client, args.arcface_model, arcface_input)

            if embedding.ndim > 1:
                embedding = embedding[0]

            # L2 normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            person_embeddings.append(embedding)
            used_images += 1
            print(f"  {img_name}: 1 face detected, embedding extracted")

        # Average embeddings for this person
        if person_embeddings:
            avg_embedding = np.mean(person_embeddings, axis=0)
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

            cursor.execute(
                "INSERT INTO faces (name, embedding) VALUES (?, ?)",
                (person_name, embedding_to_blob(avg_embedding))
            )
            conn.commit()
            enrolled_count += 1
            print(f"  \u2192 {person_name} enrolled (averaged {len(person_embeddings)} embeddings)")
        else:
            print(f"  \u2192 {person_name}: no valid embeddings, NOT enrolled")

    conn.close()

    print(f"\nSummary: {enrolled_count}/{total_persons} persons enrolled, "
          f"{used_images}/{total_images} images used, {skipped_images} skipped")
    print(f"Database written to {args.db}")


if __name__ == "__main__":
    main()
