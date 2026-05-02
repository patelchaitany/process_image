# Examples

## Quick Start

### Process a video file with face recognition

```bash
# 1. Start Triton (in another terminal)
docker run --rm -p 8001:8001 -v ./models/model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models

# 2. Enroll known faces
python3 tools/enroll_faces.py \
  --faces-dir ./faces/ \
  --db ./faces.db \
  --triton localhost:8001

# 3. Run pipeline
./build/process_image \
  --input meeting_recording.mp4 \
  --triton localhost:8001 \
  --db ./faces.db \
  --output-csv ./metrics/meeting.csv
```

---

## Face Enrollment

### Basic enrollment from a photo directory

```
faces/
├── alice/
│   ├── front.jpg
│   ├── side.jpg
│   └── smile.png
├── bob/
│   ├── headshot.jpg
│   └── profile.png
└── charlie/
    └── photo.bmp
```

```bash
python3 tools/enroll_faces.py \
  --faces-dir ./faces/ \
  --db ./faces.db \
  --triton localhost:8001
```

**Output:**
```
Connecting to Triton at localhost:8001... OK
Processing alice/ (3 images)...
  front.jpg: 1 face detected, embedding extracted
  side.jpg: 1 face detected, embedding extracted
  smile.png: 1 face detected, embedding extracted
  → alice enrolled (averaged 3 embeddings)
Processing bob/ (2 images)...
  headshot.jpg: 1 face detected, embedding extracted
  profile.png: 1 face detected, embedding extracted
  → bob enrolled (averaged 2 embeddings)
Processing charlie/ (1 image)...
  photo.bmp: 2 faces detected, skipping (ambiguous)
  → charlie: no valid embeddings, NOT enrolled

Summary: 2/3 persons enrolled, 5/6 images used, 1 skipped
Database written to ./faces.db
```

### Re-enroll (replace all existing faces)

```bash
python3 tools/enroll_faces.py \
  --faces-dir ./updated_faces/ \
  --db ./faces.db \
  --triton localhost:8001 \
  --replace
```

### Custom confidence threshold

```bash
# More strict face detection (fewer false positives)
python3 tools/enroll_faces.py \
  --faces-dir ./faces/ \
  --db ./faces.db \
  --triton localhost:8001 \
  --confidence 0.7
```

---

## Pipeline Usage

### Process a local MP4 file

```bash
./build/process_image \
  --input /path/to/video.mp4 \
  --triton localhost:8001
```

**Output:**
```
process_image v0.1
  Input:       /path/to/video.mp4
  Triton:      localhost:8001
  Device:      auto
  DB:          ./faces.db
  Metrics CSV: ./metrics/metrics.csv
  YOLO model:  yolo26_face
  ArcFace:     arcface
  Confidence:  0.50
  Match thr:   0.60

Device mode: GPU
Input opened: 1920x1080 @ 30.0 fps
Face database: 5 faces loaded
Pipeline initialized successfully
Frame 0: 2 faces detected
  [0] bbox=(412,156,589,401) det_conf=0.934 match=alice (0.847)
  [1] bbox=(891,203,1042,498) det_conf=0.891 match=bob (0.762)
Frame 1: 2 faces detected
  [0] bbox=(415,158,591,403) det_conf=0.928 match=alice (0.839)
  [1] bbox=(893,205,1044,500) det_conf=0.887 match=bob (0.755)
...
Pipeline finished. Processed: 900, Dropped: 0
Done.
```

### Process an RTSP stream (live camera)

```bash
./build/process_image \
  --input rtsp://192.168.1.100:554/stream1 \
  --triton localhost:8001 \
  --db ./faces.db \
  --output-csv ./metrics/camera1.csv
```

Press `Ctrl+C` to stop. Metrics CSV is flushed on exit.

### Force CPU mode (on Mac or for testing)

```bash
./build/process_image \
  --input video.mp4 \
  --triton localhost:8001 \
  --device cpu \
  --db ./faces.db
```

### Adjust detection thresholds

```bash
# More sensitive detection (catches more faces, may have false positives)
./build/process_image \
  --input video.mp4 \
  --triton localhost:8001 \
  --confidence 0.3 \
  --match-threshold 0.5

# Strict matching (high confidence required for face identification)
./build/process_image \
  --input video.mp4 \
  --triton localhost:8001 \
  --confidence 0.6 \
  --match-threshold 0.8
```

### Use custom model names

```bash
./build/process_image \
  --input video.mp4 \
  --triton localhost:8001 \
  --yolo-model yolov8_face_custom \
  --arcface-model arcface_r100
```

---

## Metrics CSV Analysis

After running the pipeline, analyze performance:

```bash
# View first few rows
head -5 ./metrics/metrics_2026-05-02_180000.csv

# Get average frame time
awk -F',' 'NR>1 {sum+=$NF; n++} END {printf "Avg: %.2f ms/frame\n", sum/n}' \
  ./metrics/metrics_2026-05-02_180000.csv

# Find slowest frames
sort -t',' -k21 -rn ./metrics/metrics_2026-05-02_180000.csv | head -5
```

### Python analysis

```python
import pandas as pd

df = pd.read_csv("./metrics/metrics_2026-05-02_180000.csv")

print(f"Total frames: {len(df)}")
print(f"Avg pipeline time: {df['total_pipeline_ms'].mean():.2f} ms")
print(f"P95 pipeline time: {df['total_pipeline_ms'].quantile(0.95):.2f} ms")
print(f"P99 pipeline time: {df['total_pipeline_ms'].quantile(0.99):.2f} ms")
print(f"Avg faces/frame: {df['faces_detected'].mean():.1f}")

# Breakdown
print(f"\nTime breakdown:")
print(f"  Decode:     {df['decode_ms'].mean():.2f} ms")
print(f"  Preprocess: {df['preprocess_total_ms'].mean():.2f} ms")
print(f"  YOLO:       {df['yolo_inference_ms'].mean():.2f} ms")
print(f"  NMS:        {df['nms_total_ms'].mean():.2f} ms")
print(f"  ArcFace:    {df['arcface_inference_ms'].mean():.2f} ms")
print(f"  FAISS:      {df['faiss_search_ms'].mean():.2f} ms")
```

---

## Common Workflows

### Add a new person to the database without re-enrolling everyone

```bash
# Create a folder for the new person
mkdir -p faces/dave
cp ~/photos/dave/*.jpg faces/dave/

# Run enrollment (existing faces are preserved, new person is added)
python3 tools/enroll_faces.py \
  --faces-dir ./faces/ \
  --db ./faces.db \
  --triton localhost:8001
```

Note: This re-processes all folders but only inserts new entries. If you want to add just one person, create a temporary directory:

```bash
mkdir -p /tmp/new_face/dave
cp ~/photos/dave/*.jpg /tmp/new_face/dave/

python3 tools/enroll_faces.py \
  --faces-dir /tmp/new_face/ \
  --db ./faces.db \
  --triton localhost:8001
```

### Batch process multiple videos

```bash
for video in /data/videos/*.mp4; do
  name=$(basename "$video" .mp4)
  ./build/process_image \
    --input "$video" \
    --triton localhost:8001 \
    --db ./faces.db \
    --output-csv "./metrics/${name}.csv"
done
```

### Inspect the face database

```bash
sqlite3 faces.db "SELECT id, name, created_at FROM faces;"
```

```
1|alice|2026-05-02 12:00:00
2|bob|2026-05-02 12:00:01
3|dave|2026-05-02 14:30:00
```

### Check database size

```bash
sqlite3 faces.db "SELECT COUNT(*) FROM faces;"
# 3
```

---

## Performance Expectations

| Setup | Latency | FPS | Notes |
|-------|---------|-----|-------|
| T4 GPU, 1080p, 5 faces | ~13 ms | ~77 | Target spec |
| T4 GPU, 1080p, 0 faces | ~8 ms | ~125 | No ArcFace/FAISS needed |
| CPU (Mac M2), 1080p, 5 faces | ~150 ms | ~6-7 | Functional, not real-time |
| CPU (Mac M2), 720p, 2 faces | ~80 ms | ~12 | Reduced resolution helps |
| CPU (Intel i7), 1080p, 5 faces | ~200 ms | ~5 | Varies by CPU |

To improve CPU mode performance:
- Use lower resolution input (720p instead of 1080p)
- Reduce confidence threshold to skip more non-face regions early
- Ensure Triton is using all available CPU cores
