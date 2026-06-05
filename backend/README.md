# Backend (FastAPI)

Local API for dance video upload, validation, pose extraction, and JSON output.

See the [project README](../README.md) for full MVP setup, mobile `BACKEND_URL`, recording tips, and limitations.

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | `{"status":"ok"}` |
| GET | `/references` | List saved reference extractions (from `outputs/` JSON files) |
| POST | `/extract-pose` | Upload video (`file` field); returns metadata, quality, summary, and reference media URLs on success |
| POST | `/extract-live-pose-frame` | Upload snapshot (`file`, JPEG/PNG); returns single-frame pose (not saved to disk) |
| GET | `/outputs/{filename}` | Download a `.json` file from `outputs/` (path traversal blocked) |
| GET | `/reference-media/{filename}` | Stream a saved reference video from `reference_media/` (path traversal blocked) |

Interactive docs: http://127.0.0.1:8000/docs

## Configuration (`app/config.py`)

| Setting | Default |
|---------|---------|
| `MAX_VIDEO_SECONDS` | 180 |
| `MAX_FILE_SIZE_MB` | 300 |
| `FRAME_STRIDE` | 1 |
| `MIN_POSE_DETECTION_PERCENTAGE` | 60 |
| `MIN_AVERAGE_VISIBILITY` | 0.45 |

## Safety behavior

- Uploaded videos in `uploads/` are **always deleted** after processing or failure.
- On successful extraction, the original video is copied to `reference_media/` (kept for Practice Mode playback); pose JSON stays in `outputs/`.
- Output JSON is written **atomically** (temp file + rename).
- Reference videos are copied **atomically** (temp file + rename).
- `GET /outputs/{filename}` only serves `.json` files under `outputs/` with path traversal checks.
- `GET /reference-media/{filename}` only serves allowed video types under `reference_media/` with path traversal checks.
- `POST /extract-live-pose-frame` processes images in memory only (no live frame files on disk).
- Invalid or unreadable videos return clear HTTP errors without crashing the server.
- OpenCV captures and MediaPipe backends are closed in `finally` blocks; errors are logged.

## CLI test

```bash
python scripts/test_extract_pose.py path/to/video.mp4
```
