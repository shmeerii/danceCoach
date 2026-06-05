"""Application configuration."""

from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent

# Video limits
MAX_VIDEO_SECONDS = 180
MAX_FILE_SIZE_MB = 300

# Extraction settings
FRAME_STRIDE = 1
MIN_POSE_DETECTION_PERCENTAGE = 60
MIN_AVERAGE_VISIBILITY = 0.45

# Runtime directories
OUTPUT_DIR = BACKEND_ROOT / "outputs"
UPLOAD_DIR = BACKEND_ROOT / "uploads"
REFERENCE_MEDIA_DIR = BACKEND_ROOT / "reference_media"

# Live practice frame limits (camera snapshots)
MAX_LIVE_FRAME_UPLOAD_MB = 8
MAX_LIVE_FRAME_UPLOAD_BYTES = MAX_LIVE_FRAME_UPLOAD_MB * 1024 * 1024
MAX_LIVE_FRAME_DIMENSION = 640

# Derived
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
ALLOWED_LIVE_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
