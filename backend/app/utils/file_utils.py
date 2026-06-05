"""File path and upload helpers."""

from __future__ import annotations

import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

from app.config import (
    ALLOWED_VIDEO_EXTENSIONS,
    OUTPUT_DIR,
    REFERENCE_MEDIA_DIR,
    UPLOAD_DIR,
)

VIDEO_MEDIA_TYPES: dict[str, str] = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".m4v": "video/x-m4v",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
}

_SAFE_FILENAME_PATTERN = re.compile(r"[^a-zA-Z0-9._-]+")


def ensure_runtime_directories() -> None:
    """Create upload, output, and reference media directories if missing."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCE_MEDIA_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_filename(filename: str) -> str:
    """Return a filesystem-safe basename (no path components)."""
    name = Path(filename).name
    if not name or name in {".", ".."}:
        name = "upload"
    return _SAFE_FILENAME_PATTERN.sub("_", name)


def build_upload_path(original_filename: str) -> Path:
    """Generate a unique path under UPLOAD_DIR for an incoming video."""
    safe_name = sanitize_filename(original_filename)
    unique_name = f"{uuid.uuid4().hex}_{safe_name}"
    return UPLOAD_DIR / unique_name


def _resolve_file_under_root(
    filename: str,
    *,
    root: Path,
    allowed_suffixes: set[str],
) -> Path:
    if not filename or filename != Path(filename).name:
        raise ValueError("Invalid filename")

    if "/" in filename or "\\" in filename:
        raise ValueError("Invalid filename")

    safe_name = sanitize_filename(filename)
    suffix = Path(safe_name).suffix.lower()
    if suffix not in allowed_suffixes:
        raise ValueError("Invalid filename")

    resolved = (root / safe_name).resolve()
    root_resolved = root.resolve()

    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError("Invalid filename") from exc

    return resolved


def resolve_output_file(filename: str) -> Path:
    """
    Resolve a filename under OUTPUT_DIR.

    Raises ValueError if the path would escape OUTPUT_DIR or contains separators.
    """
    return _resolve_file_under_root(
        filename,
        root=OUTPUT_DIR,
        allowed_suffixes={".json"},
    )


def resolve_reference_media_file(filename: str) -> Path:
    """
    Resolve a filename under REFERENCE_MEDIA_DIR.

    Raises ValueError if the path would escape the directory or the extension
    is not an allowed video type.
    """
    return _resolve_file_under_root(
        filename,
        root=REFERENCE_MEDIA_DIR,
        allowed_suffixes=ALLOWED_VIDEO_EXTENSIONS,
    )


def media_type_for_video_suffix(suffix: str) -> str:
    """Return Content-Type for a reference video extension."""
    normalized = suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}"
    return VIDEO_MEDIA_TYPES.get(normalized, "application/octet-stream")


def build_reference_video_filename(reference_id: str, original_suffix: str) -> str:
    """Build a safe stored reference video basename tied to reference_id."""
    suffix = original_suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video extension: {suffix}")
    safe_id = _SAFE_FILENAME_PATTERN.sub("_", reference_id)
    if not safe_id:
        raise ValueError("Invalid reference id")
    return f"{safe_id}{suffix}"


def save_reference_video_copy(source: Path, dest_filename: str) -> Path:
    """
    Copy a validated upload into REFERENCE_MEDIA_DIR with a safe unique name.

    Writes via a temp file in the same directory, then renames atomically.
    """
    dest_path = resolve_reference_media_file(dest_filename)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    try:
        shutil.copy2(source, temp_path)
        temp_path.replace(dest_path)
    except Exception:
        if temp_path.is_file():
            temp_path.unlink(missing_ok=True)
        raise
    return dest_path


def write_bytes(path: Path, data: bytes) -> None:
    """Write raw bytes to path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON via a temp file in the same directory, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        temp_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(path)
    except Exception:
        if temp_path.is_file():
            temp_path.unlink(missing_ok=True)
        raise
