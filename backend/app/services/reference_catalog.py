"""List saved reference extractions from outputs/ JSON files."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import OUTPUT_DIR
from app.utils.file_utils import resolve_reference_media_file

logger = logging.getLogger(__name__)

_POSES_JSON_SUFFIX = "_poses.json"


def reference_id_from_json_filename(filename: str) -> str | None:
    """Derive reference_id from `{id}_poses.json` basename."""
    name = Path(filename).name
    if not name.endswith(_POSES_JSON_SUFFIX):
        return None
    stem = name[: -len(_POSES_JSON_SUFFIX)]
    return stem if stem else None


def _iso_from_mtime(path: Path) -> str:
    mtime = path.stat().st_mtime
    return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()


def _parse_reference_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Skipping unreadable reference JSON %s: %s", path.name, exc)
        return None

    if not isinstance(data, dict):
        logger.warning("Skipping non-object reference JSON %s", path.name)
        return None

    return data


def _reference_video_url_for_filename(filename: str | None) -> str | None:
    if not filename:
        return None
    try:
        media_path = resolve_reference_media_file(filename)
    except ValueError:
        return None
    if not media_path.is_file():
        return None
    return f"/reference-media/{filename}"


def build_reference_list_item(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    """Build one GET /references entry from a saved output JSON file."""
    json_filename = path.name
    ref_media = data.get("reference_media")
    if not isinstance(ref_media, dict):
        ref_media = {}

    reference_id = ref_media.get("reference_id")
    if not isinstance(reference_id, str) or not reference_id:
        reference_id = reference_id_from_json_filename(json_filename)

    video_metadata = data.get("video_metadata")
    if not isinstance(video_metadata, dict):
        video_metadata = {}

    quality = data.get("quality")
    if not isinstance(quality, dict):
        quality = {}

    duration_raw = video_metadata.get("duration_seconds")
    duration_seconds: float | None
    if duration_raw is None:
        duration_seconds = None
    else:
        try:
            duration_seconds = float(duration_raw)
        except (TypeError, ValueError):
            duration_seconds = None

    reference_quality = quality.get("reference_quality")
    if reference_quality is not None and not isinstance(reference_quality, str):
        reference_quality = str(reference_quality)

    pose_pct = quality.get("pose_detection_percentage")
    pose_detection_percentage: float | None
    if pose_pct is None:
        pose_detection_percentage = None
    else:
        try:
            pose_detection_percentage = float(pose_pct)
        except (TypeError, ValueError):
            pose_detection_percentage = None

    created_at = ref_media.get("created_at")
    if created_at is not None and not isinstance(created_at, str):
        created_at = None
    if not created_at:
        try:
            created_at = _iso_from_mtime(path)
        except OSError:
            created_at = None

    reference_video_filename: str | None = None
    reference_video_url: str | None = None

    stored_video = ref_media.get("reference_video_filename")
    if isinstance(stored_video, str) and stored_video:
        reference_video_filename = stored_video
        reference_video_url = _reference_video_url_for_filename(stored_video)
    elif ref_media:
        # reference_media present but no filename — no playable URL.
        reference_video_filename = None
        reference_video_url = None
    else:
        # Older JSON without reference_media metadata.
        reference_video_filename = None
        reference_video_url = None

    return {
        "reference_id": reference_id or json_filename,
        "json_filename": json_filename,
        "json_url": f"/outputs/{json_filename}",
        "reference_video_filename": reference_video_filename,
        "reference_video_url": reference_video_url,
        "duration_seconds": duration_seconds,
        "created_at": created_at,
        "reference_quality": reference_quality,
        "pose_detection_percentage": pose_detection_percentage,
    }


def list_saved_references() -> list[dict[str, Any]]:
    """Scan outputs/ and return reference summaries newest-first."""
    if not OUTPUT_DIR.is_dir():
        return []

    items: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(OUTPUT_DIR.glob("*.json")):
        if not path.is_file():
            continue
        data = _parse_reference_json(path)
        if data is None:
            continue
        entry = build_reference_list_item(path, data)
        sort_key = entry.get("created_at") or ""
        items.append((sort_key, entry))

    items.sort(key=lambda pair: pair[0], reverse=True)
    return [entry for _, entry in items]
