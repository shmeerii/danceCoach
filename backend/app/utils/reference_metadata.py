"""Attach reference media metadata to saved pose JSON (additive only)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.utils.file_utils import write_json_atomically


def attach_reference_media_to_output_json(
    output_path: Path,
    *,
    reference_id: str,
    reference_video_filename: str,
) -> None:
    """
    Add a top-level reference_media block without changing pose_frames schema.

    Does not remove or alter video_metadata, quality, or pose_frames.
    """
    payload: dict[str, Any] = json.loads(
        output_path.read_text(encoding="utf-8"),
    )
    payload["reference_media"] = {
        "reference_id": reference_id,
        "reference_video_filename": reference_video_filename,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomically(output_path, payload)
