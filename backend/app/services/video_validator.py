"""Video validation before pose extraction.

Used by the stable reference-video pipeline. Do not modify for live camera mode
unless validation rules must apply to both flows intentionally.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.config import MAX_FILE_SIZE_BYTES, MAX_VIDEO_SECONDS

logger = logging.getLogger(__name__)

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}

MIN_DIMENSION_REJECT_PX = 240
MIN_DIMENSION_WARN_PX = 480
MIN_FPS_WARN = 15.0
MAX_FPS_WARN = 60.0
DARK_MEAN_LUMA = 45.0
BRIGHT_MEAN_LUMA = 210.0
LOW_CONTRAST_STD = 18.0

SAMPLE_POSITIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass
class VideoValidationResult:
    """Outcome of pre-extraction video checks."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }


def _empty_metadata() -> dict[str, Any]:
    return {
        "fps": 0.0,
        "frame_count": 0,
        "duration_seconds": 0.0,
        "width": 0,
        "height": 0,
    }


def _sample_frame_indices(frame_count: int) -> list[int]:
    if frame_count <= 0:
        return []
    if frame_count == 1:
        return [0]

    indices: list[int] = []
    for position in SAMPLE_POSITIONS:
        index = int(round(position * (frame_count - 1)))
        indices.append(max(0, min(index, frame_count - 1)))

    return sorted(set(indices))


def _read_frame_at(cap: cv2.VideoCapture, frame_index: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def _frame_luminance_stats(frame: np.ndarray) -> tuple[float, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_luma = float(np.mean(gray))
    contrast_std = float(np.std(gray))
    return mean_luma, contrast_std


def _analyze_sampled_frames(
    cap: cv2.VideoCapture, frame_count: int
) -> tuple[list[float], list[float], list[str]]:
    """Return per-sample mean luma, contrast std, and read failures."""
    means: list[float] = []
    stds: list[float] = []
    read_errors: list[str] = []

    for index in _sample_frame_indices(frame_count):
        frame = _read_frame_at(cap, index)
        if frame is None:
            read_errors.append(f"Could not read sample frame at index {index}.")
            continue
        mean_luma, contrast_std = _frame_luminance_stats(frame)
        means.append(mean_luma)
        stds.append(contrast_std)

    return means, stds, read_errors


def _brightness_contrast_warnings(
    means: list[float], stds: list[float], read_errors: list[str]
) -> list[str]:
    warnings: list[str] = list(read_errors)

    if not means:
        warnings.append(
            "Could not sample frames for brightness and contrast checks."
        )
        return warnings

    avg_mean = float(np.mean(means))
    avg_std = float(np.mean(stds))
    dark_samples = sum(1 for value in means if value < DARK_MEAN_LUMA)
    bright_samples = sum(1 for value in means if value > BRIGHT_MEAN_LUMA)
    low_contrast_samples = sum(1 for value in stds if value < LOW_CONTRAST_STD)

    if avg_mean < DARK_MEAN_LUMA or dark_samples >= 3:
        warnings.append(
            "Video appears very dark at sampled frames; pose detection may fail. "
            "Try brighter, even lighting."
        )
    elif dark_samples >= 2:
        warnings.append(
            "Some sampled frames are very dark; reference quality may be reduced."
        )

    if avg_mean > BRIGHT_MEAN_LUMA or bright_samples >= 3:
        warnings.append(
            "Video appears very bright or washed out at sampled frames; "
            "landmarks may be unreliable."
        )
    elif bright_samples >= 2:
        warnings.append(
            "Some sampled frames look overexposed; reference quality may suffer."
        )

    if avg_std < LOW_CONTRAST_STD or low_contrast_samples >= 3:
        warnings.append(
            "Sampled frames have low contrast; the dancer may be hard to separate "
            "from the background."
        )
    elif low_contrast_samples >= 2:
        warnings.append(
            "Some sampled frames have low contrast; pose visibility may be reduced."
        )

    return warnings


def validate_video(video_path: Path, file_size_bytes: int) -> VideoValidationResult:
    """
    Validate an uploaded dance video for reference pose extraction.

    Reference pipeline only — do not modify for live camera mode unless shared
    validation rules are updated intentionally.

    Checks file presence, extension, size, OpenCV readability, timing,
    resolution, and sampled-frame lighting/contrast heuristics.
    """
    errors: list[str] = []
    warnings: list[str] = []
    metadata = _empty_metadata()

    if not video_path.exists():
        return VideoValidationResult(
            is_valid=False,
            errors=["No video file was found at the upload path."],
            warnings=warnings,
            metadata=metadata,
        )

    if file_size_bytes <= 0:
        return VideoValidationResult(
            is_valid=False,
            errors=["Uploaded file is empty."],
            warnings=warnings,
            metadata=metadata,
        )

    extension = video_path.suffix.lower()
    if extension not in ALLOWED_VIDEO_EXTENSIONS:
        allowed = ", ".join(sorted(ext.lstrip(".") for ext in ALLOWED_VIDEO_EXTENSIONS))
        return VideoValidationResult(
            is_valid=False,
            errors=[
                f"Unsupported file extension '{extension or '(none)'}'. "
                f"Allowed video types: {allowed}."
            ],
            warnings=warnings,
            metadata=metadata,
        )

    if file_size_bytes > MAX_FILE_SIZE_BYTES:
        max_mb = MAX_FILE_SIZE_BYTES // (1024 * 1024)
        return VideoValidationResult(
            is_valid=False,
            errors=[f"File exceeds maximum size of {max_mb} MB."],
            warnings=warnings,
            metadata=metadata,
        )

    cap = cv2.VideoCapture(str(video_path))
    try:
        try:
            return _validate_open_video(cap, errors, warnings, metadata)
        except Exception as exc:
            logger.exception("Unexpected error validating video: %s", video_path.name)
            return VideoValidationResult(
                is_valid=False,
                errors=[f"Video validation failed: {exc}"],
                warnings=warnings,
                metadata=metadata,
            )
    finally:
        cap.release()


def _validate_open_video(
    cap: cv2.VideoCapture,
    errors: list[str],
    warnings: list[str],
    metadata: dict[str, Any],
) -> VideoValidationResult:
    try:
        if not cap.isOpened():
            return VideoValidationResult(
                is_valid=False,
                errors=["Video file could not be opened for reading."],
                warnings=warnings,
                metadata=metadata,
            )

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        if frame_count <= 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, probe_frame = cap.read()
            if not ok or probe_frame is None:
                return VideoValidationResult(
                    is_valid=False,
                    errors=["Video has zero readable frames."],
                    warnings=warnings,
                    metadata=metadata,
                )
            frame_count = 1
            height, width = probe_frame.shape[:2]
            warnings.append(
                "Frame count metadata is missing; validation used a single readable frame."
            )

        if fps <= 0:
            return VideoValidationResult(
                is_valid=False,
                errors=["Video reports zero FPS and cannot be timed reliably."],
                warnings=warnings,
                metadata={
                    "fps": fps,
                    "frame_count": frame_count,
                    "duration_seconds": 0.0,
                    "width": width,
                    "height": height,
                },
            )

        duration_seconds = frame_count / fps
        metadata = {
            "fps": round(fps, 3),
            "frame_count": frame_count,
            "duration_seconds": round(duration_seconds, 3),
            "width": width,
            "height": height,
        }

        smaller_side = min(width, height)
        if smaller_side < MIN_DIMENSION_REJECT_PX:
            errors.append(
                f"Resolution is too small ({width}x{height}). "
                f"Minimum {MIN_DIMENSION_REJECT_PX}px on the shorter side is required."
            )
        elif smaller_side < MIN_DIMENSION_WARN_PX:
            warnings.append(
                f"Resolution is low ({width}x{height}). "
                f"At least {MIN_DIMENSION_WARN_PX}px on the shorter side is recommended "
                "for reliable pose landmarks."
            )

        if duration_seconds > MAX_VIDEO_SECONDS:
            errors.append(
                f"Video duration ({duration_seconds:.1f}s) exceeds the maximum "
                f"of {MAX_VIDEO_SECONDS}s."
            )

        if fps < MIN_FPS_WARN:
            warnings.append(
                f"Frame rate is very low ({fps:.2f} FPS). "
                "Fast movements may be missed as a reference."
            )
        elif fps > MAX_FPS_WARN:
            warnings.append(
                f"Frame rate is very high ({fps:.2f} FPS). "
                "Processing will be heavier; consider a lower-FPS export."
            )

        if height > width:
            warnings.append(
                "Video is portrait orientation. Ensure the dancer is fully visible "
                "and not cropped by vertical framing."
            )

        means, stds, read_errors = _analyze_sampled_frames(cap, frame_count)
        warnings.extend(_brightness_contrast_warnings(means, stds, read_errors))
    except Exception as exc:
        logger.exception("Error reading video metadata during validation")
        return VideoValidationResult(
            is_valid=False,
            errors=[f"Video could not be read reliably: {exc}"],
            warnings=warnings,
            metadata=metadata,
        )

    is_valid = len(errors) == 0
    return VideoValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        metadata=metadata,
    )
