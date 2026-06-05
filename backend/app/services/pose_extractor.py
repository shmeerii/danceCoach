"""Pose landmark extraction from dance videos.

Reference video extraction is the stable base pipeline. Do not modify this
module for live camera mode unless updating shared pose schema intentionally.
"""

from __future__ import annotations

import logging
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from app.config import (
    BACKEND_ROOT,
    FRAME_STRIDE,
    MIN_AVERAGE_VISIBILITY,
    MIN_POSE_DETECTION_PERCENTAGE,
    OUTPUT_DIR,
)
from app.services.reference_usability import compute_reference_usability
from app.services.video_preprocessor import (
    estimate_frame_quality,
    generate_frame_variants,
)
from app.services.video_validator import validate_video
from app.utils.file_utils import write_json_atomically

logger = logging.getLogger(__name__)

POSE_MODEL_DIR = BACKEND_ROOT / "models"
POSE_MODEL_FILENAME = "pose_landmarker_lite.task"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

MIN_VISIBLE_LANDMARKS = 11
MIN_LANDMARK_VISIBILITY = 0.5

POSE_LANDMARK_NAMES: list[str] = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]

LIGHTING_SAMPLE_POSITIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass
class _DetectedPose:
    landmarks: list[dict[str, Any]]
    world_landmarks: list[dict[str, Any]]


@dataclass
class PoseExtractionResult:
    """Result of a pose extraction run."""

    success: bool
    output_path: Path | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    quality_metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class _PoseBackend(Protocol):
    def detect(self, frame_rgb: np.ndarray, timestamp_ms: int) -> _DetectedPose | None: ...

    def close(self) -> None: ...


def _ensure_pose_model() -> Path:
    POSE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = POSE_MODEL_DIR / POSE_MODEL_FILENAME
    if model_path.is_file():
        return model_path

    try:
        urllib.request.urlretrieve(POSE_MODEL_URL, model_path)
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not download pose model to {model_path}. "
            f"Download manually from {POSE_MODEL_URL}. Error: {exc}"
        ) from exc
    return model_path


def _landmark_visibility(landmark: Any) -> float | None:
    visibility = getattr(landmark, "visibility", None)
    if visibility is None:
        presence = getattr(landmark, "presence", None)
        if presence is not None:
            visibility = presence
    if visibility is None:
        return None
    return round(float(visibility), 4)


def _serialize_landmarks(landmark_list: Any) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for index, landmark in enumerate(landmark_list):
        name = (
            POSE_LANDMARK_NAMES[index]
            if index < len(POSE_LANDMARK_NAMES)
            else f"landmark_{index}"
        )
        serialized.append(
            {
                "id": index,
                "name": name,
                "x": round(float(landmark.x), 6),
                "y": round(float(landmark.y), 6),
                "z": round(float(landmark.z), 6),
                "visibility": _landmark_visibility(landmark),
            }
        )
    return serialized


def _pose_from_tasks_result(result: Any) -> _DetectedPose | None:
    if not result.pose_landmarks:
        return None
    landmarks = result.pose_landmarks[0]
    if not landmarks:
        return None

    visible_count = sum(
        1
        for landmark in landmarks
        if (_landmark_visibility(landmark) or 0.0) >= MIN_LANDMARK_VISIBILITY
    )
    if visible_count < MIN_VISIBLE_LANDMARKS:
        return None

    world_list: list[Any] = []
    if result.pose_world_landmarks:
        world_list = result.pose_world_landmarks[0]

    return _DetectedPose(
        landmarks=_serialize_landmarks(landmarks),
        world_landmarks=_serialize_landmarks(world_list) if world_list else [],
    )


def _create_tasks_backend() -> _PoseBackend:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    model_path = _ensure_pose_model()
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    class TasksBackend:
        def detect(self, frame_rgb: np.ndarray, timestamp_ms: int) -> _DetectedPose | None:
            try:
                if not frame_rgb.flags["C_CONTIGUOUS"]:
                    frame_rgb = np.ascontiguousarray(frame_rgb)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=frame_rgb,
                )
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                return _pose_from_tasks_result(result)
            except Exception as exc:
                logger.warning(
                    "MediaPipe detection failed for timestamp %s: %s",
                    timestamp_ms,
                    exc,
                )
                return None

        def close(self) -> None:
            landmarker.close()

    return TasksBackend()


def _create_legacy_backend() -> _PoseBackend:
    import mediapipe as mp

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    class LegacyBackend:
        def detect(self, frame_rgb: np.ndarray, timestamp_ms: int) -> _DetectedPose | None:
            _ = timestamp_ms
            try:
                results = pose.process(frame_rgb)
                if not results.pose_landmarks:
                    return None

                landmarks = results.pose_landmarks.landmark
                visible_count = sum(
                    1
                    for landmark in landmarks
                    if landmark.visibility >= MIN_LANDMARK_VISIBILITY
                )
                if visible_count < MIN_VISIBLE_LANDMARKS:
                    return None

                world_list = (
                    results.pose_world_landmarks.landmark
                    if results.pose_world_landmarks
                    else []
                )
                return _DetectedPose(
                    landmarks=_serialize_landmarks(landmarks),
                    world_landmarks=_serialize_landmarks(world_list)
                    if world_list
                    else [],
                )
            except Exception as exc:
                logger.warning(
                    "Legacy MediaPipe detection failed for timestamp %s: %s",
                    timestamp_ms,
                    exc,
                )
                return None

        def close(self) -> None:
            pose.close()

    return LegacyBackend()


def _create_pose_backend() -> tuple[_PoseBackend, str]:
    try:
        return _create_tasks_backend(), "mediapipe_tasks_video"
    except Exception as tasks_error:
        try:
            return _create_legacy_backend(), "mediapipe_solutions_pose"
        except Exception as legacy_error:
            raise RuntimeError(
                "MediaPipe pose detection is unavailable. "
                f"Tasks API error: {tasks_error}. Legacy API error: {legacy_error}."
            ) from legacy_error


def _collect_lighting_warnings(cap: cv2.VideoCapture, frame_count: int) -> list[str]:
    warnings: list[str] = []
    if frame_count <= 0:
        return warnings

    indices = sorted(
        {
            int(round(position * (frame_count - 1)))
            for position in LIGHTING_SAMPLE_POSITIONS
        }
    )

    dark_count = 0
    bright_count = 0
    low_contrast_count = 0

    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        quality = estimate_frame_quality(frame)
        if quality["is_too_dark"]:
            dark_count += 1
        if quality["is_too_bright"]:
            bright_count += 1
        if quality["is_low_contrast"]:
            low_contrast_count += 1

    if dark_count >= 3:
        warnings.append(
            "Many sampled frames are very dark; pose detection may be unreliable."
        )
    elif dark_count >= 2:
        warnings.append("Some sampled frames are very dark.")

    if bright_count >= 3:
        warnings.append(
            "Many sampled frames are overexposed; landmarks may be unreliable."
        )
    elif bright_count >= 2:
        warnings.append("Some sampled frames appear overexposed.")

    if low_contrast_count >= 3:
        warnings.append(
            "Many sampled frames have low contrast; the dancer may be hard to detect."
        )
    elif low_contrast_count >= 2:
        warnings.append("Some sampled frames have low contrast.")

    return warnings


def _average_visibility(pose_frames: list[dict[str, Any]]) -> float | None:
    values: list[float] = []
    for frame in pose_frames:
        if not frame.get("pose_detected"):
            continue
        for landmark in frame.get("landmarks", []):
            visibility = landmark.get("visibility")
            if visibility is not None:
                values.append(float(visibility))
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _reference_quality(
    pose_detection_percentage: float,
    average_visibility: float | None,
) -> str:
    if pose_detection_percentage <= 20:
        return "failed"
    if pose_detection_percentage < 60:
        return "poor"

    if average_visibility is None:
        if pose_detection_percentage >= 90:
            return "good"
        if pose_detection_percentage >= 75:
            return "good"
        return "usable"

    if pose_detection_percentage >= 90 and average_visibility >= 0.65:
        return "excellent"
    if pose_detection_percentage >= 75 and average_visibility >= 0.55:
        return "good"
    if pose_detection_percentage >= 60 and average_visibility >= 0.45:
        return "usable"

    return "poor"


# Stable reference JSON shape — do not change for live camera unless intentional.
def _build_output_payload(
    *,
    video_metadata: dict[str, Any],
    quality: dict[str, Any],
    pose_frames: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "video_metadata": video_metadata,
        "quality": quality,
        "pose_frames": pose_frames,
    }


def extract_pose_from_video(
    video_path: Path,
    *,
    output_basename: str | None = None,
) -> PoseExtractionResult:
    """
    Extract per-frame pose landmarks and write JSON to OUTPUT_DIR.

    Validates the video, processes every FRAME_STRIDE frame (including failures),
    tries preprocessing variants per frame, and reports honest quality metrics.

    Reference video extraction entry point (stable). Do not modify this for live
    camera mode unless updating shared pose schema intentionally.
    """
    warnings: list[str] = []
    errors: list[str] = []

    if not video_path.is_file():
        return PoseExtractionResult(
            success=False,
            errors=[f"Video file not found: {video_path.name}"],
        )

    file_size_bytes = video_path.stat().st_size
    validation = validate_video(video_path, file_size_bytes=file_size_bytes)
    if not validation.is_valid:
        return PoseExtractionResult(
            success=False,
            errors=validation.errors,
            warnings=validation.warnings,
            metadata={"validation": validation.to_dict()},
        )

    warnings.extend(validation.warnings)
    video_meta = validation.metadata

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return PoseExtractionResult(
            success=False,
            errors=["Could not open video for pose extraction."],
            warnings=warnings,
        )

    backend: _PoseBackend | None = None
    last_frame_timestamp_ms = -1
    pose_frames: list[dict[str, Any]] = []
    pose_detected_count = 0
    first_pose_frame: int | None = None
    last_pose_frame: int | None = None
    detector_name = "unknown"

    try:
        backend, detector_name = _create_pose_backend()

        fps = float(video_meta.get("fps") or cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(video_meta.get("frame_count") or cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(video_meta.get("width") or cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(video_meta.get("height") or cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration_seconds = float(video_meta.get("duration_seconds") or 0.0)
        if duration_seconds <= 0 and fps > 0 and frame_count > 0:
            duration_seconds = frame_count / fps

        lighting_warnings = _collect_lighting_warnings(cap, frame_count)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame_index = 0
        processed_frames = 0

        while True:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break

            if frame_index % FRAME_STRIDE != 0:
                frame_index += 1
                continue

            timestamp_seconds = frame_index / fps if fps > 0 else 0.0

            try:
                variants = generate_frame_variants(frame_bgr)
            except Exception as exc:
                logger.warning(
                    "Frame preprocessing failed at index %s: %s",
                    frame_index,
                    exc,
                )
                variants = [
                    {
                        "variant_name": "original_rgb",
                        "frame_rgb": cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
                    }
                ]

            detected_pose: _DetectedPose | None = None
            detection_variant: str | None = None

            # MediaPipe Tasks VIDEO mode expects monotonically increasing timestamps.
            # We base timestamps on the actual frame time (derived from fps) and then
            # add a tiny offset per preprocessing variant attempt without affecting
            # the output schema (which reports the real frame timestamp).
            candidate_frame_timestamp_ms = int(round(timestamp_seconds * 1000.0))
            if candidate_frame_timestamp_ms <= last_frame_timestamp_ms:
                candidate_frame_timestamp_ms = last_frame_timestamp_ms + 1
            last_frame_timestamp_ms = candidate_frame_timestamp_ms

            for variant_idx, variant in enumerate(variants):
                variant_timestamp_ms = candidate_frame_timestamp_ms + variant_idx
                pose = backend.detect(variant["frame_rgb"], variant_timestamp_ms)
                if pose is not None:
                    detected_pose = pose
                    detection_variant = variant["variant_name"]
                    break

            if detected_pose is not None:
                pose_detected_count += 1
                if first_pose_frame is None:
                    first_pose_frame = frame_index
                last_pose_frame = frame_index

            pose_frames.append(
                {
                    "frame_index": frame_index,
                    "timestamp_seconds": round(timestamp_seconds, 4),
                    "pose_detected": detected_pose is not None,
                    "detection_variant": detection_variant,
                    "landmarks": detected_pose.landmarks if detected_pose else [],
                    "world_landmarks": (
                        detected_pose.world_landmarks if detected_pose else []
                    ),
                }
            )

            processed_frames += 1
            frame_index += 1

        if processed_frames == 0:
            return PoseExtractionResult(
                success=False,
                errors=["No frames were processed from the video."],
                warnings=warnings,
            )

        pose_detection_percentage = round(
            (pose_detected_count / processed_frames) * 100.0, 2
        )
        average_visibility = _average_visibility(pose_frames)
        reference_quality = _reference_quality(
            pose_detection_percentage, average_visibility
        )

        usability = compute_reference_usability(
            pose_frames,
            lighting_warnings=lighting_warnings,
        )

        quality = {
            "pose_detection_percentage": pose_detection_percentage,
            "average_visibility": average_visibility,
            "first_pose_frame": first_pose_frame,
            "last_pose_frame": last_pose_frame,
            "lighting_warnings": lighting_warnings,
            "video_warnings": list(validation.warnings),
            "reference_quality": reference_quality,
            **usability,
        }

        output_metadata = {
            "fps": video_meta.get("fps", fps),
            "frame_count": frame_count,
            "duration_seconds": video_meta.get("duration_seconds", duration_seconds),
            "width": width,
            "height": height,
            "processed_frames": processed_frames,
            "pose_detected_frames": pose_detected_count,
            "frame_stride": FRAME_STRIDE,
        }

        payload = _build_output_payload(
            video_metadata=output_metadata,
            quality=quality,
            pose_frames=pose_frames,
        )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if output_basename:
            output_name = output_basename
        else:
            output_name = f"{video_path.stem}_poses.json"
        if not output_name.endswith(".json"):
            output_name = f"{output_name}.json"

        output_path = OUTPUT_DIR / output_name
        write_json_atomically(output_path, payload)

        if pose_detection_percentage < MIN_POSE_DETECTION_PERCENTAGE:
            warnings.append(
                f"Pose was detected in only {pose_detection_percentage:.1f}% of "
                f"processed frames (minimum recommended: "
                f"{MIN_POSE_DETECTION_PERCENTAGE}%)."
            )

        if (
            average_visibility is not None
            and average_visibility < MIN_AVERAGE_VISIBILITY
        ):
            warnings.append(
                f"Average landmark visibility ({average_visibility:.2f}) is below "
                f"the recommended minimum ({MIN_AVERAGE_VISIBILITY})."
            )

        if reference_quality in {"poor", "failed"}:
            warnings.append(
                f"Reference quality is '{reference_quality}'. "
                "This video may not be suitable as a dance reference."
            )
        elif reference_quality == "usable":
            warnings.append(
                "Reference quality is 'usable' but not ideal; consider re-recording "
                "with better lighting and framing."
            )

        for usability_warning in (
            usability.get("body_size_warning"),
            usability.get("edge_cutoff_warning"),
            usability.get("angle_warning"),
        ):
            if usability_warning and usability_warning not in warnings:
                warnings.append(usability_warning)

        if (
            usability.get("full_body_visibility_percentage", 100.0)
            < 70.0
            and pose_detected_count > 0
        ):
            warnings.append(
                "Full body visibility is low across detected frames; head-to-feet "
                "framing is recommended for a reference video."
            )

        return PoseExtractionResult(
            success=True,
            output_path=output_path,
            warnings=warnings,
            errors=errors,
            quality_metrics=quality,
            metadata={
                "source_video": video_path.name,
                "detector_backend": detector_name,
                "video_metadata": output_metadata,
            },
        )

    except Exception as exc:
        logger.exception("Pose extraction failed for %s", video_path.name)
        return PoseExtractionResult(
            success=False,
            errors=[f"Pose extraction failed: {exc}"],
            warnings=warnings,
        )
    finally:
        if cap.isOpened():
            cap.release()
        if backend is not None:
            try:
                backend.close()
            except Exception:
                logger.exception("Error closing pose detection backend")
