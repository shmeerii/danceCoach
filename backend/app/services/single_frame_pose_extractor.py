"""Single-frame pose extraction for Practice Mode live camera snapshots."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Protocol

import cv2
import numpy as np

from app.config import MAX_LIVE_FRAME_DIMENSION, MAX_LIVE_FRAME_UPLOAD_BYTES
from app.services.pose_extractor import (
    MIN_LANDMARK_VISIBILITY,
    MIN_VISIBLE_LANDMARKS,
    _DetectedPose,
    _ensure_pose_model,
    _pose_from_tasks_result,
)
from app.services.video_preprocessor import generate_frame_variants

logger = logging.getLogger(__name__)

VISIBILITY_THRESHOLD = 0.5

FULL_BODY_LANDMARKS: tuple[str, ...] = (
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

_live_pose_backend: "_ImagePoseBackend | None" = None
_live_pose_backend_lock = threading.Lock()


class _ImagePoseBackend(Protocol):
    def detect(self, frame_rgb: np.ndarray) -> _DetectedPose | None: ...

    def close(self) -> None: ...


def _create_tasks_image_backend() -> _ImagePoseBackend:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    model_path = _ensure_pose_model()
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    class TasksImageBackend:
        def detect(self, frame_rgb: np.ndarray) -> _DetectedPose | None:
            try:
                if not frame_rgb.flags["C_CONTIGUOUS"]:
                    frame_rgb = np.ascontiguousarray(frame_rgb)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=frame_rgb,
                )
                result = landmarker.detect(mp_image)
                return _pose_from_tasks_result(result)
            except Exception as exc:
                logger.warning("MediaPipe IMAGE detection failed: %s", exc)
                return None

        def close(self) -> None:
            landmarker.close()

    return TasksImageBackend()


def _create_legacy_image_backend() -> _ImagePoseBackend:
    import mediapipe as mp

    pose = mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        smooth_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    class LegacyImageBackend:
        def detect(self, frame_rgb: np.ndarray) -> _DetectedPose | None:
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

                from app.services.pose_extractor import _serialize_landmarks

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
                logger.warning("Legacy IMAGE detection failed: %s", exc)
                return None

        def close(self) -> None:
            pose.close()

    return LegacyImageBackend()


def _create_image_pose_backend() -> _ImagePoseBackend:
    try:
        return _create_tasks_image_backend()
    except Exception as tasks_error:
        try:
            return _create_legacy_image_backend()
        except Exception as legacy_error:
            raise RuntimeError(
                "MediaPipe pose detection is unavailable for live frames. "
                f"Tasks API error: {tasks_error}. Legacy API error: {legacy_error}."
            ) from legacy_error


def get_shared_image_pose_backend() -> _ImagePoseBackend:
    """Return a process-wide pose backend reused across live frame requests."""
    global _live_pose_backend
    if _live_pose_backend is not None:
        return _live_pose_backend

    with _live_pose_backend_lock:
        if _live_pose_backend is None:
            _live_pose_backend = _create_image_pose_backend()
        return _live_pose_backend


def _landmarks_by_name(
    landmarks: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {landmark["name"]: landmark for landmark in landmarks}


def _is_landmark_visible(landmark: dict[str, Any]) -> bool:
    visibility = landmark.get("visibility")
    if visibility is None:
        return True
    return float(visibility) >= VISIBILITY_THRESHOLD


def _compute_full_body_metrics(
    landmarks: list[dict[str, Any]],
) -> tuple[bool, float]:
    by_name = _landmarks_by_name(landmarks)
    visible_count = 0
    for name in FULL_BODY_LANDMARKS:
        landmark = by_name.get(name)
        if landmark is not None and _is_landmark_visible(landmark):
            visible_count += 1
    score = round(visible_count / len(FULL_BODY_LANDMARKS), 4)
    full_body_visible = visible_count == len(FULL_BODY_LANDMARKS)
    return full_body_visible, score


def _average_landmark_visibility(landmarks: list[dict[str, Any]]) -> float | None:
    values: list[float] = []
    for landmark in landmarks:
        visibility = landmark.get("visibility")
        if visibility is not None:
            values.append(float(visibility))
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _build_quality(
    landmarks: list[dict[str, Any]] | None,
    *,
    extra_warnings: list[str],
) -> dict[str, Any]:
    warnings = list(extra_warnings)
    if not landmarks:
        return {
            "average_visibility": None,
            "full_body_visible": False,
            "full_body_visibility_score": 0.0,
            "warnings": warnings,
        }

    full_body_visible, full_body_score = _compute_full_body_metrics(landmarks)
    if not full_body_visible:
        warnings.append(
            "Full body not visible: shoulders, hips, knees, and ankles must be "
            "detected with acceptable visibility."
        )

    return {
        "average_visibility": _average_landmark_visibility(landmarks),
        "full_body_visible": full_body_visible,
        "full_body_visibility_score": full_body_score,
        "warnings": warnings,
    }


def _decode_image_bytes(image_bytes: bytes) -> tuple[np.ndarray | None, list[str]]:
    warnings: list[str] = []
    if not image_bytes:
        warnings.append("Uploaded image is empty.")
        return None, warnings

    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    frame_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if frame_bgr is None or frame_bgr.size == 0:
        warnings.append("Could not decode image bytes as a readable image.")
        return None, warnings

    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        warnings.append("Decoded image is not a valid color frame.")
        return None, warnings

    height, width = frame_bgr.shape[:2]
    if height < 32 or width < 32:
        warnings.append("Image resolution is too small for reliable pose detection.")

    return frame_bgr, warnings


def _resize_frame_for_detection(
    frame_bgr: np.ndarray,
    max_dimension: int = MAX_LIVE_FRAME_DIMENSION,
) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    largest_side = max(height, width)
    if largest_side <= max_dimension:
        return frame_bgr

    scale = max_dimension / largest_side
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(
        frame_bgr,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


def _detect_pose_in_frame(
    backend: _ImagePoseBackend,
    frame_bgr: np.ndarray,
) -> tuple[_DetectedPose | None, str | None]:
    """Try original RGB first, then enhanced preprocessing variants."""
    try:
        variants = generate_frame_variants(frame_bgr)
    except Exception as exc:
        logger.warning("Frame preprocessing failed: %s", exc)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        variants = [{"variant_name": "original_rgb", "frame_rgb": rgb}]

    original = next(
        (variant for variant in variants if variant["variant_name"] == "original_rgb"),
        variants[0] if variants else None,
    )
    if original is not None:
        pose = backend.detect(original["frame_rgb"])
        if pose is not None:
            return pose, str(original["variant_name"])

    for variant in variants:
        name = variant["variant_name"]
        if name == "original_rgb":
            continue
        pose = backend.detect(variant["frame_rgb"])
        if pose is not None:
            return pose, name

    return None, None


def _empty_response(
    *,
    timestamp_server: float,
    processing_time_ms: int,
    width: int = 0,
    height: int = 0,
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "success": True,
        "pose_detected": False,
        "timestamp_server": timestamp_server,
        "processing_time_ms": processing_time_ms,
        "image_metadata": {"width": width, "height": height},
        "detection_variant": None,
        "landmarks": [],
        "world_landmarks": [],
        "quality": _build_quality(None, extra_warnings=warnings),
    }


def extract_pose_from_image_bytes(image_bytes: bytes) -> dict[str, Any]:
    """
    Decode a camera snapshot and return pose landmarks for Practice Mode.

    Does not write files to disk. Reuses a shared MediaPipe backend and resizes
    large frames before detection for stable live capture performance.
    """
    started_at = time.perf_counter()
    timestamp_server = round(time.time(), 3)
    decode_warnings: list[str] = []

    if len(image_bytes) > MAX_LIVE_FRAME_UPLOAD_BYTES:
        processing_time_ms = round((time.perf_counter() - started_at) * 1000)
        return _empty_response(
            timestamp_server=timestamp_server,
            processing_time_ms=processing_time_ms,
            warnings=decode_warnings
            + ["Uploaded live frame exceeds the maximum allowed size."],
        )

    frame_bgr, decode_warnings = _decode_image_bytes(image_bytes)
    if frame_bgr is None:
        processing_time_ms = round((time.perf_counter() - started_at) * 1000)
        return _empty_response(
            timestamp_server=timestamp_server,
            processing_time_ms=processing_time_ms,
            warnings=decode_warnings,
        )

    frame_bgr = _resize_frame_for_detection(frame_bgr)
    height, width = frame_bgr.shape[:2]

    try:
        backend = get_shared_image_pose_backend()
        detected, variant_name = _detect_pose_in_frame(backend, frame_bgr)
    except Exception as exc:
        logger.exception("Live frame pose extraction failed")
        processing_time_ms = round((time.perf_counter() - started_at) * 1000)
        failure_warnings = decode_warnings + [
            f"Pose detection unavailable: {exc}",
        ]
        return _empty_response(
            timestamp_server=timestamp_server,
            processing_time_ms=processing_time_ms,
            width=width,
            height=height,
            warnings=failure_warnings,
        )

    processing_time_ms = round((time.perf_counter() - started_at) * 1000)

    if detected is None:
        no_pose_warnings = decode_warnings + ["No pose detected in this frame."]
        return _empty_response(
            timestamp_server=timestamp_server,
            processing_time_ms=processing_time_ms,
            width=width,
            height=height,
            warnings=no_pose_warnings,
        )

    quality = _build_quality(detected.landmarks, extra_warnings=decode_warnings)

    return {
        "success": True,
        "pose_detected": True,
        "timestamp_server": timestamp_server,
        "processing_time_ms": processing_time_ms,
        "image_metadata": {"width": width, "height": height},
        "detection_variant": variant_name,
        "landmarks": detected.landmarks,
        "world_landmarks": detected.world_landmarks,
        "quality": quality,
    }
