"""Reference usability checks from extracted pose landmarks (not dance scoring)."""

from __future__ import annotations

import statistics
from typing import Any

KEY_BODY_LANDMARKS: list[str] = [
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

LEFT_SIDE_LANDMARKS = [
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "left_hip",
    "left_knee",
    "left_ankle",
]

RIGHT_SIDE_LANDMARKS = [
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "right_hip",
    "right_knee",
    "right_ankle",
]

LANDMARK_VISIBILITY_THRESHOLD = 0.5
FULL_BODY_KEY_LANDMARK_RATIO = 0.75
FULL_BODY_VISIBILITY_WARN_PCT = 70.0

BODY_AREA_TOO_SMALL = 0.06
BODY_AREA_TOO_LARGE = 0.72
EDGE_MARGIN = 0.04
EDGE_LANDMARKS_PER_FRAME = 3
EDGE_FRAME_RATIO_WARN = 0.25
SIDE_IMBALANCE_RATIO = 0.55
SIDE_IMBALANCE_FRAME_RATIO = 0.35

BODY_TOO_SMALL_MSG = "The dancer appears too far from the camera."
BODY_TOO_LARGE_MSG = (
    "The dancer may be too close to the camera or partly out of frame."
)
EDGE_CUTOFF_MSG = (
    "The dancer may be too close to the camera or partly out of frame."
)
ANGLE_MSG = "The dancer may be turned sideways or partially hidden."


def _landmark_by_name(landmarks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {landmark["name"]: landmark for landmark in landmarks}


def _is_visible(landmark: dict[str, Any], threshold: float = LANDMARK_VISIBILITY_THRESHOLD) -> bool:
    visibility = landmark.get("visibility")
    if visibility is None:
        return True
    return float(visibility) >= threshold


def _frame_has_full_body(landmarks: list[dict[str, Any]]) -> bool:
    by_name = _landmark_by_name(landmarks)
    visible_keys = [
        name
        for name in KEY_BODY_LANDMARKS
        if name in by_name and _is_visible(by_name[name])
    ]
    if not visible_keys:
        return False

    ratio = len(visible_keys) / len(KEY_BODY_LANDMARKS)
    if ratio < FULL_BODY_KEY_LANDMARK_RATIO:
        return False

    if "nose" not in by_name or not _is_visible(by_name["nose"]):
        return False

    ankle_visible = ("left_ankle" in by_name and _is_visible(by_name["left_ankle"])) or (
        "right_ankle" in by_name and _is_visible(by_name["right_ankle"])
    )
    hip_visible = ("left_hip" in by_name and _is_visible(by_name["left_hip"])) or (
        "right_hip" in by_name and _is_visible(by_name["right_hip"])
    )
    return ankle_visible and hip_visible


def _body_bbox_area(landmarks: list[dict[str, Any]]) -> float | None:
    visible_points = [
        (float(lm["x"]), float(lm["y"]))
        for lm in landmarks
        if _is_visible(lm)
    ]
    if len(visible_points) < 4:
        return None

    xs = [point[0] for point in visible_points]
    ys = [point[1] for point in visible_points]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    if width <= 0 or height <= 0:
        return None
    return width * height


def _frame_has_edge_cutoff(landmarks: list[dict[str, Any]]) -> bool:
    edge_count = 0
    for landmark in landmarks:
        if not _is_visible(landmark):
            continue
        x = float(landmark["x"])
        y = float(landmark["y"])
        if (
            x <= EDGE_MARGIN
            or x >= 1.0 - EDGE_MARGIN
            or y <= EDGE_MARGIN
            or y >= 1.0 - EDGE_MARGIN
        ):
            edge_count += 1
    return edge_count >= EDGE_LANDMARKS_PER_FRAME


def _side_visibility_score(
    by_name: dict[str, dict[str, Any]], side_names: list[str]
) -> float | None:
    values: list[float] = []
    for name in side_names:
        landmark = by_name.get(name)
        if landmark is None:
            continue
        visibility = landmark.get("visibility")
        if visibility is None:
            values.append(1.0)
        else:
            values.append(float(visibility))
    if not values:
        return None
    return sum(values) / len(values)


def _frame_has_side_imbalance(landmarks: list[dict[str, Any]]) -> bool:
    by_name = _landmark_by_name(landmarks)
    left_score = _side_visibility_score(by_name, LEFT_SIDE_LANDMARKS)
    right_score = _side_visibility_score(by_name, RIGHT_SIDE_LANDMARKS)
    if left_score is None or right_score is None:
        return False
    if left_score <= 0 or right_score <= 0:
        return False
    return (
        left_score < SIDE_IMBALANCE_RATIO * right_score
        or right_score < SIDE_IMBALANCE_RATIO * left_score
    )


def _find_low_confidence_segments(
    pose_frames: list[dict[str, Any]],
) -> list[dict[str, float | int]]:
    """Merge consecutive frames with pose_detected=false into time ranges."""
    segments: list[dict[str, float | int]] = []
    segment_start: dict[str, Any] | None = None
    previous_frame: dict[str, Any] | None = None

    for frame in pose_frames:
        if not frame.get("pose_detected"):
            if segment_start is None:
                segment_start = frame
        elif segment_start is not None and previous_frame is not None:
            segments.append(
                {
                    "start_frame": int(segment_start["frame_index"]),
                    "end_frame": int(previous_frame["frame_index"]),
                    "start_time": round(float(segment_start["timestamp_seconds"]), 4),
                    "end_time": round(float(previous_frame["timestamp_seconds"]), 4),
                }
            )
            segment_start = None
        previous_frame = frame

    if segment_start is not None and previous_frame is not None:
        segments.append(
            {
                "start_frame": int(segment_start["frame_index"]),
                "end_frame": int(previous_frame["frame_index"]),
                "start_time": round(float(segment_start["timestamp_seconds"]), 4),
                "end_time": round(float(previous_frame["timestamp_seconds"]), 4),
            }
        )

    return segments


def _build_user_guidance(
    *,
    full_body_visibility_percentage: float,
    body_size_warning: str | None,
    edge_cutoff_warning: str | None,
    angle_warning: str | None,
    low_confidence_segments: list[dict[str, Any]],
    lighting_warnings: list[str],
) -> list[str]:
    guidance: list[str] = []

    if full_body_visibility_percentage < FULL_BODY_VISIBILITY_WARN_PCT:
        guidance.append("Record with the full body visible from head to feet.")

    if body_size_warning == BODY_TOO_SMALL_MSG:
        guidance.append(
            "Move the camera closer so the full body fills more of the frame."
        )
    elif body_size_warning == BODY_TOO_LARGE_MSG:
        guidance.append("Move the camera farther back.")

    if edge_cutoff_warning is not None:
        if "Record with the full body visible from head to feet." not in guidance:
            guidance.append("Record with the full body visible from head to feet.")

    if angle_warning is not None:
        guidance.append("Avoid filming from extreme side angles.")

    if lighting_warnings:
        guidance.append("Use brighter lighting facing the dancer.")

    if len(low_confidence_segments) >= 2:
        guidance.append(
            "Keep the dancer fully in view for the entire recording; "
            "several sections had missing pose detection."
        )
    elif low_confidence_segments:
        guidance.append("Keep the dancer fully in view throughout the recording.")

    guidance.append("Keep the camera stable.")
    guidance.append("Use a plain background if possible.")

    seen: set[str] = set()
    ordered: list[str] = []
    for message in guidance:
        if message not in seen:
            seen.add(message)
            ordered.append(message)
    return ordered


def compute_reference_usability(
    pose_frames: list[dict[str, Any]],
    *,
    lighting_warnings: list[str] | None = None,
) -> dict[str, Any]:
    """
    Analyze landmark quality across detected frames for reference suitability.

    These are extraction-quality checks only, not dance scores or coaching.
    """
    lighting_warnings = lighting_warnings or []
    detected_frames = [frame for frame in pose_frames if frame.get("pose_detected")]

    if not detected_frames:
        low_confidence_segments = _find_low_confidence_segments(pose_frames)
        return {
            "full_body_visibility_percentage": 0.0,
            "body_size_warning": None,
            "edge_cutoff_warning": None,
            "angle_warning": None,
            "low_confidence_segments": low_confidence_segments,
            "user_guidance": _build_user_guidance(
                full_body_visibility_percentage=0.0,
                body_size_warning=None,
                edge_cutoff_warning=None,
                angle_warning=None,
                low_confidence_segments=low_confidence_segments,
                lighting_warnings=lighting_warnings,
            ),
        }

    full_body_count = sum(
        1
        for frame in detected_frames
        if _frame_has_full_body(frame.get("landmarks", []))
    )
    full_body_visibility_percentage = round(
        (full_body_count / len(detected_frames)) * 100.0, 2
    )

    body_areas: list[float] = []
    edge_frame_count = 0
    side_imbalance_count = 0

    for frame in detected_frames:
        landmarks = frame.get("landmarks", [])
        area = _body_bbox_area(landmarks)
        if area is not None:
            body_areas.append(area)
        if _frame_has_edge_cutoff(landmarks):
            edge_frame_count += 1
        if _frame_has_side_imbalance(landmarks):
            side_imbalance_count += 1

    body_size_warning: str | None = None
    if body_areas:
        median_area = float(statistics.median(body_areas))
        if median_area < BODY_AREA_TOO_SMALL:
            body_size_warning = BODY_TOO_SMALL_MSG
        elif median_area > BODY_AREA_TOO_LARGE:
            body_size_warning = BODY_TOO_LARGE_MSG

    edge_cutoff_warning: str | None = None
    edge_ratio = edge_frame_count / len(detected_frames)
    if edge_ratio >= EDGE_FRAME_RATIO_WARN:
        edge_cutoff_warning = EDGE_CUTOFF_MSG

    angle_warning: str | None = None
    side_ratio = side_imbalance_count / len(detected_frames)
    if side_ratio >= SIDE_IMBALANCE_FRAME_RATIO:
        angle_warning = ANGLE_MSG

    low_confidence_segments = _find_low_confidence_segments(pose_frames)

    return {
        "full_body_visibility_percentage": full_body_visibility_percentage,
        "body_size_warning": body_size_warning,
        "edge_cutoff_warning": edge_cutoff_warning,
        "angle_warning": angle_warning,
        "low_confidence_segments": low_confidence_segments,
        "user_guidance": _build_user_guidance(
            full_body_visibility_percentage=full_body_visibility_percentage,
            body_size_warning=body_size_warning,
            edge_cutoff_warning=edge_cutoff_warning,
            angle_warning=angle_warning,
            low_confidence_segments=low_confidence_segments,
            lighting_warnings=lighting_warnings,
        ),
    }
