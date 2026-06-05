#!/usr/bin/env python3
"""
Smoke test: run reference pose extraction locally (same as POST /extract-pose).

Processes a local video through extract_pose_from_video and verifies the saved
JSON contains video_metadata, quality, and pose_frames.

Run from the backend/ directory:

  python scripts/test_extract_pose.py path/to/video.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

# Allow `from app...` when invoked as a script from backend/.
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.services.pose_extractor import extract_pose_from_video
from app.utils.file_utils import ensure_runtime_directories

# Keys required in saved reference pose JSON (stable schema).
_REQUIRED_OUTPUT_KEYS = ("video_metadata", "quality", "pose_frames")


def _verify_saved_output_json(output_path: Path) -> tuple[bool, str | None]:
    """Smoke-check that extraction wrote the stable reference JSON shape."""
    try:
        data = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"Could not read output JSON: {exc}"

    missing = [key for key in _REQUIRED_OUTPUT_KEYS if key not in data]
    if missing:
        return False, f"Output JSON missing required keys: {', '.join(missing)}"

    if not isinstance(data["pose_frames"], list):
        return False, "pose_frames must be a list"

    for key in ("video_metadata", "quality"):
        if not isinstance(data[key], dict):
            return False, f"{key} must be an object"

    return True, None


def _print_section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def _print_result(video_path: Path, output_basename: str | None) -> int:
    ensure_runtime_directories()

    print(f"Input video: {video_path.resolve()}")
    print("Running validation and pose extraction...")

    result = extract_pose_from_video(
        video_path,
        output_basename=output_basename,
    )

    if not result.success:
        _print_section("Extraction failed")
        if result.errors:
            print("Errors:")
            for error in result.errors:
                print(f"  - {error}")
        else:
            print("  (no error details returned)")

        validation = result.metadata.get("validation")
        if validation:
            print()
            print("Validation details:")
            if validation.get("errors"):
                for error in validation["errors"]:
                    print(f"  - {error}")
            if validation.get("warnings"):
                print("Warnings:")
                for warning in validation["warnings"]:
                    print(f"  - {warning}")

        if result.warnings:
            print()
            print("Warnings:")
            for warning in result.warnings:
                print(f"  - {warning}")

        return 1

    video_metadata = result.metadata.get("video_metadata", {})
    quality = result.quality_metrics or {}

    fps = video_metadata.get("fps")
    duration = video_metadata.get("duration_seconds")
    width = video_metadata.get("width")
    height = video_metadata.get("height")
    processed_frames = video_metadata.get("processed_frames", 0)
    pose_detected_frames = video_metadata.get("pose_detected_frames", 0)

    resolution = (
        f"{width} x {height}" if width and height else "unknown"
    )

    _print_section("Video")
    print(f"FPS:                    {fps}")
    print(f"Duration (seconds):     {duration}")
    print(f"Resolution:             {resolution}")

    _print_section("Pose extraction")
    print(f"Processed frames:       {processed_frames}")
    print(f"Pose detected frames:   {pose_detected_frames}")
    print(
        f"Pose detection %:       {quality.get('pose_detection_percentage', 'n/a')}"
    )
    print(f"Average visibility:     {quality.get('average_visibility', 'n/a')}")
    print(
        "Full body visibility %: "
        f"{quality.get('full_body_visibility_percentage', 'n/a')}"
    )
    print(f"Reference quality:      {quality.get('reference_quality', 'n/a')}")

    all_warnings = list(result.warnings)
    video_warnings = quality.get("video_warnings") or []
    lighting_warnings = quality.get("lighting_warnings") or []
    user_guidance = quality.get("user_guidance") or []

    combined_warnings: list[str] = []
    seen: set[str] = set()
    for message in all_warnings + video_warnings + lighting_warnings:
        if message and message not in seen:
            seen.add(message)
            combined_warnings.append(message)

    _print_section("Warnings")
    if combined_warnings:
        for warning in combined_warnings:
            print(f"  - {warning}")
    else:
        print("  (none)")

    if user_guidance:
        _print_section("User guidance")
        for tip in user_guidance:
            print(f"  - {tip}")

    output_name = result.output_path.name if result.output_path else "unknown"
    output_full = (
        result.output_path.resolve() if result.output_path else "unknown"
    )

    _print_section("Output")
    print(f"JSON filename:          {output_name}")
    print(f"Saved to:               {output_full}")

    if result.output_path is not None:
        ok, err = _verify_saved_output_json(result.output_path)
        if not ok:
            _print_section("Output JSON smoke check failed")
            print(f"  {err}")
            return 1
        print()
        print(
            "Output JSON smoke check: OK "
            f"({', '.join(_REQUIRED_OUTPUT_KEYS)} present)"
        )

    print()
    print("Done.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract pose landmarks from a local video using the same logic "
            "as POST /extract-pose."
        ),
    )
    parser.add_argument(
        "video",
        type=Path,
        help="Path to a local dance video (.mp4, .mov, .m4v, .avi, .mkv)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Optional output JSON basename (saved under backend/outputs/)",
    )
    args = parser.parse_args()

    video_path = args.video.expanduser().resolve()
    if not video_path.is_file():
        print(f"Error: video file not found: {video_path}", file=sys.stderr)
        return 1

    output_basename = args.output_name
    if output_basename is None:
        output_basename = f"{uuid.uuid4().hex}_poses.json"
    elif not output_basename.endswith(".json"):
        output_basename = f"{output_basename}.json"

    return _print_result(video_path, output_basename)


if __name__ == "__main__":
    raise SystemExit(main())
