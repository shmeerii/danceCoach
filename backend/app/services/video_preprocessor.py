"""Frame preprocessing for reliable pose detection across varied capture conditions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Aligned with video_validator luminance heuristics.
DARK_MEAN_LUMA = 45.0
BRIGHT_MEAN_LUMA = 210.0
LOW_CONTRAST_STD = 18.0

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


@dataclass
class PreprocessResult:
    """Paths and notes from the video-level preprocessing step."""

    processed_path: Path
    warnings: list[str] = field(default_factory=list)


def _bgr_to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _luminance_stats(frame_bgr: np.ndarray) -> tuple[float, float]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray)), float(np.std(gray))


def estimate_frame_quality(frame_bgr: np.ndarray) -> dict[str, float | bool]:
    """
    Estimate basic frame quality from luminance statistics.

    Used to pick preprocessing variants and to annotate extraction quality later.
    """
    brightness, contrast = _luminance_stats(frame_bgr)
    return {
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "is_too_dark": brightness < DARK_MEAN_LUMA,
        "is_too_bright": brightness > BRIGHT_MEAN_LUMA,
        "is_low_contrast": contrast < LOW_CONTRAST_STD,
    }


def _apply_gamma(frame_bgr: np.ndarray, gamma: float) -> np.ndarray:
    gamma = max(gamma, 1e-6)
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((index / 255.0) ** inv_gamma) * 255 for index in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(frame_bgr, table)


def _brightness_contrast_corrected(frame_bgr: np.ndarray) -> np.ndarray:
    """Helps harsh light, washed-out highlights, and dull indoor lighting."""
    mean_luma, _ = _luminance_stats(frame_bgr)
    alpha = float(np.clip(1.0 + (128.0 - mean_luma) / 512.0, 0.88, 1.12))
    beta = float(np.clip((128.0 - mean_luma) * 0.15, -18.0, 18.0))
    return cv2.convertScaleAbs(frame_bgr, alpha=alpha, beta=beta)


def _gamma_corrected_low_light(frame_bgr: np.ndarray) -> np.ndarray:
    """Helps low light, dark studios, and mildly underexposed footage."""
    mean_luma, _ = _luminance_stats(frame_bgr)
    if mean_luma >= 140.0:
        gamma = 1.0
    elif mean_luma >= 90.0:
        gamma = 0.97
    else:
        gamma = float(np.clip(0.82 + mean_luma / 500.0, 0.85, 0.95))
    return _apply_gamma(frame_bgr, gamma)


def _clahe_luminance(frame_bgr: np.ndarray) -> np.ndarray:
    """Helps low contrast, flat indoor lighting, and uneven local shadows."""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    lightness, green_red, blue_yellow = cv2.split(lab)
    lightness = _CLAHE.apply(lightness)
    merged = cv2.merge([lightness, green_red, blue_yellow])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _mild_sharpened(frame_bgr: np.ndarray) -> np.ndarray:
    """Helps motion blur, soft focus, and phone captures with sharpening disabled."""
    blurred = cv2.GaussianBlur(frame_bgr, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(frame_bgr, 1.15, blurred, -0.15, 0)


def generate_frame_variants(frame_bgr: np.ndarray) -> list[dict[str, Any]]:
    """
    Build conservative RGB frame variants for MediaPipe at the original resolution.

    All variants keep the same width and height as the input BGR frame so landmark
    coordinates map to the original video pixels.

    Variant order and typical use:
    1. original_rgb — baseline when lighting and contrast are already adequate.
    2. brightness_contrast_corrected — harsh light, washed-out scenes, dull indoor light.
    3. gamma_corrected_low_light — low light and underexposed captures.
    4. clahe_luminance — low contrast and uneven lighting across the frame.
    5. mild_sharpened — motion blur or slight softness.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("frame_bgr is empty")
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must be a 3-channel BGR image")

    height, width = frame_bgr.shape[:2]

    variant_builders: list[tuple[str, np.ndarray]] = [
        ("original_rgb", frame_bgr.copy()),
        ("brightness_contrast_corrected", _brightness_contrast_corrected(frame_bgr)),
        ("gamma_corrected_low_light", _gamma_corrected_low_light(frame_bgr)),
        ("clahe_luminance", _clahe_luminance(frame_bgr)),
        ("mild_sharpened", _mild_sharpened(frame_bgr)),
    ]

    result: list[dict[str, Any]] = []
    for variant_name, bgr in variant_builders:
        if bgr.shape[0] != height or bgr.shape[1] != width:
            raise RuntimeError(
                f"Variant '{variant_name}' changed frame size; landmarks would not align."
            )
        result.append(
            {
                "variant_name": variant_name,
                "frame_rgb": _bgr_to_rgb(bgr),
            }
        )
    return result


def preprocess_video(video_path: Path) -> PreprocessResult:
    """
    Video-level preprocessing hook (orientation, container fixes).

    Per-frame lighting variants are produced by generate_frame_variants() during
    pose extraction. The uploaded file path is returned unchanged so coordinates
    stay tied to the original pixels.
    """
    return PreprocessResult(processed_path=video_path, warnings=[])
