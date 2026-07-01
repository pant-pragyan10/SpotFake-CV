"""Brightness periodicity detector for recapture forensics.

Screens often reveal scanline behavior, refresh-related banding, or periodic brightness
modulation. This detector focuses on 1D oscillations in row and column luminance
profiles and measures how strong that periodic variation is.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from ..feature_extraction import BaseFeatureExtractor
from ..registry import register_feature_extractor
from ._common import detrend_1d, grayscale_image, line_profile


def _periodic_strength(signal: np.ndarray) -> tuple[float, float]:
    if signal.size < 4:
        return 0.0, 0.0
    centered = signal - signal.mean()
    spectrum = np.abs(np.fft.rfft(centered))
    spectrum[0] = 0.0
    dominant_index = int(np.argmax(spectrum[1:]) + 1) if spectrum.size > 1 else 0
    dominant_strength = float(spectrum[dominant_index] / (spectrum[1:].sum() + 1e-12)) if dominant_index > 0 else 0.0
    if dominant_index <= 0:
        return 0.0, dominant_strength
    period_estimate = float(signal.size / dominant_index)
    return period_estimate, dominant_strength


@register_feature_extractor
class BrightnessPeriodicityExtractor(BaseFeatureExtractor):
    """Measure scanline-like oscillation in row and column brightness profiles."""

    def name(self) -> str:
        return "brightness_periodicity"

    def feature_names(self) -> tuple[str, ...]:
        return (
            "brightness_row_period",
            "brightness_column_period",
            "brightness_row_periodicity",
            "brightness_column_periodicity",
            "brightness_scanline_strength",
            "brightness_intensity_variation",
        )

    def extract(self, image: Image.Image) -> tuple[float, ...]:
        gray = grayscale_image(image)
        row_profile = detrend_1d(line_profile(gray, axis=1), kernel_size=15)
        column_profile = detrend_1d(line_profile(gray, axis=0), kernel_size=15)

        row_period, row_periodicity = _periodic_strength(row_profile)
        column_period, column_periodicity = _periodic_strength(column_profile)
        scanline_strength = float(max(row_periodicity, column_periodicity))

        intensity_variation = float((np.std(row_profile) + np.std(column_profile)) / (gray.mean() + 1e-12))

        return (
            row_period,
            column_period,
            row_periodicity,
            column_periodicity,
            scanline_strength,
            intensity_variation,
        )

    def debug_visualization(self, image: Image.Image) -> dict[str, np.ndarray]:
        gray = grayscale_image(image)
        row_profile = detrend_1d(line_profile(gray, axis=1), kernel_size=15)
        column_profile = detrend_1d(line_profile(gray, axis=0), kernel_size=15)
        return {"grayscale": gray, "row_profile": row_profile, "column_profile": column_profile}