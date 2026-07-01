"""Pixel-grid detector for recapture forensics.

Display panels introduce repeating spatial structure from pixels and subpixels. When a
camera re-photographs a display, that grid can survive as a periodic pattern in row and
column luminance profiles. This detector estimates the spacing and consistency of those
periodic oscillations.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from ..feature_extraction import BaseFeatureExtractor
from ..registry import register_feature_extractor
from ._common import autocorrelation_1d, grayscale_image, line_profile, detrend_1d


def _dominant_period(signal: np.ndarray, min_lag: int = 2, max_lag: int | None = None) -> tuple[float, float]:
    if signal.size <= min_lag + 1:
        return 0.0, 0.0
    autocorr = autocorrelation_1d(signal)
    autocorr = autocorr / (autocorr[0] + 1e-12)
    max_lag = min(max_lag or signal.size // 2, autocorr.size - 1)
    search = autocorr[min_lag : max_lag + 1]
    if search.size == 0:
        return 0.0, 0.0
    dominant_index = int(np.argmax(search)) + min_lag
    dominant_strength = float(search.max())
    return float(dominant_index), dominant_strength


@register_feature_extractor
class PixelGridExtractor(BaseFeatureExtractor):
    """Estimate repeating horizontal and vertical screen-grid spacing."""

    def name(self) -> str:
        return "pixel_grid"

    def feature_names(self) -> tuple[str, ...]:
        return (
            "pixel_grid_horizontal_spacing",
            "pixel_grid_vertical_spacing",
            "pixel_grid_horizontal_periodicity",
            "pixel_grid_vertical_periodicity",
            "pixel_grid_stripe_consistency",
            "pixel_grid_lattice_energy_ratio",
        )

    def extract(self, image: Image.Image) -> tuple[float, ...]:
        gray = grayscale_image(image)
        row_profile = detrend_1d(line_profile(gray, axis=1), kernel_size=11)
        column_profile = detrend_1d(line_profile(gray, axis=0), kernel_size=11)

        horizontal_spacing, horizontal_periodicity = _dominant_period(row_profile)
        vertical_spacing, vertical_periodicity = _dominant_period(column_profile)

        row_spectrum = np.abs(np.fft.rfft(row_profile))
        column_spectrum = np.abs(np.fft.rfft(column_profile))
        lattice_energy_ratio = float((row_spectrum[1:].max(initial=0.0) + column_spectrum[1:].max(initial=0.0)) / ((row_spectrum[1:].mean() + column_spectrum[1:].mean()) + 1e-12))

        stripe_consistency = float((horizontal_periodicity + vertical_periodicity) / 2.0)

        return (
            horizontal_spacing,
            vertical_spacing,
            horizontal_periodicity,
            vertical_periodicity,
            stripe_consistency,
            lattice_energy_ratio,
        )

    def debug_visualization(self, image: Image.Image) -> dict[str, np.ndarray]:
        gray = grayscale_image(image)
        row_profile = detrend_1d(line_profile(gray, axis=1), kernel_size=11)
        column_profile = detrend_1d(line_profile(gray, axis=0), kernel_size=11)
        return {"grayscale": gray, "row_profile": row_profile, "column_profile": column_profile}