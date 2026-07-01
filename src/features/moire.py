"""Moire interference detector.

Recaptured displays often create interference between the camera sampling grid and the
display pixel/subpixel lattice. That interference appears as directional periodic
energy, repeated stripes, and aliasing-like peaks in the spectrum.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from ..feature_extraction import BaseFeatureExtractor
from ..registry import register_feature_extractor
from ._common import fft_magnitude, gradient_orientation, grayscale_image, orientation_histogram, safe_entropy


@register_feature_extractor
class MoireExtractor(BaseFeatureExtractor):
    """Estimate the strength of aliasing and periodic stripe interference."""

    def name(self) -> str:
        return "moire"

    def feature_names(self) -> tuple[str, ...]:
        return (
            "moire_periodic_stripe_strength",
            "moire_directional_peak_strength",
            "moire_aliasing_score",
            "moire_interference_entropy",
            "moire_interference_peak_count",
            "moire_high_frequency_energy_ratio",
        )

    def extract(self, image: Image.Image) -> tuple[float, ...]:
        gray = grayscale_image(image)
        spectrum = fft_magnitude(gray)
        magnitude = spectrum.copy()
        height, width = magnitude.shape
        center_y, center_x = height // 2, width // 2
        y_indices, x_indices = np.indices(magnitude.shape)
        radii = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)
        angle = np.arctan2(y_indices - center_y, x_indices - center_x)

        high_frequency_mask = radii >= min(height, width) * 0.2
        high_frequency_energy = float(magnitude[high_frequency_mask].sum())
        total_energy = float(magnitude.sum() + 1e-12)
        high_frequency_ratio = high_frequency_energy / total_energy

        # Moiré often manifests as narrow directional peaks outside the DC region.
        non_center_mask = radii >= min(height, width) * 0.08
        directional_histogram = orientation_histogram(angle[non_center_mask], magnitude[non_center_mask], bins=18)
        directional_histogram = directional_histogram / (directional_histogram.sum() + 1e-12)
        directional_peak_strength = float(directional_histogram.max())
        directional_entropy = safe_entropy(directional_histogram)

        median_energy = float(np.median(magnitude[non_center_mask]) + 1e-12)
        peaks = magnitude[non_center_mask] > (np.percentile(magnitude[non_center_mask], 99.2))
        peak_count = float(peaks.sum())
        stripe_strength = float(np.percentile(magnitude[non_center_mask], 99.5) / median_energy)
        aliasing_score = float((directional_peak_strength * high_frequency_ratio) / (1.0 + directional_entropy))

        return (
            stripe_strength,
            directional_peak_strength,
            aliasing_score,
            directional_entropy,
            peak_count,
            high_frequency_ratio,
        )

    def debug_visualization(self, image: Image.Image) -> dict[str, np.ndarray]:
        gray = grayscale_image(image)
        spectrum = fft_magnitude(gray)
        gradient = np.abs(np.sin(gradient_orientation(gray)))
        return {"grayscale": gray, "moire_spectrum": spectrum, "orientation_response": gradient}
