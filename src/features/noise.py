"""Noise statistics for recapture forensics.

Camera sensor noise has a spatially local and often somewhat consistent signature.
When an image is photographed from a display, the texture can be altered by display
subpixel structure, resampling, compression, and the screen's own emission profile.
These features summarize how much residual high-frequency content remains and how
consistent that residual is across the image.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from ..feature_extraction import BaseFeatureExtractor
from ..registry import register_feature_extractor
from ._common import grayscale_image, patch_statistics


@register_feature_extractor
class NoiseExtractor(BaseFeatureExtractor):
    """Estimate residual noise energy and its spatial consistency."""

    def name(self) -> str:
        return "noise"

    def feature_names(self) -> tuple[str, ...]:
        return (
            "noise_local_variance_mean",
            "noise_local_variance_std",
            "noise_high_frequency_residual_ratio",
            "noise_estimated_sensor_noise",
            "noise_consistency",
            "noise_residual_kurtosis",
        )

    def extract(self, image: Image.Image) -> tuple[float, ...]:
        gray = grayscale_image(image)
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.5)
        residual = gray - blurred

        patch_means, patch_vars = patch_statistics(residual, patch_size=16)
        local_variance_mean = float(np.mean(patch_vars))
        local_variance_std = float(np.std(patch_vars))

        residual_energy = float(np.sum(residual ** 2))
        total_energy = float(np.sum(gray ** 2) + 1e-12)
        high_frequency_residual_ratio = residual_energy / total_energy

        estimated_sensor_noise = float(np.sqrt(max(local_variance_mean, 0.0)))
        noise_consistency = float(1.0 / (1.0 + (local_variance_std / (local_variance_mean + 1e-12))))

        residual_centered = residual.ravel() - residual.mean()
        residual_std = residual_centered.std() + 1e-12
        kurtosis = float(np.mean((residual_centered / residual_std) ** 4))

        return (
            local_variance_mean,
            local_variance_std,
            high_frequency_residual_ratio,
            estimated_sensor_noise,
            noise_consistency,
            kurtosis,
        )

    def debug_visualization(self, image: Image.Image) -> dict[str, np.ndarray]:
        gray = grayscale_image(image)
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.5)
        residual = gray - blurred
        return {"grayscale": gray, "blurred": blurred, "residual": residual}
