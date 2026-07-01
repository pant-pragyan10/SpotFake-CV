"""Local contrast statistics for recapture forensics.

Display recapture can create contrast inconsistency because emitted light, resampling,
and camera exposure interact differently from reflections on a real scene. This detector
summarizes global and local contrast strength as well as how concentrated the contrast
energy is across the image.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from ..feature_extraction import BaseFeatureExtractor
from ..registry import register_feature_extractor
from ._common import grayscale_image, patch_statistics


@register_feature_extractor
class ContrastExtractor(BaseFeatureExtractor):
    """Measure global and local luminance contrast structure."""

    def name(self) -> str:
        return "contrast"

    def feature_names(self) -> tuple[str, ...]:
        return (
            "contrast_rms",
            "contrast_michelson",
            "contrast_local_mean",
            "contrast_local_std",
            "contrast_uniformity",
            "contrast_balance",
        )

    def extract(self, image: Image.Image) -> tuple[float, ...]:
        gray = grayscale_image(image)
        rms_contrast = float(np.std(gray) / (np.mean(gray) + 1e-12))
        michelson = float((gray.max() - gray.min()) / (gray.max() + gray.min() + 1e-12))
        _, patch_variances = patch_statistics(gray, patch_size=16)
        local_contrast_mean = float(np.mean(np.sqrt(patch_variances)))
        local_contrast_std = float(np.std(np.sqrt(patch_variances)))
        uniformity = float(1.0 / (1.0 + np.std(patch_variances) / (np.mean(patch_variances) + 1e-12)))
        dark_ratio = float(np.mean(gray < np.percentile(gray, 25.0)))
        bright_ratio = float(np.mean(gray > np.percentile(gray, 75.0)))
        balance = float(1.0 - abs(dark_ratio - bright_ratio))
        return (
            rms_contrast,
            michelson,
            local_contrast_mean,
            local_contrast_std,
            uniformity,
            balance,
        )

    def debug_visualization(self, image: Image.Image) -> dict[str, np.ndarray]:
        return {"grayscale": grayscale_image(image)}
