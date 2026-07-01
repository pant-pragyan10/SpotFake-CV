"""Local texture statistics for recapture forensics.

Display recapture can alter fine-scale texture through resampling, subpixel structure,
and repeated sampling of the same physical surface. This detector summarizes local
entropy, patch variance, and the overall complexity of texture variation.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from ..feature_extraction import BaseFeatureExtractor
from ..registry import register_feature_extractor
from ._common import gradient_magnitude, grayscale_image, histogram_entropy, patch_statistics


@register_feature_extractor
class TextureExtractor(BaseFeatureExtractor):
    """Measure patch entropy, local variance, and texture complexity."""

    def name(self) -> str:
        return "texture"

    def feature_names(self) -> tuple[str, ...]:
        return (
            "texture_local_entropy_mean",
            "texture_local_entropy_std",
            "texture_uniformity",
            "texture_local_variance_mean",
            "texture_local_variance_std",
            "texture_complexity",
        )

    def extract(self, image: Image.Image) -> tuple[float, ...]:
        gray = grayscale_image(image)
        patch_means, patch_vars = patch_statistics(gray, patch_size=16)

        entropy_values = []
        height, width = gray.shape
        patch_size = 16
        for y in range(0, height, patch_size):
            for x in range(0, width, patch_size):
                patch = gray[y : y + patch_size, x : x + patch_size]
                if patch.size == 0:
                    continue
                entropy_values.append(histogram_entropy(patch, bins=16, value_range=(0.0, 1.0)))

        entropy_values = np.asarray(entropy_values, dtype=np.float32) if entropy_values else np.zeros(1, dtype=np.float32)
        local_entropy_mean = float(entropy_values.mean())
        local_entropy_std = float(entropy_values.std())
        texture_uniformity = float(1.0 - local_entropy_mean / np.log2(16.0))
        local_variance_mean = float(patch_vars.mean())
        local_variance_std = float(patch_vars.std())

        grad = gradient_magnitude(gray)
        texture_complexity = float((grad.mean() + grad.std()) / (gray.std() + 1e-12))

        return (
            local_entropy_mean,
            local_entropy_std,
            texture_uniformity,
            local_variance_mean,
            local_variance_std,
            texture_complexity,
        )

    def debug_visualization(self, image: Image.Image) -> dict[str, np.ndarray]:
        gray = grayscale_image(image)
        grad = gradient_magnitude(gray)
        return {"grayscale": gray, "gradient_magnitude": grad}