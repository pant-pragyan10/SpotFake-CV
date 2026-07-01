"""Sharpness statistics for recapture forensics.

Sharp natural photographs contain image formation blur from optics and sensor sampling,
but recaptured displays often introduce different blur structure: display emissive pixels,
camera focus on a screen plane, and resampling can preserve or suppress high-frequency
detail in distinctive ways. These features summarize edge sharpness and blur strength.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from ..feature_extraction import BaseFeatureExtractor
from ..registry import register_feature_extractor
from ._common import gradient_magnitude, grayscale_image, variance_of_laplacian


@register_feature_extractor
class SharpnessExtractor(BaseFeatureExtractor):
    """Measure blur and edge acutance using Laplacian and gradient statistics."""

    def name(self) -> str:
        return "sharpness"

    def feature_names(self) -> tuple[str, ...]:
        return (
            "sharpness_laplacian_variance",
            "sharpness_gradient_mean",
            "sharpness_gradient_std",
            "sharpness_edge_sharpness_ratio",
            "sharpness_blur_estimate",
        )

    def extract(self, image: Image.Image) -> tuple[float, ...]:
        gray = grayscale_image(image)
        grad = gradient_magnitude(gray)
        laplacian_variance = variance_of_laplacian(gray)

        gradient_mean = float(np.mean(grad))
        gradient_std = float(np.std(grad))
        strong_edge_threshold = float(np.percentile(grad, 90.0))
        strong_edge_ratio = float(np.mean(grad >= strong_edge_threshold))
        edge_sharpness_ratio = float(np.mean(grad[grad >= strong_edge_threshold]) / (gradient_mean + 1e-12))
        blur_estimate = float(1.0 / (1.0 + laplacian_variance))

        return (
            float(laplacian_variance),
            gradient_mean,
            gradient_std,
            edge_sharpness_ratio,
            blur_estimate,
        )

    def debug_visualization(self, image: Image.Image) -> dict[str, np.ndarray]:
        gray = grayscale_image(image)
        grad = gradient_magnitude(gray)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        return {"grayscale": gray, "gradient_magnitude": grad, "laplacian": laplacian}