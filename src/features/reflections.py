"""Reflection detector for recapture forensics.

Real scenes often contain diffuse reflected illumination, but screens can produce sharp
specular highlights and mirror-like patches when the display surface catches ambient
light. This detector measures the area and intensity of those bright reflective regions.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from ..feature_extraction import BaseFeatureExtractor
from ..registry import register_feature_extractor
from ._common import connected_component_statistics, hsv_image, rgb_image_to_array


@register_feature_extractor
class ReflectionExtractor(BaseFeatureExtractor):
    """Measure bright, low-saturation regions consistent with specular reflection."""

    def name(self) -> str:
        return "reflections"

    def feature_names(self) -> tuple[str, ...]:
        return (
            "reflection_specular_ratio",
            "reflection_area_ratio",
            "reflection_intensity_mean",
            "reflection_intensity_peak",
            "reflection_component_count",
        )

    def extract(self, image: Image.Image) -> tuple[float, ...]:
        rgb = rgb_image_to_array(image)
        hsv = hsv_image(image)
        saturation = hsv[..., 1] / 255.0
        value = hsv[..., 2] / 255.0

        specular_mask = (value >= np.percentile(value, 97.5)) & (saturation <= np.percentile(saturation, 35.0))
        reflection_component_count, reflection_largest_area, reflection_mean_area = connected_component_statistics(specular_mask.astype(np.uint8))

        specular_ratio = float(specular_mask.mean())
        reflection_area_ratio = float(reflection_largest_area / (image.width * image.height + 1e-12))
        intensity_mean = float(value[specular_mask].mean() if specular_mask.any() else 0.0)
        intensity_peak = float(value[specular_mask].max() if specular_mask.any() else 0.0)

        return (
            specular_ratio,
            reflection_area_ratio,
            intensity_mean,
            intensity_peak,
            float(reflection_component_count),
        )

    def debug_visualization(self, image: Image.Image) -> dict[str, np.ndarray]:
        rgb = rgb_image_to_array(image)
        hsv = hsv_image(image)
        saturation = hsv[..., 1] / 255.0
        value = hsv[..., 2] / 255.0
        specular_mask = (value >= np.percentile(value, 97.5)) & (saturation <= np.percentile(saturation, 35.0))
        return {"value_channel": value, "specular_mask": specular_mask.astype(np.float32)}
