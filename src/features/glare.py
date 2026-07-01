"""Glare detector for recapture forensics.

Screens are emissive and commonly produce blooming, hotspots, and saturated glare when
photographed under ambient light. This detector looks for bright contiguous regions,
their expansion after blurring, and how much of the image becomes saturated.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from ..feature_extraction import BaseFeatureExtractor
from ..registry import register_feature_extractor
from ._common import connected_component_statistics, hsv_image, luminance_from_rgb, rgb_image_to_array


@register_feature_extractor
class GlareExtractor(BaseFeatureExtractor):
    """Measure hotspots, blooming, and saturated highlights."""

    def name(self) -> str:
        return "glare"

    def feature_names(self) -> tuple[str, ...]:
        return (
            "glare_hotspot_count",
            "glare_bloom_estimation",
            "glare_saturated_region_ratio",
            "glare_peak_intensity",
            "glare_highlight_mean_intensity",
        )

    def extract(self, image: Image.Image) -> tuple[float, ...]:
        rgb = rgb_image_to_array(image)
        luminance = luminance_from_rgb(rgb)
        hsv = hsv_image(image)
        value = hsv[..., 2] / 255.0
        blurred = np.clip((luminance + value) / 2.0, 0.0, 1.0)

        threshold = float(np.percentile(blurred, 99.0))
        hotspot_mask = blurred >= threshold
        hotspot_count, largest_hotspot, mean_hotspot = connected_component_statistics(hotspot_mask.astype(np.uint8))

        saturated_region_ratio = float(np.mean(value >= 0.98))
        peak_intensity = float(blurred.max())
        highlight_mean_intensity = float(blurred[hotspot_mask].mean() if hotspot_mask.any() else 0.0)

        bloom_mask = blurred >= np.percentile(blurred, 98.5)
        bloom_ratio = float(bloom_mask.mean() - hotspot_mask.mean())

        return (
            float(hotspot_count),
            bloom_ratio,
            saturated_region_ratio,
            peak_intensity,
            highlight_mean_intensity,
        )

    def debug_visualization(self, image: Image.Image) -> dict[str, np.ndarray]:
        rgb = rgb_image_to_array(image)
        luminance = luminance_from_rgb(rgb)
        hsv = hsv_image(image)
        value = hsv[..., 2] / 255.0
        blurred = np.clip((luminance + value) / 2.0, 0.0, 1.0)
        hotspot_mask = blurred >= np.percentile(blurred, 99.0)
        bloom_mask = blurred >= np.percentile(blurred, 98.5)
        return {"luminance": luminance, "hotspot_mask": hotspot_mask.astype(np.float32), "bloom_mask": bloom_mask.astype(np.float32)}
