"""Color statistics for recapture forensics.

Screens and cameras differ in color reproduction because displays emit light through
subpixel primaries while cameras observe reflected illumination filtered by a sensor.
This extractor summarizes channel intensities, channel balance, chroma structure, and
luminance behavior without trying to classify objects semantically.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from ..feature_extraction import BaseFeatureExtractor
from ..registry import register_feature_extractor
from ._common import histogram_entropy, hsv_image, luminance_from_rgb, rgb_image_to_array


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).ravel()
    right = np.asarray(right, dtype=np.float64).ravel()
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator < 1e-12:
        return 0.0
    return float(np.dot(left_centered, right_centered) / denominator)


@register_feature_extractor
class ColorsExtractor(BaseFeatureExtractor):
    """Measure RGB, HSV, luminance, correlation, and white-balance statistics."""

    def name(self) -> str:
        return "colors"

    def feature_names(self) -> tuple[str, ...]:
        return (
            "rgb_mean_r",
            "rgb_mean_g",
            "rgb_mean_b",
            "rgb_std_r",
            "rgb_std_g",
            "rgb_std_b",
            "rgb_entropy_r",
            "rgb_entropy_g",
            "rgb_entropy_b",
            "hsv_hue_cos_mean",
            "hsv_hue_sin_mean",
            "hsv_saturation_mean",
            "hsv_saturation_std",
            "hsv_value_mean",
            "hsv_value_std",
            "luminance_mean",
            "luminance_std",
            "channel_corr_rg",
            "channel_corr_rb",
            "channel_corr_gb",
            "white_balance_red_green_ratio",
            "white_balance_blue_green_ratio",
        )

    def extract(self, image: Image.Image) -> tuple[float, ...]:
        rgb = rgb_image_to_array(image)
        channels = [rgb[..., index] for index in range(3)]
        means = [float(channel.mean()) for channel in channels]
        stds = [float(channel.std()) for channel in channels]
        entropies = [float(histogram_entropy(channel, bins=32, value_range=(0.0, 1.0))) for channel in channels]

        hsv = hsv_image(image)
        hue = hsv[..., 0] / 180.0 * 2.0 * np.pi
        saturation = hsv[..., 1]
        value = hsv[..., 2]

        hue_cos_mean = float(np.cos(hue).mean())
        hue_sin_mean = float(np.sin(hue).mean())
        saturation_mean = float(saturation.mean())
        saturation_std = float(saturation.std())
        value_mean = float(value.mean())
        value_std = float(value.std())

        luminance = luminance_from_rgb(rgb)
        luminance_mean = float(luminance.mean())
        luminance_std = float(luminance.std())

        flat_channels = [channel.ravel() for channel in channels]
        corr_rg = _safe_correlation(flat_channels[0], flat_channels[1])
        corr_rb = _safe_correlation(flat_channels[0], flat_channels[2])
        corr_gb = _safe_correlation(flat_channels[1], flat_channels[2])

        white_balance_red_green_ratio = float(means[0] / (means[1] + 1e-12))
        white_balance_blue_green_ratio = float(means[2] / (means[1] + 1e-12))

        return (
            *means,
            *stds,
            *entropies,
            hue_cos_mean,
            hue_sin_mean,
            saturation_mean,
            saturation_std,
            value_mean,
            value_std,
            luminance_mean,
            luminance_std,
            corr_rg,
            corr_rb,
            corr_gb,
            white_balance_red_green_ratio,
            white_balance_blue_green_ratio,
        )

    def debug_visualization(self, image: Image.Image) -> dict[str, np.ndarray]:
        rgb = rgb_image_to_array(image)
        hsv = hsv_image(image)
        luminance = luminance_from_rgb(rgb)
        return {"rgb": rgb, "hsv": hsv, "luminance": luminance}
