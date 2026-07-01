"""Frequency-domain forensic features.

This detector measures spectral structure in the image. A real camera photo tends to
contain broad natural-image spectra, while a recaptured display can inject periodic
energy from pixel lattices, scanlines, subpixel layouts, and moiré interference.
The extractor summarizes where the strongest spectral peaks occur and how energy is
distributed radially across low, mid, and high frequencies.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from ..feature_extraction import BaseFeatureExtractor
from ..registry import register_feature_extractor
from ._common import fft_magnitude, grayscale_image, radial_profile, safe_entropy, top_k_values


@register_feature_extractor
class FFTExtractor(BaseFeatureExtractor):
    """Measure periodic structure and high-frequency concentration in the 2D FFT."""

    def name(self) -> str:
        return "fft"

    def feature_names(self) -> tuple[str, ...]:
        return (
            "fft_peak_1",
            "fft_peak_2",
            "fft_peak_3",
            "fft_peak_radius_1",
            "fft_peak_radius_2",
            "fft_peak_radius_3",
            "fft_low_frequency_ratio",
            "fft_mid_frequency_ratio",
            "fft_high_frequency_ratio",
            "fft_frequency_entropy",
            "fft_periodicity_score",
        )

    def extract(self, image: Image.Image) -> tuple[float, ...]:
        gray = grayscale_image(image)
        spectrum = fft_magnitude(gray)
        radial_energy, _ = radial_profile(spectrum)

        total_energy = float(radial_energy.sum() + 1e-12)
        bins = len(radial_energy)
        low_end = max(1, int(bins * 0.15))
        mid_end = max(low_end + 1, int(bins * 0.55))

        low_ratio = float(radial_energy[:low_end].sum() / total_energy)
        mid_ratio = float(radial_energy[low_end:mid_end].sum() / total_energy)
        high_ratio = float(radial_energy[mid_end:].sum() / total_energy)

        normalized_profile = radial_energy / (total_energy + 1e-12)
        entropy = safe_entropy(normalized_profile)

        center_y, center_x = np.array(spectrum.shape) // 2
        y_indices, x_indices = np.indices(spectrum.shape)
        radii = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)
        valid_mask = radii > max(2, min(spectrum.shape) * 0.04)
        flattened_spectrum = spectrum[valid_mask]
        flattened_radii = radii[valid_mask]
        top_peaks = top_k_values(flattened_spectrum, k=3)

        peak_indices = np.argsort(flattened_spectrum)[::-1][:3]
        peak_radii = flattened_radii[peak_indices] if peak_indices.size else np.zeros(3, dtype=np.float32)
        if peak_radii.size < 3:
            peak_radii = np.pad(peak_radii, (0, 3 - peak_radii.size))
        max_radius = float(radii.max() + 1e-12)
        peak_radii = peak_radii / max_radius

        median_energy = float(np.median(flattened_spectrum) + 1e-12)
        periodicity_score = float(top_peaks[0] / median_energy)

        return (
            float(top_peaks[0]),
            float(top_peaks[1]),
            float(top_peaks[2]),
            float(peak_radii[0]),
            float(peak_radii[1]),
            float(peak_radii[2]),
            low_ratio,
            mid_ratio,
            high_ratio,
            float(entropy),
            periodicity_score,
        )

    def debug_visualization(self, image: Image.Image) -> dict[str, np.ndarray]:
        gray = grayscale_image(image)
        spectrum = fft_magnitude(gray)
        return {"grayscale": gray, "fft_spectrum": spectrum}
