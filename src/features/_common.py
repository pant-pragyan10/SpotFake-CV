from __future__ import annotations

from dataclasses import dataclass
from math import log2

import cv2
import numpy as np
from scipy import ndimage


EPSILON = 1e-12


def rgb_image_to_array(image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def grayscale_image(image) -> np.ndarray:
    rgb = rgb_image_to_array(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)


def hsv_image(image) -> np.ndarray:
    rgb = np.clip(rgb_image_to_array(image) * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)


def luminance_from_rgb(rgb: np.ndarray) -> np.ndarray:
    return np.dot(rgb[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32))


def safe_entropy(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[values > 0]
    if values.size == 0:
        return 0.0
    probabilities = values / values.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def histogram_entropy(values: np.ndarray, bins: int = 32, value_range: tuple[float, float] | None = None) -> float:
    histogram, _ = np.histogram(values, bins=bins, range=value_range, density=False)
    return safe_entropy(histogram.astype(np.float64))


def gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(sobel_x, sobel_y)


def gradient_orientation(gray: np.ndarray) -> np.ndarray:
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.arctan2(sobel_y, sobel_x)


def variance_of_laplacian(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def autocorrelation_1d(signal: np.ndarray) -> np.ndarray:
    centered = np.asarray(signal, dtype=np.float64) - np.mean(signal)
    correlation = np.correlate(centered, centered, mode="full")
    return correlation[correlation.size // 2 :]


def top_k_values(values: np.ndarray, k: int = 5, ignore_center: int = 0) -> np.ndarray:
    flattened = np.asarray(values, dtype=np.float64).ravel()
    if ignore_center > 0 and flattened.size > ignore_center:
        flattened = flattened[ignore_center:]
    if flattened.size == 0:
        return np.zeros(k, dtype=np.float64)
    sorted_values = np.sort(flattened)[::-1]
    if sorted_values.size < k:
        sorted_values = np.pad(sorted_values, (0, k - sorted_values.size))
    return sorted_values[:k]


def radial_profile(spectrum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = spectrum.shape
    center_y, center_x = height // 2, width // 2
    y_indices, x_indices = np.indices((height, width))
    radii = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)
    radii_int = radii.astype(np.int32)
    max_radius = radii_int.max() + 1
    radial_energy = np.bincount(radii_int.ravel(), weights=spectrum.ravel(), minlength=max_radius)
    radial_count = np.bincount(radii_int.ravel(), minlength=max_radius)
    radial_mean = radial_energy / np.maximum(radial_count, 1)
    return radial_mean.astype(np.float32), radial_count.astype(np.float32)


def fft_magnitude(gray: np.ndarray) -> np.ndarray:
    window = np.outer(np.hanning(gray.shape[0]), np.hanning(gray.shape[1]))
    fft = np.fft.fftshift(np.fft.fft2(gray * window))
    magnitude = np.log1p(np.abs(fft))
    return magnitude.astype(np.float32)


def normalize_map(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    min_value = float(values.min())
    max_value = float(values.max())
    if abs(max_value - min_value) < EPSILON:
        return np.zeros_like(values)
    return (values - min_value) / (max_value - min_value)


def patch_statistics(image: np.ndarray, patch_size: int = 16) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    patch_means = []
    patch_vars = []
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            patch = image[y : y + patch_size, x : x + patch_size]
            if patch.size == 0:
                continue
            patch_means.append(float(np.mean(patch)))
            patch_vars.append(float(np.var(patch)))
    if not patch_means:
        return np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32)
    return np.asarray(patch_means, dtype=np.float32), np.asarray(patch_vars, dtype=np.float32)


def connected_component_statistics(mask: np.ndarray) -> tuple[int, float, float]:
    labels, count = ndimage.label(mask.astype(bool))
    if count == 0:
        return 0, 0.0, 0.0
    sizes = ndimage.sum(mask.astype(np.float32), labels, index=range(1, count + 1))
    if np.isscalar(sizes):
        sizes = np.array([float(sizes)], dtype=np.float32)
    sizes = np.asarray(sizes, dtype=np.float32)
    return int(count), float(sizes.max(initial=0.0)), float(sizes.mean() if sizes.size else 0.0)


def orientation_histogram(orientation: np.ndarray, magnitude: np.ndarray, bins: int = 18) -> np.ndarray:
    angles = np.mod(np.degrees(orientation), 180.0)
    histogram, _ = np.histogram(angles.ravel(), bins=bins, range=(0.0, 180.0), weights=magnitude.ravel())
    return histogram.astype(np.float32)


def line_profile(image: np.ndarray, axis: int) -> np.ndarray:
    return image.mean(axis=axis)


def detrend_1d(signal: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    kernel_size = max(3, int(kernel_size) | 1)
    trend = ndimage.median_filter(signal.astype(np.float32), size=kernel_size)
    return signal.astype(np.float32) - trend