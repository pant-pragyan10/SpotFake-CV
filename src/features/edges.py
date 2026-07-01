"""Edge statistics for recapture forensics.

Display recapture often introduces unnaturally straight, repeated horizontal and
vertical structure from bezels, pixel rows, scanlines, and panel borders. This detector
summarizes how dense the edge field is, which orientations dominate, and how continuous
the strongest edges appear.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from ..feature_extraction import BaseFeatureExtractor
from ..registry import register_feature_extractor
from ._common import gradient_magnitude, gradient_orientation, grayscale_image, orientation_histogram


def _longest_true_run(values: np.ndarray) -> int:
    boolean_values = np.asarray(values, dtype=bool).astype(np.int8)
    if boolean_values.size == 0 or not boolean_values.any():
        return 0
    padded = np.concatenate(([0], boolean_values, [0]))
    transitions = np.flatnonzero(padded[1:] != padded[:-1])
    run_lengths = transitions[1::2] - transitions[::2]
    return int(run_lengths.max()) if run_lengths.size else 0


@register_feature_extractor
class EdgesExtractor(BaseFeatureExtractor):
    """Measure orientation structure and continuity in the edge map."""

    def name(self) -> str:
        return "edges"

    def feature_names(self) -> tuple[str, ...]:
        return (
            "edge_density",
            "dominant_edge_orientation_1",
            "dominant_edge_orientation_2",
            "horizontal_edge_ratio",
            "vertical_edge_ratio",
            "edge_continuity",
            "edge_orientation_entropy",
        )

    def extract(self, image: Image.Image) -> tuple[float, ...]:
        gray = grayscale_image(image)
        gray_uint8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
        edges = cv2.Canny(gray_uint8, threshold1=50, threshold2=150).astype(np.float32) / 255.0
        edge_density = float(edges.mean())

        orientation = gradient_orientation(gray)
        magnitude = gradient_magnitude(gray)
        edge_pixels = edges > 0
        if edge_pixels.any():
            histogram = orientation_histogram(orientation[edge_pixels], magnitude[edge_pixels], bins=18)
            histogram = histogram / (histogram.sum() + 1e-12)
            entropy = float(-np.sum(histogram * np.log2(histogram + 1e-12)))

            dominant_bins = np.argsort(histogram)[::-1][:2]
            bin_width = 180.0 / len(histogram)
            dominant_orientations = ((dominant_bins + 0.5) * bin_width) / 180.0

            edge_angles = np.mod(np.degrees(orientation[edge_pixels]), 180.0)
            horizontal_edge_ratio = float(np.mean((edge_angles >= 80.0) & (edge_angles <= 100.0)))
            vertical_edge_ratio = float(np.mean((edge_angles <= 10.0) | (edge_angles >= 170.0)))
        else:
            histogram = np.zeros(18, dtype=np.float32)
            entropy = 0.0
            dominant_orientations = np.zeros(2, dtype=np.float32)
            horizontal_edge_ratio = 0.0
            vertical_edge_ratio = 0.0

        if edges.any():
            row_runs = []
            col_runs = []
            for row in edges:
                row_runs.append(float(_longest_true_run(row)))
            for column in edges.T:
                col_runs.append(float(_longest_true_run(column)))
            continuity = float((np.mean(row_runs) + np.mean(col_runs)) / (2.0 * max(edges.shape))) if row_runs and col_runs else 0.0
        else:
            continuity = 0.0

        return (
            edge_density,
            float(dominant_orientations[0]),
            float(dominant_orientations[1]),
            horizontal_edge_ratio,
            vertical_edge_ratio,
            continuity,
            entropy,
        )

    def debug_visualization(self, image: Image.Image) -> dict[str, np.ndarray]:
        gray = grayscale_image(image)
        gray_uint8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
        edges = cv2.Canny(gray_uint8, threshold1=50, threshold2=150).astype(np.float32) / 255.0
        return {"grayscale": gray, "canny_edges": edges}
