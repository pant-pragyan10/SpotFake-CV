from __future__ import annotations

from abc import ABC, abstractmethod
from inspect import getdoc
from typing import Sequence

import numpy as np
from PIL import Image

from .results import FeatureMetadata


class BaseFeatureExtractor(ABC):
    """Abstract base class for physically grounded forensic feature extractors."""

    @abstractmethod
    def name(self) -> str:
        """Return a unique, stable name for this extractor."""

    @abstractmethod
    def extract(self, image: Image.Image) -> tuple[float, ...]:
        """Convert one RGB image into a fixed-length numeric feature vector."""

    @abstractmethod
    def feature_names(self) -> tuple[str, ...]:
        """Return the ordered feature names emitted by :meth:`extract`."""

    def feature_metadata(self) -> tuple[FeatureMetadata, ...]:
        description = getdoc(self.__class__) or ""
        return tuple(
            FeatureMetadata(
                extractor_name=self.name(),
                feature_name=feature_name,
                description=description,
            )
            for feature_name in self.feature_names()
        )

    def validate_output(self, values: Sequence[float]) -> tuple[float, ...]:
        return tuple(float(value) for value in values)

    def debug_visualization(self, image: Image.Image) -> dict[str, np.ndarray]:
        """Return optional intermediate arrays for offline debugging and analysis."""

        return {}


FeatureExtractor = BaseFeatureExtractor