"""Production inference: image -> preprocessing -> features -> saved model -> probability."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

from configs.config import ModelConfig, PipelineConfig

from ..features.feature_vector import FeatureVectorBuilder
from ..preprocessing.loader import load_image
from ..preprocessing.normalize import normalize_image
from ..preprocessing.resize import resize_image


class ScreenRecaptureDetector:
    """Loads the trained model once and scores images with no per-call setup cost."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()

        selection = json.loads(Path(self.config.selected_features_path).read_text())
        self.feature_names: tuple[str, ...] = tuple(selection["feature_names"])
        self.needs_scaling: bool = selection["needs_scaling"]
        self.decision_threshold: float = selection.get("decision_threshold", 0.5)

        self.model = joblib.load(self.config.model_path)
        scaler_path = Path(self.config.scaler_path)
        self.scaler = joblib.load(scaler_path) if scaler_path.exists() else None

        self.pipeline_config = PipelineConfig()
        self.feature_vector_builder = FeatureVectorBuilder(
            enabled_feature_names=(),  # empty tuple -> FeatureDatasetBuilder convention: () means "all"
            record_memory=False,
        )

    def _preprocess(self, image: Image.Image) -> Image.Image:
        if self.pipeline_config.preprocessing.resize_enabled:
            image = resize_image(
                image,
                (self.pipeline_config.preprocessing.resize_width, self.pipeline_config.preprocessing.resize_height),
            )
        return normalize_image(image)

    def predict_proba(self, image_path: str | Path) -> float:
        return self.predict_proba_image(load_image(image_path))

    def predict_proba_image(self, image: Image.Image) -> float:
        image = self._preprocess(image.convert("RGB"))
        feature_result = self.feature_vector_builder.build(image)
        values_by_name = dict(zip(feature_result.feature_names, feature_result.feature_vector))

        missing = [name for name in self.feature_names if name not in values_by_name]
        if missing:
            raise RuntimeError(f"Feature extraction did not produce required features: {missing}")

        vector = np.array([[values_by_name[name] for name in self.feature_names]], dtype=np.float32)
        if self.scaler is not None:
            vector = self.scaler.transform(vector)
        return float(self.model.predict_proba(vector)[0, 1])


_DETECTOR: ScreenRecaptureDetector | None = None


def get_detector() -> ScreenRecaptureDetector:
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = ScreenRecaptureDetector()
    return _DETECTOR


def predict_proba(image_path: str | Path) -> float:
    return get_detector().predict_proba(image_path)


def predict_proba_image(image: Image.Image) -> float:
    return get_detector().predict_proba_image(image)
