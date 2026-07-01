from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from .timing import TimingResult


@dataclass(frozen=True)
class ImageInfo:
    path: Path
    width: int
    height: int
    mode: str


@dataclass(frozen=True)
class FeatureMetadata:
    extractor_name: str
    feature_name: str
    description: str = ""
    value_type: str = "float"


@dataclass(frozen=True)
class PreprocessingResult:
    original_size: tuple[int, int]
    processed_size: tuple[int, int]
    applied_steps: tuple[str, ...] = ()
    timing: TimingResult = field(default_factory=lambda: TimingResult(0.0))


@dataclass(frozen=True)
class FeatureResult:
    extractor_name: str
    feature_names: tuple[str, ...]
    feature_values: tuple[float, ...]
    feature_metadata: tuple[FeatureMetadata, ...]
    timing: TimingResult
    error: str | None = None


@dataclass(frozen=True)
class FeatureVectorResult:
    feature_vector: tuple[float, ...]
    feature_names: tuple[str, ...]
    feature_metadata: tuple[FeatureMetadata, ...]
    timing: TimingResult
    extractor_results: tuple[FeatureResult, ...] = ()
    extractor_errors: tuple[FeatureResult, ...] = ()


@dataclass(frozen=True)
class PipelineResult:
    image_info: ImageInfo
    preprocessing: PreprocessingResult
    feature_vector: FeatureVectorResult
    probability: float
    timing: TimingResult


@dataclass(frozen=True)
class ExperimentResult:
    name: str
    pipeline_result: PipelineResult
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: tuple[Path, ...] = ()