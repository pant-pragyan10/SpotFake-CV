from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "saved_models"


@dataclass(frozen=True)
class PreprocessingConfig:
    enabled: bool = True
    resize_enabled: bool = True
    resize_width: int = 256
    resize_height: int = 256


@dataclass(frozen=True)
class FeatureConfig:
    enabled_feature_extractors: tuple[str, ...] = ()


@dataclass(frozen=True)
class LoggingConfig:
    enabled: bool = True
    level: str = "INFO"


@dataclass(frozen=True)
class TimingConfig:
    enabled: bool = True
    record_memory: bool = True


@dataclass(frozen=True)
class ModelConfig:
    model_path: Path = MODELS_DIR / "trained_model.pkl"
    scaler_path: Path = MODELS_DIR / "feature_scaler.pkl"
    selected_features_path: Path = MODELS_DIR / "selected_features.json"


@dataclass(frozen=True)
class PipelineConfig:
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


@dataclass(frozen=True)
class ExperimentConfig:
    image_size: tuple[int, int] = (256, 256)
    random_state: int = 42
    test_size: float = 0.2
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)


@dataclass(frozen=True)
class LabConfig:
    run_statistics: bool = True
    run_visualizations: bool = True
    run_correlation: bool = True
    run_importance: bool = True
    run_ablation: bool = True
    run_latency: bool = True
    run_reports: bool = True
    enable_fft_only: bool = False
    disable_glare: bool = False
    disable_color: bool = False
    disable_expensive_detectors: bool = False
    disable_texture: bool = False
    latency_only: bool = False
    visualization_only: bool = False
    statistics_only: bool = False
    expensive_detectors: tuple[str, ...] = ("fft", "moire", "pixel_grid", "brightness_periodicity")
