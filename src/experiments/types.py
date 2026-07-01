from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureDataset:
    dataframe: pd.DataFrame
    feature_names: tuple[str, ...]
    matrix: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float32))
    labels: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int64))
    metadata: pd.DataFrame = field(default_factory=pd.DataFrame)
    latency_records: pd.DataFrame = field(default_factory=pd.DataFrame)
    failures: pd.DataFrame = field(default_factory=pd.DataFrame)
    label_column: str = "label"
    path_column: str = "path"


@dataclass(frozen=True)
class FeatureSummary:
    statistics: pd.DataFrame
    per_class_statistics: pd.DataFrame
    missing_values: pd.DataFrame
    summary_text: str


@dataclass(frozen=True)
class DatasetSummary:
    sample_count: int
    class_balance: dict[int, int]
    feature_dimensions: tuple[int, int]
    extracted_feature_count: int
    average_extraction_time_seconds: float
    failed_images: int
    summary_text: str


@dataclass(frozen=True)
class CorrelationSummary:
    pearson: pd.DataFrame
    spearman: pd.DataFrame
    mutual_information: pd.DataFrame
    highly_correlated_pairs: pd.DataFrame
    report_text: str


@dataclass(frozen=True)
class ImportanceSummary:
    ranking: pd.DataFrame
    report_text: str


@dataclass(frozen=True)
class LatencySummary:
    benchmark: pd.DataFrame
    average_runtime_seconds: float
    slowest_detector: str
    fastest_detector: str


@dataclass(frozen=True)
class ExtractorAuditSummary:
    audit: pd.DataFrame
    report_text: str


@dataclass(frozen=True)
class AblationSummary:
    results: pd.DataFrame
    ablation_groups: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExperimentArtifacts:
    csv_files: tuple[Path, ...] = ()
    figure_files: tuple[Path, ...] = ()
    report_files: tuple[Path, ...] = ()
    benchmark_files: tuple[Path, ...] = ()
    log_files: tuple[Path, ...] = ()


@dataclass(frozen=True)
class ExperimentLabResult:
    dataset: FeatureDataset
    dataset_summary: DatasetSummary | None = None
    feature_summary: FeatureSummary | None = None
    correlation_summary: CorrelationSummary | None = None
    importance_summary: ImportanceSummary | None = None
    latency_summary: LatencySummary | None = None
    extractor_audit_summary: ExtractorAuditSummary | None = None
    ablation_summary: AblationSummary | None = None
    artifacts: ExperimentArtifacts = field(default_factory=ExperimentArtifacts)