from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from configs.config import ExperimentConfig

from ..features.feature_vector import FeatureVectorBuilder
from ..logging_utils import get_logger
from ..pipeline import ExperimentPipeline
from ..results import PipelineResult
from ..utils.io import ensure_directory, save_dataframe, save_json
from .types import FeatureDataset


@dataclass(frozen=True)
class DatasetSample:
    path: Path
    label: int


class FeatureDatasetBuilder:
    def __init__(
        self,
        config: ExperimentConfig | None = None,
        feature_vector_builder: FeatureVectorBuilder | None = None,
    ) -> None:
        self.config = config or ExperimentConfig()
        self.pipeline = ExperimentPipeline(config=self.config.pipeline, feature_vector_builder=feature_vector_builder)
        self.logger = get_logger(__name__)

    def iter_samples(self, dataset_root: str | Path) -> Iterable[DatasetSample]:
        root = Path(dataset_root)
        valid_suffixes = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
        for label_name, label_value in (('real', 0), ('screen', 1)):
            for image_path in sorted((root / label_name).glob('*')):
                if image_path.is_file() and image_path.suffix.lower() in valid_suffixes:
                    yield DatasetSample(path=image_path, label=label_value)

    def build(self, dataset_root: str | Path) -> FeatureDataset:
        rows: list[dict[str, float | int | str]] = []
        metadata_rows: list[dict[str, float | int | str]] = []
        latency_rows: list[dict[str, float | int | str]] = []
        failure_rows: list[dict[str, float | int | str]] = []
        feature_names: tuple[str, ...] = ()

        for sample in self.iter_samples(dataset_root):
            self.logger.info('Extracting features for %s', sample.path)
            try:
                result = self.pipeline.run(sample.path)
                if result.feature_vector.extractor_errors:
                    error_message = '; '.join(
                        f"{error.extractor_name}: {error.error}" for error in result.feature_vector.extractor_errors
                    )
                    self.logger.error('Skipping %s due to extractor failures: %s', sample.path, error_message)
                    failure_rows.append(
                        {
                            'path': str(sample.path),
                            'label': sample.label,
                            'error_type': 'extractor_failure',
                            'error_message': error_message,
                        }
                    )
                    continue

                feature_names = result.feature_vector.feature_names
                rows.append(self._row_from_result(sample, result))
                metadata_rows.append(self._metadata_row_from_result(sample, result))
                latency_rows.extend(self._latency_rows_from_result(sample, result))
            except Exception as exc:
                error_message = f"{type(exc).__name__}: {exc}"
                self.logger.exception('Skipping %s due to processing failure', sample.path)
                failure_rows.append(
                    {
                        'path': str(sample.path),
                        'label': sample.label,
                        'error_type': type(exc).__name__,
                        'error_message': error_message,
                    }
                )

        dataframe = pd.DataFrame(rows)
        metadata = pd.DataFrame(metadata_rows, columns=['path', 'label', 'width', 'height', 'mode', 'preprocessing_time_seconds', 'feature_extraction_time_seconds', 'total_time_seconds', 'feature_count'])
        latency_records = pd.DataFrame(latency_rows, columns=['path', 'label', 'extractor_name', 'execution_time_seconds', 'feature_count', 'error'])
        failures = pd.DataFrame(failure_rows, columns=['path', 'label', 'error_type', 'error_message'])
        matrix = dataframe[list(feature_names)].to_numpy(dtype=np.float32) if feature_names and not dataframe.empty else np.empty((0, 0), dtype=np.float32)
        labels = dataframe['label'].to_numpy(dtype=np.int64) if 'label' in dataframe.columns and not dataframe.empty else np.empty((0,), dtype=np.int64)
        return FeatureDataset(
            dataframe=dataframe,
            feature_names=feature_names,
            matrix=matrix,
            labels=labels,
            metadata=metadata,
            latency_records=latency_records,
            failures=failures,
        )

    def _row_from_result(self, sample: DatasetSample, result: PipelineResult) -> dict[str, float | int | str]:
        row: dict[str, float | int | str] = {
            'path': str(sample.path),
            'label': sample.label,
        }
        for feature_name, feature_value in zip(result.feature_vector.feature_names, result.feature_vector.feature_vector):
            row[feature_name] = float(feature_value)
        return row

    def _latency_rows_from_result(self, sample: DatasetSample, result: PipelineResult) -> list[dict[str, float | int | str]]:
        rows: list[dict[str, float | int | str]] = []
        for extractor_result in result.feature_vector.extractor_results:
            rows.append(
                {
                    'path': str(sample.path),
                    'label': sample.label,
                    'extractor_name': extractor_result.extractor_name,
                    'execution_time_seconds': extractor_result.timing.execution_time_seconds,
                    'feature_count': len(extractor_result.feature_values),
                    'error': extractor_result.error or '',
                }
            )
        for extractor_result in result.feature_vector.extractor_errors:
            rows.append(
                {
                    'path': str(sample.path),
                    'label': sample.label,
                    'extractor_name': extractor_result.extractor_name,
                    'execution_time_seconds': extractor_result.timing.execution_time_seconds,
                    'feature_count': 0,
                    'error': extractor_result.error or '',
                }
            )
        return rows

    def _metadata_row_from_result(self, sample: DatasetSample, result: PipelineResult) -> dict[str, float | int | str]:
        return {
            'path': str(sample.path),
            'label': sample.label,
            'width': result.image_info.width,
            'height': result.image_info.height,
            'mode': result.image_info.mode,
            'preprocessing_time_seconds': result.preprocessing.timing.execution_time_seconds,
            'feature_extraction_time_seconds': result.feature_vector.timing.execution_time_seconds,
            'total_time_seconds': result.timing.execution_time_seconds,
            'feature_count': len(result.feature_vector.feature_vector),
        }

    def export(self, feature_dataset: FeatureDataset, output_dir: str | Path) -> dict[str, Path]:
        output_directory = ensure_directory(output_dir)
        csv_path = save_dataframe(feature_dataset.dataframe, output_directory / 'features.csv')
        npy_path = output_directory / 'features.npy'
        np.save(npy_path, feature_dataset.matrix)
        labels_path = output_directory / 'labels.npy'
        np.save(labels_path, feature_dataset.labels)
        feature_names_path = output_directory / 'feature_names.json'
        save_json({'feature_names': list(feature_dataset.feature_names)}, feature_names_path)
        metadata_path = save_dataframe(feature_dataset.metadata, output_directory / 'metadata.csv')
        failures_path = save_dataframe(feature_dataset.failures, output_directory / 'failed_images.csv')
        parquet_path = output_directory / 'features.parquet'
        try:
            feature_dataset.dataframe.to_parquet(parquet_path, index=False)
        except Exception:
            parquet_path = Path()
        latency_path = save_dataframe(feature_dataset.latency_records, output_directory / 'feature_latency.csv')
        return {
            'csv': csv_path,
            'npy': npy_path,
            'labels': labels_path,
            'feature_names': feature_names_path,
            'metadata': metadata_path,
            'failures': failures_path,
            'latency': latency_path,
            'parquet': parquet_path,
        }