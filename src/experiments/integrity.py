from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .types import DatasetSummary, ExtractorAuditSummary, FeatureDataset


@dataclass(frozen=True)
class DatasetIntegrityAnalyzer:
    def summarize(self, feature_dataset: FeatureDataset) -> DatasetSummary:
        dataframe = feature_dataset.dataframe
        metadata = feature_dataset.metadata
        sample_count = int(len(dataframe))
        class_balance = {
            int(label): int(count)
            for label, count in dataframe['label'].value_counts().sort_index().items()
        }
        feature_dimensions = (int(dataframe.shape[0]), int(len(feature_dataset.feature_names)))
        extracted_feature_count = int(feature_dataset.matrix.shape[1] if feature_dataset.matrix.ndim == 2 else len(feature_dataset.feature_names))
        average_extraction_time_seconds = float(metadata['feature_extraction_time_seconds'].mean()) if not metadata.empty and 'feature_extraction_time_seconds' in metadata.columns else 0.0
        failed_images = int(len(feature_dataset.failures))
        summary_text = self._render_summary_text(sample_count, class_balance, feature_dimensions, extracted_feature_count, average_extraction_time_seconds, failed_images)
        return DatasetSummary(
            sample_count=sample_count,
            class_balance=class_balance,
            feature_dimensions=feature_dimensions,
            extracted_feature_count=extracted_feature_count,
            average_extraction_time_seconds=average_extraction_time_seconds,
            failed_images=failed_images,
            summary_text=summary_text,
        )

    def _render_summary_text(
        self,
        sample_count: int,
        class_balance: dict[int, int],
        feature_dimensions: tuple[int, int],
        extracted_feature_count: int,
        average_extraction_time_seconds: float,
        failed_images: int,
    ) -> str:
        lines = ['# Dataset Integrity Summary', '']
        lines.append(f'- Number of images: {sample_count}')
        lines.append(f'- Class balance: {class_balance}')
        lines.append(f'- Feature dimensions: {feature_dimensions[0]} x {feature_dimensions[1]}')
        lines.append(f'- Extracted feature count: {extracted_feature_count}')
        lines.append(f'- Average extraction time: {average_extraction_time_seconds:.6f} s')
        lines.append(f'- Failed images: {failed_images}')
        return '\n'.join(lines)


@dataclass(frozen=True)
class ExtractorAuditAnalyzer:
    def summarize(self, feature_dataset: FeatureDataset) -> ExtractorAuditSummary:
        latency = feature_dataset.latency_records.copy()
        if latency.empty:
            audit = pd.DataFrame(columns=['extractor_name', 'execution_time_seconds', 'feature_count', 'error_count', 'error_messages'])
            report_text = '# Extractor Audit\n\nNo extractor telemetry was recorded.'
            return ExtractorAuditSummary(audit=audit, report_text=report_text)

        grouped = latency.groupby('extractor_name', as_index=False).agg(
            execution_time_seconds=('execution_time_seconds', 'mean'),
            feature_count=('feature_count', 'mean'),
        )
        error_counts = latency.assign(has_error=latency['error'].fillna('').astype(str).str.len() > 0).groupby('extractor_name', as_index=False).agg(
            error_count=('has_error', 'sum'),
            error_messages=('error', lambda values: '; '.join(sorted({str(value) for value in values if str(value)}))),
        )
        audit = grouped.merge(error_counts, on='extractor_name', how='left').fillna({'error_count': 0, 'error_messages': ''})
        audit = audit.sort_values('execution_time_seconds', ascending=False).reset_index(drop=True)
        report_text = self._render_report(audit)
        return ExtractorAuditSummary(audit=audit, report_text=report_text)

    def _render_report(self, audit: pd.DataFrame) -> str:
        lines = ['# Extractor Audit', '', '| extractor | execution_time_seconds | number_of_features | errors |', '| --- | --- | --- | --- |']
        for _, row in audit.iterrows():
            lines.append(
                f"| {row['extractor_name']} | {float(row['execution_time_seconds']):.6f} | {float(row['feature_count']):.1f} | {int(row['error_count'])} | {row['error_messages']} |"
            )
        return '\n'.join(lines)
