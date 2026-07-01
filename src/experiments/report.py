from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..results import FeatureMetadata
from ..utils.io import ensure_directory, save_text


class FeatureReportWriter:
    def write(
        self,
        output_dir: str | Path,
        feature_summary: pd.DataFrame,
        correlation_summary: pd.DataFrame | None,
        importance_summary: pd.DataFrame | None,
        latency_summary: pd.DataFrame | None,
    ) -> Path:
        output_directory = ensure_directory(output_dir)
        lines = ['# Feature Evaluation Report', '']
        lines.append('## Feature Table')
        lines.append('| Feature Name | Category | Execution Time | Number of Values Produced | Importance Score | Correlation Summary | Comments |')
        lines.append('| --- | --- | --- | --- | --- | --- | --- |')

        for feature_name, row in feature_summary.iterrows():
            category = feature_name.split('_')[0]
            importance = 'N/A'
            if importance_summary is not None and 'importance_score' in importance_summary.columns:
                match = importance_summary[importance_summary['feature_name'] == feature_name]
                if not match.empty:
                    importance = f"{float(match.iloc[0]['importance_score']):.4f}"
            correlation = 'See correlation report'
            execution_time = 'See latency report'
            lines.append(f'| {feature_name} | {category} | {execution_time} | 1 | {importance} | {correlation} | Auto-generated research feature |')

        if correlation_summary is not None:
            lines.append('')
            lines.append('## Correlation Summary')
            lines.append(correlation_summary.to_string())

        if latency_summary is not None:
            lines.append('')
            lines.append('## Latency Summary')
            lines.append(latency_summary.to_string())

        report_path = output_directory / 'feature_report.md'
        return save_text('\n'.join(lines), report_path)