from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .types import FeatureDataset, FeatureSummary


@dataclass(frozen=True)
class FeatureStatisticsCalculator:
    def summarize(self, feature_dataset: FeatureDataset) -> FeatureSummary:
        dataframe = feature_dataset.dataframe
        feature_columns = list(feature_dataset.feature_names)
        numeric = dataframe[feature_columns]

        summary = pd.DataFrame({
            'mean': numeric.mean(),
            'median': numeric.median(),
            'variance': numeric.var(ddof=0),
            'std': numeric.std(ddof=0),
            'min': numeric.min(),
            'max': numeric.max(),
            'missing': numeric.isna().sum(),
        })
        summary.index.name = 'feature_name'

        class_stats = (
            dataframe.groupby('label')[feature_columns]
            .agg(['mean', 'median', 'var', 'std', 'min', 'max'])
        )
        missing_values = numeric.isna().sum().rename('missing_values').reset_index().rename(columns={'index': 'feature_name'})
        summary_text = self._render_summary_text(summary)
        return FeatureSummary(statistics=summary, per_class_statistics=class_stats, missing_values=missing_values, summary_text=summary_text)

    def _render_summary_text(self, summary: pd.DataFrame) -> str:
        lines = ['# Feature Statistics', '', f'Total features: {len(summary)}', '']
        lines.append('Top missing values:')
        top_missing = summary.sort_values('missing', ascending=False).head(10)
        for feature_name, row in top_missing.iterrows():
            lines.append(f'- {feature_name}: missing={int(row["missing"])} mean={row["mean"]:.4f} std={row["std"]:.4f}')
        return '\n'.join(lines)