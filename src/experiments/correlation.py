from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from ._analysis_utils import safe_mutual_information
from .types import CorrelationSummary, FeatureDataset


class FeatureCorrelationAnalyzer:
    def analyze(self, feature_dataset: FeatureDataset, correlation_threshold: float = 0.95) -> CorrelationSummary:
        dataframe = feature_dataset.dataframe
        feature_columns = list(feature_dataset.feature_names)
        feature_frame = dataframe[feature_columns]
        labels = dataframe['label'].astype(int)

        pearson = feature_frame.corr(method='pearson')
        spearman = feature_frame.corr(method='spearman')
        mi_scores = safe_mutual_information(feature_frame, labels)
        mutual_information = pd.DataFrame({'feature_name': feature_columns, 'mutual_information': mi_scores}).sort_values('mutual_information', ascending=False)

        correlated_pairs = []
        for left, right in itertools.combinations(feature_columns, 2):
            value = float(pearson.loc[left, right])
            if abs(value) >= correlation_threshold:
                correlated_pairs.append({'feature_a': left, 'feature_b': right, 'pearson': value})
        highly_correlated_pairs = pd.DataFrame(correlated_pairs)

        report_text = self._render_report(pearson, spearman, mutual_information, highly_correlated_pairs, correlation_threshold)
        return CorrelationSummary(
            pearson=pearson,
            spearman=spearman,
            mutual_information=mutual_information,
            highly_correlated_pairs=highly_correlated_pairs,
            report_text=report_text,
        )

    def _render_report(
        self,
        pearson: pd.DataFrame,
        spearman: pd.DataFrame,
        mutual_information: pd.DataFrame,
        correlated_pairs: pd.DataFrame,
        threshold: float,
    ) -> str:
        lines = ['# Feature Correlation Report', '', f'Correlation threshold: {threshold:.2f}', '']
        lines.append('Most informative features by mutual information:')
        for _, row in mutual_information.head(10).iterrows():
            lines.append(f'- {row["feature_name"]}: {row["mutual_information"]:.6f}')
        lines.append('')
        lines.append(f'Highly correlated feature pairs: {len(correlated_pairs)}')
        return '\n'.join(lines)