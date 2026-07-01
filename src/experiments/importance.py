from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.inspection import permutation_importance

from ._analysis_utils import safe_mutual_information
from .types import FeatureDataset, ImportanceSummary


class FeatureImportanceAnalyzer:
    def rank(self, feature_dataset: FeatureDataset, include_model_based: bool = False) -> ImportanceSummary:
        dataframe = feature_dataset.dataframe
        feature_columns = list(feature_dataset.feature_names)
        if not feature_columns:
            empty = pd.DataFrame(columns=['feature_name', 'mutual_information', 'variance', 'importance_score', 'random_forest_importance', 'permutation_importance'])
            return ImportanceSummary(ranking=empty, report_text='# Feature Importance Ranking\n\nNo features available.')
        X = dataframe[feature_columns].fillna(dataframe[feature_columns].mean())
        y = dataframe['label'].astype(int)

        mutual_information = safe_mutual_information(X, y.rename('label'))
        variance_selector = VarianceThreshold()
        variance_selector.fit(X)
        variance_scores = variance_selector.variances_

        ranking = pd.DataFrame({
            'feature_name': feature_columns,
            'mutual_information': mutual_information,
            'variance': variance_scores,
        })

        ranking['importance_score'] = (
            ranking['mutual_information'].rank(pct=True) * 0.7
            + ranking['variance'].rank(pct=True) * 0.3
        )

        if include_model_based and len(feature_columns) > 1:
            forest = RandomForestClassifier(n_estimators=128, random_state=42, n_jobs=-1)
            forest.fit(X, y)
            ranking['random_forest_importance'] = forest.feature_importances_
            try:
                perm = permutation_importance(forest, X, y, n_repeats=10, random_state=42, n_jobs=-1)
                ranking['permutation_importance'] = perm.importances_mean
            except Exception:
                ranking['permutation_importance'] = np.nan
        else:
            ranking['random_forest_importance'] = np.nan
            ranking['permutation_importance'] = np.nan

        ranking = ranking.sort_values('importance_score', ascending=False).reset_index(drop=True)
        report_text = self._render_report(ranking)
        return ImportanceSummary(ranking=ranking, report_text=report_text)

    def _render_report(self, ranking: pd.DataFrame) -> str:
        lines = ['# Feature Importance Ranking', '']
        for _, row in ranking.head(20).iterrows():
            lines.append(f'- {row["feature_name"]}: importance={row["importance_score"]:.4f} mi={row["mutual_information"]:.4f} var={row["variance"]:.4f}')
        return '\n'.join(lines)