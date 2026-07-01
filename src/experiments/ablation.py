from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._analysis_utils import safe_mutual_information

from .types import AblationSummary, FeatureDataset


@dataclass(frozen=True)
class AblationGroup:
    name: str
    include: tuple[str, ...] | None = None
    exclude: tuple[str, ...] | None = None


class AblationFramework:
    def __init__(self, groups: tuple[AblationGroup, ...] | None = None) -> None:
        self.groups = groups or self.default_groups()

    def default_groups(self) -> tuple[AblationGroup, ...]:
        return (
            AblationGroup('only_fft', include=('fft',)),
            AblationGroup('only_color', include=('colors',)),
            AblationGroup('only_texture', include=('texture', 'sharpness', 'noise')),
            AblationGroup('only_lighting', include=('glare', 'reflections')),
            AblationGroup('only_screen_specific', include=('pixel_grid', 'brightness_periodicity', 'moire')),
            AblationGroup('fft_plus_texture', include=('fft', 'texture', 'sharpness', 'noise')),
            AblationGroup('fft_plus_color', include=('fft', 'colors', 'contrast')),
            AblationGroup('all_except_fft', exclude=('fft',)),
            AblationGroup('all_except_glare', exclude=('glare',)),
            AblationGroup('all_except_texture', exclude=('texture', 'sharpness', 'noise')),
        )

    def run(self, feature_dataset: FeatureDataset) -> AblationSummary:
        dataframe = feature_dataset.dataframe
        feature_columns = list(feature_dataset.feature_names)
        if not feature_columns:
            return AblationSummary(results=pd.DataFrame(columns=['group_name', 'feature_count', 'selected_features', 'mutual_information_sum', 'mutual_information_mean']), ablation_groups=tuple(group.name for group in self.groups))
        y = dataframe['label'].astype(int)
        rows = []

        for group in self.groups:
            selected_columns = self._select_columns(feature_columns, group)
            if not selected_columns:
                continue
            X = dataframe[selected_columns].fillna(dataframe[selected_columns].mean())
            mi = safe_mutual_information(X, y.rename('label'))
            rows.append({
                'group_name': group.name,
                'feature_count': len(selected_columns),
                'selected_features': ', '.join(selected_columns),
                'mutual_information_sum': float(np.sum(mi)),
                'mutual_information_mean': float(np.mean(mi)),
            })

        results = pd.DataFrame(rows).sort_values('mutual_information_sum', ascending=False)
        return AblationSummary(results=results, ablation_groups=tuple(group.name for group in self.groups))

    def _select_columns(self, feature_columns: list[str], group: AblationGroup) -> list[str]:
        selected = feature_columns
        if group.include is not None:
            selected = [column for column in selected if any(token in column for token in group.include)]
        if group.exclude is not None:
            selected = [column for column in selected if not any(token in column for token in group.exclude)]
        return selected