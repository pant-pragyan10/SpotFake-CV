from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.plotting import scatter_matrix

from ..utils.io import ensure_directory


class FeatureVisualizer:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = ensure_directory(output_dir)
        sns.set_theme(style='whitegrid')

    def histograms(self, dataframe: pd.DataFrame, feature_columns: list[str], max_features: int = 12) -> list[Path]:
        return [self._save_univariate_plot(dataframe, feature_name, 'histogram', lambda ax, series: sns.histplot(series, kde=True, ax=ax)) for feature_name in feature_columns[:max_features]]

    def box_plots(self, dataframe: pd.DataFrame, feature_columns: list[str], max_features: int = 12) -> list[Path]:
        return [self._save_univariate_plot(dataframe, feature_name, 'boxplot', lambda ax, series: sns.boxplot(y=series, ax=ax)) for feature_name in feature_columns[:max_features]]

    def violin_plots(self, dataframe: pd.DataFrame, feature_columns: list[str], max_features: int = 12) -> list[Path]:
        return [self._save_univariate_plot(dataframe, feature_name, 'violin', lambda ax, series: sns.violinplot(y=series, ax=ax)) for feature_name in feature_columns[:max_features]]

    def kde_distributions(self, dataframe: pd.DataFrame, feature_columns: list[str], max_features: int = 12) -> list[Path]:
        return [self._save_univariate_plot(dataframe, feature_name, 'kde', lambda ax, series: sns.kdeplot(series, fill=True, ax=ax)) for feature_name in feature_columns[:max_features]]

    def class_wise_distributions(self, dataframe: pd.DataFrame, feature_columns: list[str], max_features: int = 12) -> list[Path]:
        saved_paths: list[Path] = []
        for feature_name in feature_columns[:max_features]:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.kdeplot(data=dataframe, x=feature_name, hue='label', fill=True, common_norm=False, ax=ax)
            ax.set_title(f'{feature_name} by class')
            saved_paths.append(self._save_figure(fig, f'{feature_name}_classwise_kde.png'))
        return saved_paths

    def correlation_heatmap(self, correlation_matrix: pd.DataFrame, name: str = 'pearson_heatmap') -> Path:
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(correlation_matrix, cmap='coolwarm', center=0.0, ax=ax)
        ax.set_title('Feature Correlation Heatmap')
        return self._save_figure(fig, f'{name}.png')

    def scatter_matrix(self, dataframe: pd.DataFrame, feature_columns: list[str], max_features: int = 6) -> Path:
        selected = feature_columns[:max_features]
        fig = scatter_matrix(dataframe[selected], figsize=(12, 12), diagonal='kde')
        plt.tight_layout()
        return self._save_figure(plt.gcf(), 'scatter_matrix.png')

    def pair_plot(self, dataframe: pd.DataFrame, feature_columns: list[str], max_features: int = 6) -> Path:
        selected = feature_columns[:max_features]
        grid = sns.pairplot(dataframe[selected + ['label']], hue='label', diag_kind='kde')
        return self._save_figure(grid.fig, 'pair_plot.png')

    def _save_univariate_plot(self, dataframe: pd.DataFrame, feature_name: str, suffix: str, plotter) -> Path:
        fig, ax = plt.subplots(figsize=(8, 4))
        plotter(ax, dataframe[feature_name].dropna())
        ax.set_title(f'{feature_name} {suffix}')
        return self._save_figure(fig, f'{feature_name}_{suffix}.png')

    def _save_figure(self, figure: plt.Figure, filename: str) -> Path:
        output_path = self.output_dir / filename
        figure.tight_layout()
        figure.savefig(output_path, dpi=160, bbox_inches='tight')
        plt.close(figure)
        return output_path