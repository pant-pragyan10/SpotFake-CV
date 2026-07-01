from __future__ import annotations

import pandas as pd

from .types import FeatureDataset, LatencySummary


class LatencyBenchmark:
    def run(self, feature_dataset: FeatureDataset) -> LatencySummary:
        benchmark = feature_dataset.latency_records.copy()
        if benchmark.empty:
            benchmark = pd.DataFrame(columns=['extractor_name', 'execution_time_seconds', 'feature_count'])
        else:
            benchmark = (
                benchmark.groupby('extractor_name', as_index=False)
                .agg(
                    execution_time_seconds=('execution_time_seconds', 'mean'),
                    feature_count=('feature_count', 'mean'),
                )
                .sort_values('execution_time_seconds', ascending=False)
            )
        average_runtime = float(benchmark['execution_time_seconds'].mean()) if not benchmark.empty else 0.0
        slowest = str(benchmark.sort_values('execution_time_seconds', ascending=False).iloc[0]['extractor_name']) if not benchmark.empty else ''
        fastest = str(benchmark.sort_values('execution_time_seconds', ascending=True).iloc[0]['extractor_name']) if not benchmark.empty else ''
        return LatencySummary(benchmark=benchmark, average_runtime_seconds=average_runtime, slowest_detector=slowest, fastest_detector=fastest)