from __future__ import annotations

from .latency import LatencyBenchmark


def benchmark_features(*args, **kwargs):
    return LatencyBenchmark().run(*args, **kwargs)
