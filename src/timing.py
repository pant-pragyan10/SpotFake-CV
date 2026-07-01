from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from time import perf_counter
from typing import Any, Callable, TypeVar

import sys

try:
    import resource
except ImportError:  # pragma: no cover - non-Unix fallback
    resource = None


@dataclass(frozen=True)
class TimingResult:
    execution_time_seconds: float
    memory_usage_mb: float | None = None
    item_count: int | None = None


T = TypeVar("T")


def current_memory_usage_mb() -> float | None:
    if resource is None:
        return None

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024 * 1024)
    return usage / 1024.0


def measure_call(
    function: Callable[..., T],
    *args: Any,
    record_memory: bool = True,
    **kwargs: Any,
) -> tuple[T, TimingResult]:
    start_time = perf_counter()
    value = function(*args, **kwargs)
    elapsed = perf_counter() - start_time
    memory_usage = current_memory_usage_mb() if record_memory else None
    return value, TimingResult(execution_time_seconds=elapsed, memory_usage_mb=memory_usage)


def timed(function: Callable[..., T]) -> Callable[..., tuple[T, TimingResult]]:
    @wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> tuple[T, TimingResult]:
        return measure_call(function, *args, **kwargs)

    return wrapper