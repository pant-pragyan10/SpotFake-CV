from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter


@contextmanager
def timer():
    start = perf_counter()
    elapsed = None
    try:
        yield lambda: perf_counter() - start
    finally:
        elapsed = perf_counter() - start
        _ = elapsed
