"""Benchmark the production predictor over a folder of images.

Usage:
    python scripts/predict_benchmark.py --dataset-root data/raw

If the folder has real/ and screen/ subfolders, also reports accuracy at the
model's default decision threshold. Images that fail to load/process are
logged and skipped rather than aborting the run.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

logging.disable(logging.CRITICAL)

from src.models.inference import get_detector

VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def iter_images(dataset_root: Path):
    subfolders = {"real": 0, "screen": 1}
    has_labels = all((dataset_root / name).is_dir() for name in subfolders)
    if has_labels:
        for name, label in subfolders.items():
            for path in sorted((dataset_root / name).glob("*")):
                if path.is_file() and path.suffix.lower() in VALID_SUFFIXES:
                    yield path, label
    else:
        for path in sorted(dataset_root.glob("*")):
            if path.is_file() and path.suffix.lower() in VALID_SUFFIXES:
                yield path, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark predict.py over a folder of images.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    detector = get_detector()  # load once, outside the timed loop

    latencies_ms: list[float] = []
    correct = 0
    labeled_count = 0
    failed: list[tuple[Path, str]] = []

    start = time.perf_counter()
    for path, label in iter_images(args.dataset_root):
        try:
            call_start = time.perf_counter()
            probability = detector.predict_proba(path)
            latencies_ms.append((time.perf_counter() - call_start) * 1000.0)
        except Exception as exc:
            failed.append((path, f"{type(exc).__name__}: {exc}"))
            continue
        if label is not None:
            labeled_count += 1
            prediction = 1 if probability >= args.threshold else 0
            correct += int(prediction == label)
    total_elapsed = time.perf_counter() - start

    n = len(latencies_ms)
    print(f"Images processed: {n}  Failed: {len(failed)}")
    if n:
        print(f"Average latency: {sum(latencies_ms) / n:.2f} ms/image")
        print(f"Throughput: {n / total_elapsed:.1f} images/sec (single process, includes I/O)")
    if labeled_count:
        print(f"Accuracy @ threshold={args.threshold}: {correct / labeled_count:.4f} ({correct}/{labeled_count})")
    for path, error in failed[:10]:
        print(f"  failed: {path} -> {error}")


if __name__ == "__main__":
    main()
