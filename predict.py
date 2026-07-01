"""Final submission entry point.

Usage:
    python predict.py path/to/image.jpg

Prints ONE number in [0, 1] where:
    0 = genuine photo
    1 = photo of a display (recapture / fraud)
"""

from __future__ import annotations

import logging
import sys

logging.disable(logging.CRITICAL)

from src.models.inference import predict_proba


def predict(image_path: str) -> float:
    return predict_proba(image_path)


if __name__ == "__main__":
    try:
        print(predict(sys.argv[1]))
    except Exception as exc:  # invalid path / unsupported / corrupted image
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
