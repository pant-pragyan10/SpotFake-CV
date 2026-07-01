from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image


def load_image(image_path: str | Path) -> Image.Image:
    return Image.open(Path(image_path)).convert("RGB")
