from __future__ import annotations

from PIL import Image


def resize_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return image.resize(size, Image.Resampling.LANCZOS)
