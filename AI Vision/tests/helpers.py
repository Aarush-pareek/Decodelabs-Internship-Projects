"""Shared helpers for tests: build small synthetic images in memory."""

import io

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src import config


def tesseract_available() -> bool:
    return config.find_tesseract_cmd() is not None


def _font(size: int):
    for name in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def text_image_bgr(lines=("Welcome to DecodeLabs",), size=60, width=900) -> np.ndarray:
    """Black text on white as a BGR array."""
    height = int(60 * 2 + len(lines) * size * 1.5)
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    for i, line in enumerate(lines):
        draw.text((60, 60 + i * size * 1.5), line, font=_font(size), fill=(20, 20, 20))
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def to_bytes(image_bgr: np.ndarray, ext: str = ".png") -> bytes:
    ok, buffer = cv2.imencode(ext, image_bgr)
    assert ok
    return buffer.tobytes()


def pil_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    stream = io.BytesIO()
    image.save(stream, format=fmt)
    return stream.getvalue()
