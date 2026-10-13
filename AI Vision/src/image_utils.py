"""Image input validation, decoding and small helpers.

Images are decoded with Pillow (handles EXIF rotation, transparency and 16-bit
files) and then converted to an OpenCV-style BGR uint8 NumPy array.
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from src import config
from src.exceptions import InvalidImageError, UnsupportedFormatError

_FORMAT_LABEL = ", ".join(e.lstrip(".").upper() for e in config.SUPPORTED_EXTENSIONS)


def validate_upload(filename: str, data: bytes | None) -> None:
    """Check the file name and raw bytes before attempting to decode."""
    if not data:
        raise InvalidImageError("The file is empty. Choose an image that contains data.")
    extension = Path(filename or "").suffix.lower()
    if extension not in config.SUPPORTED_EXTENSIONS:
        shown = extension or "no extension"
        raise UnsupportedFormatError(
            f"Files with {shown} are not supported. Use one of: {_FORMAT_LABEL}."
        )
    if len(data) > config.MAX_FILE_BYTES:
        limit_mb = config.MAX_FILE_BYTES // (1024 * 1024)
        raise InvalidImageError(f"The file is larger than the {limit_mb} MB limit.")


def decode_image(data: bytes) -> np.ndarray:
    """Decode image bytes into a BGR uint8 array, raising InvalidImageError on failure."""
    try:
        with Image.open(io.BytesIO(data)) as probe:
            width, height = probe.size  # read from the header, before full decode
            _check_dimensions(width, height)
            image = ImageOps.exif_transpose(probe)
            image.load()
            rgb = _to_rgb_array(image)
    except InvalidImageError:
        raise
    except Exception as exc:  # Pillow raises many different types for bad files
        raise InvalidImageError(
            "The image could not be read. The file may be corrupted or not a real image."
        ) from exc

    _check_dimensions(rgb.shape[1], rgb.shape[0])
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def load_image_file(path: str | Path) -> np.ndarray:
    """Read, validate and decode an image from disk."""
    path = Path(path)
    if not path.is_file():
        raise InvalidImageError(f"File not found: {path.name}")
    data = path.read_bytes()
    validate_upload(path.name, data)
    return decode_image(data)


def _check_dimensions(width: int, height: int) -> None:
    if width < config.MIN_DIMENSION_PX or height < config.MIN_DIMENSION_PX:
        raise InvalidImageError(
            f"The image is too small ({width}x{height} px). "
            f"The minimum is {config.MIN_DIMENSION_PX} px on each side."
        )
    if width > config.MAX_DIMENSION_PX or height > config.MAX_DIMENSION_PX:
        raise InvalidImageError(
            f"The image is too large ({width}x{height} px). "
            f"The maximum is {config.MAX_DIMENSION_PX} px on each side."
        )


def _to_rgb_array(image: Image.Image) -> np.ndarray:
    """Convert any Pillow image to an RGB uint8 array (transparency -> white)."""
    has_alpha = image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    )
    if has_alpha:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        return np.asarray(background.convert("RGB"), dtype=np.uint8)

    if image.mode in ("I;16", "I", "F"):
        values = np.asarray(image, dtype=np.float64)
        low, high = float(values.min()), float(values.max())
        scaled = np.zeros_like(values) if high == low else (values - low) / (high - low) * 255
        gray = scaled.astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)

    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def image_info(image: np.ndarray) -> dict:
    """Basic array facts shown in the UI (the 'H x W x C' array from the brief)."""
    height, width = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    return {"height": height, "width": width, "channels": channels, "dtype": str(image.dtype)}


def to_bgr(image: np.ndarray) -> np.ndarray:
    """Return a 3-channel version of a grayscale/binary image (for drawing/display)."""
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image


def encode_png(image: np.ndarray) -> bytes:
    """Encode an array as PNG bytes (used for downloads and HTML previews)."""
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise InvalidImageError("The image could not be encoded as PNG.")
    return buffer.tobytes()


def make_preview(image: np.ndarray, max_side: int = 900) -> np.ndarray:
    """Shrink large images for display; never enlarges."""
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image
    scale = max_side / longest
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)
