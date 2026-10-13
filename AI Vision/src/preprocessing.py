"""Image preprocessing for OCR.

Stage order (matches the Project 4 brief):
    size normalisation -> grayscale -> [polarity fix] -> Gaussian blur
    -> deskew -> thresholding

Each stage is a small pure function so it can be tested and reused alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src import config
from src.exceptions import PreprocessingError


@dataclass(frozen=True)
class PreprocessSettings:
    """User-adjustable preprocessing options."""

    upscale: bool = True
    blur_kernel: int = config.DEFAULT_BLUR_KERNEL
    deskew: bool = True
    threshold_method: str = config.DEFAULT_THRESHOLD_METHOD
    block_size: int | None = config.DEFAULT_ADAPTIVE_BLOCK_SIZE
    c: int = config.DEFAULT_ADAPTIVE_C
    auto_invert: bool = True

    def __post_init__(self) -> None:
        if self.threshold_method not in config.THRESHOLD_METHODS:
            raise PreprocessingError(f"Unknown threshold method: {self.threshold_method}")
        if self.blur_kernel < 0 or (self.blur_kernel > 1 and self.blur_kernel % 2 == 0):
            raise PreprocessingError("The blur kernel size must be 0, 1 or an odd number.")
        if self.block_size is not None and (self.block_size < 3 or self.block_size % 2 == 0):
            raise PreprocessingError("The threshold block size must be an odd number of 3 or more.")


@dataclass
class ThresholdResult:
    binary: np.ndarray
    method: str
    block_size: int | None  # None for Otsu
    c: int | None  # None for Otsu
    otsu_cutoff: float | None  # only set for Otsu


@dataclass
class PreprocessResult:
    """Every intermediate image plus the parameters that produced it."""

    original: np.ndarray  # after size normalisation, still colour
    grayscale: np.ndarray
    blurred: np.ndarray
    binary: np.ndarray  # final image sent to OCR (deskewed if applicable)
    scale_factor: float
    inverted: bool
    blur_kernel: int
    skew_angle: float  # correction applied, degrees (0.0 = none)
    threshold: ThresholdResult


# --------------------------------------------------------------------------
# Individual stages
# --------------------------------------------------------------------------
def normalise_size(image: np.ndarray, upscale: bool = True) -> tuple[np.ndarray, float]:
    """Shrink very large images and enlarge small ones. Returns (image, scale)."""
    height, width = image.shape[:2]
    longest = max(height, width)

    if longest > config.MAX_PROCESS_PX:
        scale, interpolation = config.MAX_PROCESS_PX / longest, cv2.INTER_AREA
    elif upscale and longest < config.UPSCALE_BELOW_PX:
        scale, interpolation = config.UPSCALE_FACTOR, cv2.INTER_CUBIC
    else:
        return image, 1.0

    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, size, interpolation=interpolation), scale


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to single-channel grayscale (already-gray input is copied)."""
    if image.ndim == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def needs_inversion(gray: np.ndarray) -> bool:
    """True when the background is dark (light text on dark), so polarity must flip.

    Adaptive thresholding assumes dark text on a light background; Tesseract
    also performs best that way.
    """
    return bool(np.median(gray) < 127)


def apply_gaussian_blur(gray: np.ndarray, kernel: int = config.DEFAULT_BLUR_KERNEL) -> np.ndarray:
    """Gaussian blur for noise reduction. kernel 0 or 1 disables it."""
    if kernel <= 1:
        return gray.copy()
    return cv2.GaussianBlur(gray, (kernel, kernel), 0)


def _rotate(
    image: np.ndarray, angle: float, expand: bool, border_value: int = 0
) -> np.ndarray:
    """Rotate counter-clockwise by `angle` degrees about the centre.

    With expand=True the canvas grows so no corner is cropped.
    """
    height, width = image.shape[:2]
    centre = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
    new_w, new_h = width, height
    if expand:
        cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
        new_w = int(round(height * sin + width * cos))
        new_h = int(round(height * cos + width * sin))
        matrix[0, 2] += new_w / 2.0 - centre[0]
        matrix[1, 2] += new_h / 2.0 - centre[1]
    return cv2.warpAffine(
        image, matrix, (new_w, new_h),
        flags=cv2.INTER_CUBIC if expand else cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=border_value,
    )


def estimate_skew_angle(gray: np.ndarray) -> float:
    """Estimate the rotation (degrees, counter-clockwise) that straightens the text.

    Projection-profile method: text lines are straight when the row-wise sum of
    text pixels has maximum variance (sharp peaks at lines, valleys between).
    Returns 0.0 when the image is already straight, has no text-like content,
    or no angle clearly improves on the original.
    """
    height, width = gray.shape[:2]
    longest = max(height, width)
    small = gray
    if longest > 600:
        factor = 600 / longest
        small = cv2.resize(gray, (round(width * factor), round(height * factor)),
                           interpolation=cv2.INTER_AREA)

    # A LOCAL threshold isolates text even under uneven lighting. A global
    # (Otsu) mask would treat a shadowed half of the page as "text" and
    # produce a false skew angle.
    text_mask = cv2.adaptiveThreshold(
        small, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
    )
    if np.count_nonzero(text_mask) < 50:
        return 0.0

    def score(angle: float) -> float:
        rotated = _rotate(text_mask, angle, expand=False)
        return float(np.var(rotated.sum(axis=1, dtype=np.float64)))

    limit = config.DESKEW_MAX_ANGLE_DEG
    coarse = np.arange(-limit, limit + 0.01, 1.0)
    best = float(max(coarse, key=score))
    fine = np.clip(np.arange(best - 1.0, best + 1.01, 0.1), -limit, limit)
    best = float(max(fine, key=score))

    # Require a clear gain over "no rotation" so straight images stay untouched.
    if abs(best) < config.DESKEW_MIN_ANGLE_DEG or score(best) < score(0.0) * 1.03:
        return 0.0
    return round(best, 1)


def deskew(gray: np.ndarray, angle: float) -> np.ndarray:
    """Rotate by `angle` degrees, padding with the background tone (median)."""
    if angle == 0.0:
        return gray
    return _rotate(gray, angle, expand=True, border_value=int(np.median(gray)))


def auto_block_size(shape: tuple[int, ...]) -> int:
    """Pick an odd neighbourhood size proportional to the image (about 3% of the
    longest side, clamped to 15-101). A block much smaller than the stroke width
    would hollow out thick letters; a huge one would stop adapting to shadows."""
    longest = max(shape[:2])
    block = int(round(longest * 0.03))
    block = max(15, min(101, block))
    return block if block % 2 == 1 else block + 1


def apply_threshold(
    gray: np.ndarray,
    method: str = config.DEFAULT_THRESHOLD_METHOD,
    block_size: int | None = None,
    c: int = config.DEFAULT_ADAPTIVE_C,
) -> ThresholdResult:
    """Convert grayscale to pure black text on white background."""
    if method == "otsu":
        cutoff, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return ThresholdResult(binary, method, None, None, float(cutoff))

    if method not in ("adaptive_gaussian", "adaptive_mean"):
        raise PreprocessingError(f"Unknown threshold method: {method}")

    block = block_size or auto_block_size(gray.shape)
    adaptive = (cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method == "adaptive_gaussian"
                else cv2.ADAPTIVE_THRESH_MEAN_C)
    binary = cv2.adaptiveThreshold(gray, 255, adaptive, cv2.THRESH_BINARY, block, c)
    return ThresholdResult(binary, method, block, c, None)


# --------------------------------------------------------------------------
# Full preprocessing run
# --------------------------------------------------------------------------
def preprocess(image: np.ndarray, settings: PreprocessSettings | None = None) -> PreprocessResult:
    """Run every preprocessing stage and return all intermediate images."""
    settings = settings or PreprocessSettings()
    try:
        sized, scale = normalise_size(image, settings.upscale)
        gray = to_grayscale(sized)

        inverted = settings.auto_invert and needs_inversion(gray)
        working = cv2.bitwise_not(gray) if inverted else gray

        blurred = apply_gaussian_blur(working, settings.blur_kernel)

        angle = estimate_skew_angle(blurred) if settings.deskew else 0.0
        straightened = deskew(blurred, angle)

        threshold = apply_threshold(
            straightened, settings.threshold_method, settings.block_size, settings.c
        )
    except PreprocessingError:
        raise
    except cv2.error as exc:
        raise PreprocessingError("Image preprocessing failed inside OpenCV.") from exc

    return PreprocessResult(
        original=sized,
        grayscale=working,
        blurred=blurred,
        binary=threshold.binary,
        scale_factor=scale,
        inverted=inverted,
        blur_kernel=settings.blur_kernel,
        skew_angle=angle,
        threshold=threshold,
    )
