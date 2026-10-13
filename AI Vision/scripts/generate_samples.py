"""Generate the synthetic sample images used by the demo and the tests.

These images are rendered with Pillow, so they are SYNTHETIC: they exercise the
pipeline (clean text, noise, shadow, skew, low contrast, inverted colours,
blank page) but are not a measure of real-world accuracy.

Usage (from the project root):
    python scripts/generate_samples.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402

PARAGRAPH = [
    "Optical character recognition converts",
    "images of printed text into machine",
    "readable characters. Pre-processing",
    "improves the accuracy of the result.",
]
INVOICE = [
    ((40, 40), "INVOICE"),
    ((520, 44), "No: 10234"),
    ((40, 180), "Date: 12 March 2026"),
    ((380, 300), "Customer: DecodeLabs"),
    ((40, 420), "Total due: 1250.00"),
]


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try common TrueType fonts, falling back to Pillow's built-in font."""
    for name in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def text_page(lines, size=40, width=1000, margin=50, line_gap=1.5,
              fg=(20, 20, 20), bg=(255, 255, 255)) -> Image.Image:
    font = load_font(size)
    height = int(margin * 2 + len(lines) * size * line_gap)
    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)
    for index, line in enumerate(lines):
        draw.text((margin, margin + index * size * line_gap), line, font=font, fill=fg)
    return image


def add_noise(image: Image.Image, sigma: float, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    array = np.asarray(image, dtype=np.float32)
    array += rng.normal(0, sigma, array.shape)
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def add_shadow(image: Image.Image, strength: float) -> Image.Image:
    """Darken the image with a left-to-right illumination gradient."""
    array = np.asarray(image, dtype=np.float32)
    ramp = np.linspace(1.0, 1.0 - strength, array.shape[1], dtype=np.float32)
    return Image.fromarray(np.clip(array * ramp[None, :, None], 0, 255).astype(np.uint8))


def build_samples() -> dict[str, Image.Image]:
    samples: dict[str, Image.Image] = {}

    samples["welcome_clean"] = text_page(["Welcome to DecodeLabs"], size=64, width=900, margin=60)
    samples["paragraph_clean"] = text_page(PARAGRAPH)
    samples["document_noisy_shadow"] = add_noise(
        add_shadow(text_page(PARAGRAPH), strength=0.75), sigma=14, seed=7
    )

    page = text_page(PARAGRAPH, width=1100)
    samples["text_skewed"] = page.rotate(6, expand=True, fillcolor=(255, 255, 255),
                                         resample=Image.BICUBIC)

    faint = text_page(PARAGRAPH, size=30, fg=(165, 165, 165), bg=(215, 215, 215))
    samples["low_contrast_blurred"] = add_noise(
        faint.filter(ImageFilter.GaussianBlur(2.2)), sigma=10, seed=3
    )

    samples["light_on_dark"] = text_page(PARAGRAPH, fg=(240, 240, 240), bg=(25, 30, 45))

    sparse = Image.new("RGB", (1000, 560), (255, 255, 255))
    draw = ImageDraw.Draw(sparse)
    for position, label in INVOICE:
        draw.text(position, label, font=load_font(44), fill=(20, 20, 20))
    samples["invoice_sparse"] = sparse

    samples["blank_page"] = Image.new("RGB", (800, 400), (255, 255, 255))
    return samples


def main() -> None:
    config.SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for name, image in build_samples().items():
        target = config.SAMPLES_DIR / f"{name}.png"
        image.save(target)
        print(f"wrote {target.relative_to(config.PROJECT_ROOT)}  {image.size[0]}x{image.size[1]}")


if __name__ == "__main__":
    main()
