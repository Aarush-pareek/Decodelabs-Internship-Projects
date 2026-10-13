"""Command-line runner: process one image and print the validated result.

Usage (from the project root):
    python scripts/run_ocr.py assets/samples/welcome_clean.png --psm 7
    python scripts/run_ocr.py photo.jpg --threshold otsu --save-dir outputs/run1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.confidence import format_confidence  # noqa: E402
from src.exceptions import VisionError  # noqa: E402
from src.image_utils import encode_png  # noqa: E402
from src.pipeline import PipelineSettings, run_pipeline  # noqa: E402
from src.preprocessing import PreprocessSettings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the OCR pipeline on one image.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--psm", type=int, default=config.DEFAULT_PSM, choices=sorted(config.PSM_MODES))
    parser.add_argument("--lang", default=config.DEFAULT_LANGUAGE)
    parser.add_argument("--threshold", default=config.DEFAULT_THRESHOLD_METHOD,
                        choices=config.THRESHOLD_METHODS, help="thresholding method")
    parser.add_argument("--no-deskew", action="store_true")
    parser.add_argument("--save-dir", type=Path, help="write the processed images here")
    args = parser.parse_args()

    settings = PipelineSettings(
        psm=args.psm, lang=args.lang,
        preprocess=PreprocessSettings(threshold_method=args.threshold, deskew=not args.no_deskew),
    )
    try:
        result = run_pipeline(args.image.read_bytes(), args.image.name, settings)
    except (VisionError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    val = result.validation
    print("-" * 46)
    print("OCR RESULT")
    print("-" * 46)
    print("Detected text:")
    print(result.text or "(no text passed the confidence filter)")
    print()
    print(f"Average confidence : {format_confidence(val.average)}")
    print(f"Required           : {val.required:.0f}%")
    print(f"Words              : {val.total_words} detected, {val.kept_words} kept, {val.dropped_words} dropped")
    print(f"Status             : {val.status.value.replace('_', ' ').upper()}")
    print("-" * 46)

    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        prep = result.preprocessing
        for name, image in (("gray", prep.grayscale), ("blurred", prep.blurred),
                            ("binary", prep.binary), ("overlay", result.annotated)):
            (args.save_dir / f"{name}.png").write_bytes(encode_png(image))
        (args.save_dir / "text.txt").write_text(result.text, encoding="utf-8")
    return 0 if val.passed else 2


if __name__ == "__main__":
    sys.exit(main())
