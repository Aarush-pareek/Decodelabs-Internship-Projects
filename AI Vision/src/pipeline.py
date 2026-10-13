"""End-to-end recognition pipeline: validate -> preprocess -> OCR -> validate confidence.

This module has no UI dependency, so the Streamlit app, the command-line script
and the tests all call the same function.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from src import config
from src.confidence import ValidationResult, split_by_confidence, validate
from src.exceptions import PreprocessingError
from src.image_utils import decode_image, image_info, to_bgr, validate_upload
from src.ocr_engine import WordResult, build_text, run_ocr
from src.preprocessing import PreprocessResult, PreprocessSettings, preprocess

logger = logging.getLogger("aivision.pipeline")

KEPT_COLOUR = (60, 140, 20)  # BGR green
DROPPED_COLOUR = (40, 40, 200)  # BGR red


@dataclass(frozen=True)
class PipelineSettings:
    psm: int = config.DEFAULT_PSM
    lang: str = config.DEFAULT_LANGUAGE
    threshold: float = config.CONFIDENCE_THRESHOLD
    preprocess: PreprocessSettings = field(default_factory=PreprocessSettings)


@dataclass
class PipelineResult:
    input_info: dict
    preprocessing: PreprocessResult
    words: list[WordResult]  # every recognised word
    kept_words: list[WordResult]  # words >= threshold
    dropped_words: list[WordResult]
    raw_text: str  # all words, unfiltered
    text: str  # final text, only words >= threshold
    validation: ValidationResult
    annotated: np.ndarray  # OCR image with boxes (green kept / red dropped)
    settings: PipelineSettings
    timings_ms: dict[str, float]


def run_pipeline_on_array(
    image: np.ndarray, settings: PipelineSettings | None = None
) -> PipelineResult:
    """Run the pipeline on an already-decoded BGR image."""
    settings = settings or PipelineSettings()
    timings: dict[str, float] = {}
    started = time.perf_counter()

    t0 = time.perf_counter()
    prep = preprocess(image, settings.preprocess)
    timings["preprocessing"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    words = run_ocr(prep.binary, psm=settings.psm, lang=settings.lang)
    timings["ocr"] = (time.perf_counter() - t0) * 1000

    kept, dropped = split_by_confidence(words, settings.threshold)
    validation = validate(words, settings.threshold)
    annotated = draw_word_boxes(prep.binary, kept, dropped)
    timings["total"] = (time.perf_counter() - started) * 1000

    logger.info(
        "OCR finished: words=%d kept=%d avg=%s status=%s psm=%d total_ms=%.0f",
        len(words), len(kept), validation.average, validation.status.value,
        settings.psm, timings["total"],
    )
    return PipelineResult(
        input_info=image_info(image),
        preprocessing=prep,
        words=words,
        kept_words=kept,
        dropped_words=dropped,
        raw_text=build_text(words),
        text=build_text(kept),
        validation=validation,
        annotated=annotated,
        settings=settings,
        timings_ms=timings,
    )


def run_pipeline(
    data: bytes, filename: str, settings: PipelineSettings | None = None
) -> PipelineResult:
    """Validate and decode an uploaded file, then run the pipeline."""
    validate_upload(filename, data)
    image = decode_image(data)
    try:
        return run_pipeline_on_array(image, settings)
    except cv2.error as exc:
        logger.exception("Unexpected OpenCV error")
        raise PreprocessingError("Image processing failed unexpectedly.") from exc


def draw_word_boxes(
    image: np.ndarray, kept: list[WordResult], dropped: list[WordResult]
) -> np.ndarray:
    """Draw word boxes on a copy of the OCR image: green = kept, red = dropped."""
    canvas = to_bgr(image).copy()
    for words, colour in ((dropped, DROPPED_COLOUR), (kept, KEPT_COLOUR)):
        for word in words:
            top_left = (word.left, word.top)
            bottom_right = (word.left + word.width, word.top + word.height)
            cv2.rectangle(canvas, top_left, bottom_right, colour, 2)
    return canvas
