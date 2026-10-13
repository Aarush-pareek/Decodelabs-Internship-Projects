"""Tesseract OCR wrapper (pytesseract) returning per-word confidence."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
import pytesseract

from src import config
from src.exceptions import OCRFailureError, TesseractNotFoundError

logger = logging.getLogger("aivision.ocr")

_LANG_PATTERN = re.compile(r"^[A-Za-z0-9_]+(\+[A-Za-z0-9_]+)*$")

TESSERACT_HELP = (
    "Tesseract OCR was not found. Install it and either add it to your PATH or set "
    "TESSERACT_CMD in the .env file (see the README, section 'Windows Tesseract Setup')."
)


@dataclass(frozen=True)
class WordResult:
    """One recognised word with the confidence Tesseract reported for it (0-100)."""

    text: str
    confidence: float
    left: int
    top: int
    width: int
    height: int
    block: int = 0
    paragraph: int = 0
    line: int = 0


def configure_tesseract() -> str:
    """Point pytesseract at the Tesseract executable and return its path."""
    command = config.find_tesseract_cmd()
    if not command:
        raise TesseractNotFoundError(TESSERACT_HELP)
    pytesseract.pytesseract.tesseract_cmd = command
    return command


def get_tesseract_version() -> str:
    """Return the installed Tesseract version string."""
    configure_tesseract()
    try:
        return str(pytesseract.get_tesseract_version())
    except Exception as exc:
        raise OCRFailureError("Tesseract was found but could not be started.") from exc


def list_languages() -> list[str]:
    """Languages installed for Tesseract (the 'osd' helper model is excluded)."""
    configure_tesseract()
    try:
        languages = pytesseract.get_languages(config="")
    except Exception as exc:
        raise OCRFailureError("The installed OCR languages could not be listed.") from exc
    return sorted(lang for lang in languages if lang != "osd")


def run_ocr(
    image: np.ndarray,
    psm: int = config.DEFAULT_PSM,
    lang: str = config.DEFAULT_LANGUAGE,
    timeout: int = 60,
) -> list[WordResult]:
    """Recognise text and return one WordResult per word with its real confidence.

    Tesseract reports -1 for layout elements that are not words; those and empty
    strings are discarded. No confidence value is ever invented here.
    """
    if psm not in config.PSM_MODES:
        raise OCRFailureError(f"Unsupported page segmentation mode: {psm}.")
    if not _LANG_PATTERN.match(lang or ""):
        raise OCRFailureError(f"Invalid OCR language code: {lang!r}.")
    configure_tesseract()

    try:
        data = pytesseract.image_to_data(
            image, lang=lang, config=f"--psm {psm}",
            output_type=pytesseract.Output.DICT, timeout=timeout,
        )
    except pytesseract.TesseractNotFoundError as exc:
        raise TesseractNotFoundError(TESSERACT_HELP) from exc
    except pytesseract.TesseractError as exc:
        logger.error("Tesseract error: %s", exc)
        hint = " Check that the selected language is installed." if "language" in str(exc).lower() else ""
        raise OCRFailureError(f"Tesseract could not process the image.{hint}") from exc
    except RuntimeError as exc:  # pytesseract raises RuntimeError on timeout
        logger.error("Tesseract runtime error: %s", exc)
        raise OCRFailureError("Text recognition took too long and was stopped.") from exc

    return parse_ocr_data(data)


def parse_ocr_data(data: dict) -> list[WordResult]:
    """Convert pytesseract's image_to_data dictionary into WordResult objects."""
    words: list[WordResult] = []
    count = len(data.get("text", []))
    for i in range(count):
        text = str(data["text"][i]).strip()
        try:
            confidence = float(data["conf"][i])
        except (TypeError, ValueError):
            continue
        if not text or confidence < 0:
            continue
        words.append(
            WordResult(
                text=text,
                confidence=confidence,
                left=int(data["left"][i]),
                top=int(data["top"][i]),
                width=int(data["width"][i]),
                height=int(data["height"][i]),
                block=int(data["block_num"][i]),
                paragraph=int(data["par_num"][i]),
                line=int(data["line_num"][i]),
            )
        )
    return words


def build_text(words: list[WordResult]) -> str:
    """Rebuild readable text: words on a line joined by spaces, paragraphs
    separated by a blank line."""
    lines: list[str] = []
    current_key = None
    current: list[str] = []
    last_paragraph = None

    for word in words:
        key = (word.block, word.paragraph, word.line)
        paragraph = (word.block, word.paragraph)
        if key != current_key:
            if current:
                lines.append(" ".join(current))
            if last_paragraph is not None and paragraph != last_paragraph:
                lines.append("")
            current, current_key, last_paragraph = [], key, paragraph
        current.append(word.text)
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)
