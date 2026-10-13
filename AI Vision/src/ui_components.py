"""HTML builders for the Streamlit UI.

Every function returns an HTML string and has no Streamlit dependency, so the
markup (especially escaping of OCR text and file names) can be unit-tested.
"""

from __future__ import annotations

import base64
import html
import re

import cv2
import numpy as np

from src import config
from src.confidence import Status, format_confidence
from src.image_utils import encode_png, image_info, make_preview
from src.ocr_engine import WordResult
from src.pipeline import PipelineResult

LANGUAGE_NAMES = {
    "eng": "English", "deu": "German", "fra": "French", "spa": "Spanish", "ita": "Italian",
    "por": "Portuguese", "nld": "Dutch", "hin": "Hindi", "ara": "Arabic", "rus": "Russian",
}
THRESHOLD_LABELS = {
    "adaptive_gaussian": "Adaptive (Gaussian weighted)",
    "adaptive_mean": "Adaptive (mean)",
    "otsu": "Global (Otsu)",
}
MAX_TABLE_ROWS = 300


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def compact(markup: str) -> str:
    """Remove newlines/indentation so Markdown never treats the HTML as a code block."""
    return "".join(line.strip() for line in markup.splitlines())


def load_css() -> str:
    css = (config.PROJECT_ROOT / "src" / "style.css").read_text(encoding="utf-8")
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    return f"<style>{compact(css)}</style>"


def data_uri(image: np.ndarray, max_side: int = 700, jpeg: bool = False) -> str:
    """Embed an image in HTML (downscaled for display)."""
    small = make_preview(image, max_side)
    if jpeg:
        ok, buffer = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise ValueError("JPEG encoding failed")
        payload, mime = buffer.tobytes(), "image/jpeg"
    else:
        payload, mime = encode_png(small), "image/png"
    return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"


# --------------------------------------------------------------------------
# Header, notices, preview
# --------------------------------------------------------------------------
def header_html(tesseract_version: str | None) -> str:
    chip = (
        f'<span class="avl-chip"><span class="avl-dot"></span>Tesseract {esc(tesseract_version)} ready</span>'
        if tesseract_version
        else '<span class="avl-chip bad"><span class="avl-dot"></span>Tesseract not found</span>'
    )
    return compact(f"""
    <div class="avl-header">
      <div><h1>AI Vision Lab</h1><p>Image &amp; Text Recognition</p></div>
      {chip}
    </div>""")


def notice_html(kind: str, title: str, body: str = "", steps: list[str] | None = None) -> str:
    """kind: error | success | warn | info. Title and body are escaped."""
    items = f"<ol>{''.join(f'<li>{esc(s)}</li>' for s in steps)}</ol>" if steps else ""
    detail = esc(body) if body else ""
    return compact(f'<div class="avl-notice {esc(kind)}"><b>{esc(title)}</b>{detail}{items}</div>')


def section_html(title: str, note: str = "") -> str:
    note_html = f'<p class="avl-note">{esc(note)}</p>' if note else ""
    return compact(f'<h2 class="avl-h">{esc(title)}</h2>{note_html}')


def preview_html(image: np.ndarray) -> str:
    return compact(f'<div class="avl-preview"><img alt="Uploaded image preview" '
                   f'src="{data_uri(image, 900, jpeg=True)}"></div>')


def file_summary(name: str, size_bytes: int, image: np.ndarray) -> str:
    info = image_info(image)
    kb = size_bytes / 1024
    size = f"{kb / 1024:.1f} MB" if kb >= 1024 else f"{kb:.0f} KB"
    return (f"{name} - {info['width']} x {info['height']} px, {info['channels']} channels, "
            f"{info['dtype']}, {size}")


# --------------------------------------------------------------------------
# Pipeline strip
# --------------------------------------------------------------------------
def pipeline_html(result: PipelineResult | None = None) -> str:
    """The seven pipeline stages; 'idle' before a run, with real details after."""
    names = ["Original image", "Grayscale", "Gaussian blur", "Adaptive threshold",
             "OCR", "Confidence validation", "Final text"]
    if result is None:
        stages = [(n, "idle", "Waiting for input") for n in names]
    else:
        prep, val = result.preprocessing, result.validation
        info = result.input_info
        blur = prep.blur_kernel
        thr = prep.threshold
        if thr.method == "otsu":
            thr_detail = f"Otsu cutoff {thr.otsu_cutoff:.0f}"
        else:
            thr_detail = f"block {thr.block_size}, C {thr.c}"
        if prep.skew_angle:
            thr_detail += f", deskew {prep.skew_angle:+.1f} deg"
        passed = val.status is Status.VALIDATED
        stages = [
            (names[0], "done", f"{info['width']}x{info['height']} px"),
            (names[1], "done", "inverted" if prep.inverted else "1 channel"),
            (names[2], "done" if blur > 1 else "skip",
             f"Gaussian {blur}x{blur}" if blur > 1 else "Skipped"),
            ("Threshold" if thr.method == "otsu" else names[3], "done", thr_detail),
            (names[4], "done", f"{val.total_words} words, PSM {result.settings.psm}"),
            (names[5], "ok" if passed else "warn",
             "Meets 80%" if passed else ("No text" if val.status is Status.NO_TEXT else "Below 80%")),
            (names[6], "ok" if passed else "warn", f"{val.kept_words} words kept"),
        ]
    items = "".join(
        f'<li class="avl-stage {state}"><span class="n">{esc(name)}</span><span class="d">{esc(detail)}</span></li>'
        for name, state, detail in stages
    )
    return compact(f'<ol class="avl-pipeline">{items}</ol>')


# --------------------------------------------------------------------------
# Validation + metrics
# --------------------------------------------------------------------------
def validation_html(result: PipelineResult) -> str:
    val = result.validation
    required = f"{val.required:.0f}%"
    if val.status is Status.VALIDATED:
        state, badge = "pass", "PASS"
    elif val.status is Status.BELOW_THRESHOLD:
        state, badge = "fail", "BELOW THRESHOLD"
    else:
        state, badge = "nodata", "NO TEXT DETECTED"
    width = 0.0 if val.average is None else max(0.0, min(100.0, val.average))
    return compact(f"""
    <section class="avl-valid {state}" aria-label="Confidence validation">
      <div>
        <div class="avl-vl">Average confidence</div>
        <div class="avl-vs">{esc(format_confidence(val.average))}</div>
        <div class="avl-vr">Required minimum: {esc(required)}</div>
      </div>
      <div>
        <span class="avl-badge">{esc(badge)}</span>
        <div class="avl-bar" role="img" aria-label="Confidence {esc(format_confidence(val.average))}, required {esc(required)}">
          <div class="avl-fill" style="width:{width:.2f}%"></div>
          <div class="avl-mark" style="left:{val.required:.2f}%"><span>{esc(required)} required</span></div>
        </div>
        <p class="avl-vm">{esc(val.message)}</p>
      </div>
    </section>""")


def metrics_html(result: PipelineResult) -> str:
    val = result.validation
    tiles = [
        ("Words detected", str(val.total_words)),
        (f"Kept (>= {val.required:.0f}%)", str(val.kept_words)),
        ("Dropped", str(val.dropped_words)),
        ("Lowest confidence", format_confidence(val.min_confidence)),
        ("Highest confidence", format_confidence(val.max_confidence)),
        ("Processing time", f"{result.timings_ms['total'] / 1000:.2f} s"),
    ]
    cells = "".join(f'<div class="avl-metric"><div class="l">{esc(l)}</div><div class="v">{esc(v)}</div></div>'
                    for l, v in tiles)
    return compact(f'<div class="avl-metrics">{cells}</div>')


# --------------------------------------------------------------------------
# Images
# --------------------------------------------------------------------------
def _figure(image: np.ndarray, title: str, caption: str, jpeg: bool = False) -> str:
    return (f'<figure class="avl-fig"><div class="img"><img alt="{esc(title)}" src="{data_uri(image, jpeg=jpeg)}"></div>'
            f'<figcaption><b>{esc(title)}</b>{esc(caption)}</figcaption></figure>')


def stage_gallery_html(result: PipelineResult) -> str:
    prep = result.preprocessing
    h, w = prep.original.shape[:2]
    scale = f", enlarged {prep.scale_factor:g}x" if prep.scale_factor > 1 else (
        ", reduced for processing" if prep.scale_factor < 1 else "")
    thr = prep.threshold
    if thr.method == "otsu":
        thr_caption = f"Otsu global threshold, cutoff {thr.otsu_cutoff:.0f}"
    else:
        thr_caption = f"{THRESHOLD_LABELS[thr.method]}, block {thr.block_size}, C {thr.c}"
    if prep.skew_angle:
        thr_caption += f"; deskewed {prep.skew_angle:+.1f} deg"
    blur_caption = (f"Gaussian {prep.blur_kernel}x{prep.blur_kernel}" if prep.blur_kernel > 1
                    else "Blur disabled")
    gray_caption = "Inverted (light text on dark)" if prep.inverted else "Single channel"
    figures = "".join([
        _figure(prep.original, "Original", f"{w}x{h} px{scale}", jpeg=True),
        _figure(prep.grayscale, "Grayscale", gray_caption),
        _figure(prep.blurred, "Blurred", blur_caption),
        _figure(prep.binary, "Thresholded", thr_caption),
    ])
    return compact(f'<div class="avl-gallery">{figures}</div>')


def overlay_html(result: PipelineResult) -> str:
    val = result.validation
    figures = (
        _figure(result.preprocessing.binary, "Image sent to OCR", "Black text on white, after preprocessing")
        + _figure(result.annotated, "Recognised words",
                  f"{val.kept_words} kept, {val.dropped_words} dropped")
    )
    legend = (
        '<p class="avl-legend">'
        f'<span><i style="border-color:#3c8c14"></i>Kept (at or above {esc(f"{val.required:.0f}")}%)</span>'
        '<span><i style="border-color:#c82828"></i>Dropped (below threshold)</span></p>'
    )
    return compact(f'<div class="avl-gallery two">{figures}</div>{legend}')


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def words_table_html(words: list[WordResult], threshold: float = config.CONFIDENCE_THRESHOLD) -> str:
    if not words:
        return notice_html("info", "No words to list.")
    rows = []
    for w in words[:MAX_TABLE_ROWS]:
        kept = w.confidence >= threshold
        rows.append(
            f'<tr class="{"" if kept else "drop"}"><td>{esc(w.text)}</td>'
            f'<td class="num">{w.confidence:.1f}%</td><td>{"Kept" if kept else "Dropped"}</td></tr>'
        )
    extra = (f'<p class="avl-note">Showing the first {MAX_TABLE_ROWS} of {len(words)} words.</p>'
             if len(words) > MAX_TABLE_ROWS else "")
    return compact(
        '<div class="avl-scroll"><table class="avl-table"><thead><tr><th>Word</th><th>Confidence</th>'
        f'<th>Status</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>{extra}'
    )


def details_html(result: PipelineResult) -> str:
    prep, s, t = result.preprocessing, result.settings, result.timings_ms
    thr = prep.threshold
    rows = [
        ("Page mode", f"PSM {s.psm} - {config.PSM_MODES[s.psm]}"),
        ("Language", LANGUAGE_NAMES.get(s.lang, s.lang)),
        ("Threshold method", THRESHOLD_LABELS[thr.method]),
        ("Block size / C", "n/a" if thr.method == "otsu" else f"{thr.block_size} / {thr.c}"),
        ("Gaussian blur kernel", "off" if prep.blur_kernel <= 1 else f"{prep.blur_kernel}x{prep.blur_kernel}"),
        ("Skew correction", f"{prep.skew_angle:+.1f} deg" if prep.skew_angle else "none applied"),
        ("Scale factor", f"{prep.scale_factor:g}x"),
        ("Polarity inverted", "yes" if prep.inverted else "no"),
        ("Preprocessing time", f"{t['preprocessing']:.0f} ms"),
        ("OCR time", f"{t['ocr']:.0f} ms"),
    ]
    body = "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in rows)
    return compact(f'<dl class="avl-kv">{body}</dl>')
