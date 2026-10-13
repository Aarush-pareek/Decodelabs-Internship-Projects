"""AI Vision Lab - Image & Text Recognition (Streamlit front end).

Run with:  streamlit run app.py
All recognition logic lives in src/; this file only handles layout and state.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st

from src import config
from src import ui_components as ui
from src.exceptions import VisionError
from src.image_utils import decode_image, encode_png, validate_upload
from src.ocr_engine import get_tesseract_version, list_languages
from src.pipeline import PipelineSettings, run_pipeline
from src.preprocessing import PreprocessSettings

logger = config.configure_logging()

UPLOAD_TYPES = [ext.lstrip(".") for ext in config.SUPPORTED_EXTENSIONS]
BLOCK_OPTIONS = ["Auto", 15, 21, 31, 41, 51, 71, 101]
BLUR_OPTIONS = [0, 3, 5, 7, 9]


@st.cache_data(show_spinner=False)
def tesseract_info() -> tuple[str | None, list[str]]:
    """(version, languages) or (None, []) when Tesseract is unavailable."""
    try:
        return get_tesseract_version(), list_languages()
    except VisionError:
        return None, []


def init_state() -> None:
    defaults = {"widget_n": 0, "result": None, "error": None, "input_id": None, "source": None}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset() -> None:
    """Clear results and recreate the input widgets (new keys) for a fresh start."""
    st.session_state.widget_n += 1
    st.session_state.result = None
    st.session_state.error = None
    st.session_state.input_id = None
    st.session_state.source = None


def sidebar_settings(languages: list[str]) -> PipelineSettings:
    st.sidebar.markdown("### Recognition settings")
    psm = st.sidebar.selectbox(
        "Page mode (PSM)", list(config.PSM_MODES), index=list(config.PSM_MODES).index(config.DEFAULT_PSM),
        format_func=lambda m: f"{m} - {config.PSM_MODES[m]}",
        help="How Tesseract assumes the text is laid out on the page.",
    )
    default_lang = config.DEFAULT_LANGUAGE if config.DEFAULT_LANGUAGE in languages else languages[0]
    lang = st.sidebar.selectbox(
        "Language", languages, index=languages.index(default_lang),
        format_func=lambda code: f"{ui.LANGUAGE_NAMES.get(code, code)} ({code})",
        help="Only languages installed for Tesseract are listed.",
    )

    st.sidebar.markdown("### Preprocessing")
    upscale = st.sidebar.toggle("Enlarge small images", value=True,
                                help="Images under 1000 px are enlarged 2x, which helps small text.")
    deskew = st.sidebar.toggle("Correct skew", value=True,
                               help="Rotates tilted text straight. Straight images are left unchanged.")
    auto_invert = st.sidebar.toggle("Handle light text on dark", value=True,
                                    help="Flips polarity when the background is dark.")
    method = st.sidebar.selectbox(
        "Threshold method", list(config.THRESHOLD_METHODS),
        format_func=lambda m: ui.THRESHOLD_LABELS[m],
        help="Adaptive methods cope with uneven lighting; Otsu uses one global cutoff.",
    )
    blur = st.sidebar.selectbox("Gaussian blur kernel", BLUR_OPTIONS, index=BLUR_OPTIONS.index(3),
                                format_func=lambda k: "Off" if k == 0 else f"{k} x {k}",
                                help="Smooths noise before thresholding.")
    with st.sidebar.expander("Adaptive threshold parameters"):
        block = st.selectbox("Block size", BLOCK_OPTIONS, help="Auto scales with the image size.")
        c_value = st.slider("Constant C", 1, 30, config.DEFAULT_ADAPTIVE_C,
                            help="Subtracted from the local average. Higher values remove more background.")

    st.sidebar.markdown("### Validation")
    st.sidebar.markdown(
        f'<div class="avl-fixed">Minimum confidence: <b>{config.CONFIDENCE_THRESHOLD:.0f}%</b> '
        "(project requirement, fixed)</div>", unsafe_allow_html=True)

    return PipelineSettings(
        psm=psm, lang=lang, threshold=config.CONFIDENCE_THRESHOLD,
        preprocess=PreprocessSettings(
            upscale=upscale, blur_kernel=blur, deskew=deskew, threshold_method=method,
            block_size=None if block == "Auto" else int(block), c=int(c_value), auto_invert=auto_invert,
        ),
    )


def read_input() -> tuple[bytes, str] | None:
    """Return (bytes, filename) from the uploader or the sample picker."""
    n = st.session_state.widget_n
    upload_col, sample_col = st.columns([3, 2])
    with upload_col:
        uploaded = st.file_uploader(
            "Upload an image", type=UPLOAD_TYPES, key=f"upload_{n}",
            help=f"Supported formats: {', '.join(t.upper() for t in UPLOAD_TYPES)}. Maximum 20 MB.",
        )
    with sample_col:
        samples = sorted(p.name for p in config.SAMPLES_DIR.glob("*.png"))
        chosen = st.selectbox("Or choose a sample image", ["None"] + samples, key=f"sample_{n}",
                              help="Synthetic images for demonstrating the pipeline.")
    if uploaded is not None:
        return uploaded.getvalue(), uploaded.name
    if chosen != "None":
        return (config.SAMPLES_DIR / chosen).read_bytes(), chosen
    return None


def show_result(result, name: str) -> None:
    st.markdown(ui.section_html("Recognition pipeline"), unsafe_allow_html=True)
    st.markdown(ui.pipeline_html(result), unsafe_allow_html=True)

    st.markdown(ui.section_html("Confidence validation"), unsafe_allow_html=True)
    st.markdown(ui.validation_html(result), unsafe_allow_html=True)

    st.markdown(ui.section_html("Extracted text", "Only words at or above the minimum confidence are included."),
                unsafe_allow_html=True)
    st.markdown(ui.metrics_html(result), unsafe_allow_html=True)
    if result.text:
        st.code(result.text, language=None)  # built-in copy button appears on hover
    else:
        st.markdown(ui.notice_html(
            "warn", "No text passed the confidence filter.",
            "Try a different page mode, another threshold method, or a clearer image."), unsafe_allow_html=True)

    stem = Path(name).stem or "result"
    text_col, image_col, _ = st.columns([1, 1, 2])
    with text_col:
        st.download_button("Download text (.txt)", data=result.text.encode("utf-8"),
                           file_name=f"{stem}_text.txt", mime="text/plain", disabled=not result.text)
    with image_col:
        st.download_button("Download processed image", data=encode_png(result.preprocessing.binary),
                           file_name=f"{stem}_processed.png", mime="image/png")

    if result.raw_text and result.raw_text != result.text:
        with st.expander("All recognised text, before the confidence filter"):
            st.code(result.raw_text, language=None)

    st.markdown(ui.section_html("Preprocessing stages",
                                "Each stage applied to the image before text recognition."),
                unsafe_allow_html=True)
    st.markdown(ui.stage_gallery_html(result), unsafe_allow_html=True)

    st.markdown(ui.section_html("Recognition overlay"), unsafe_allow_html=True)
    st.markdown(ui.overlay_html(result), unsafe_allow_html=True)

    with st.expander("Word-level confidence"):
        st.markdown(ui.words_table_html(result.words), unsafe_allow_html=True)
    with st.expander("Run details"):
        st.markdown(ui.details_html(result), unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="AI Vision Lab", layout="wide")
    st.markdown(ui.load_css(), unsafe_allow_html=True)
    init_state()

    version, languages = tesseract_info()
    st.markdown(ui.header_html(version), unsafe_allow_html=True)
    if version is None:
        st.markdown(ui.notice_html(
            "error", "Tesseract OCR could not be found.",
            "Text recognition needs the Tesseract program installed on this computer.",
            steps=["Install Tesseract (Windows: the UB Mannheim installer).",
                   "Restart this app. If it is still not found, set TESSERACT_CMD in the .env file.",
                   "See the README section 'Windows Tesseract Setup' for details."]), unsafe_allow_html=True)
        st.stop()

    settings = sidebar_settings(languages)

    st.markdown(ui.section_html("Input image"), unsafe_allow_html=True)
    selected = read_input()

    data = name = None
    if selected is None:
        st.markdown(ui.notice_html("info", "No image selected.",
                                   "Upload an image that contains text, or choose a sample, to begin."),
                    unsafe_allow_html=True)
        st.markdown(ui.section_html("Recognition pipeline"), unsafe_allow_html=True)
        st.markdown(ui.pipeline_html(None), unsafe_allow_html=True)
    else:
        data, name = selected
        input_id = hashlib.sha256(data).hexdigest() + name
        if st.session_state.input_id != input_id:  # a different file: drop old results
            st.session_state.result, st.session_state.error = None, None
            st.session_state.input_id = input_id
        try:
            validate_upload(name, data)
            image = decode_image(data)
            st.markdown(ui.notice_html("success", "Image loaded.", ui.file_summary(name, len(data), image)),
                        unsafe_allow_html=True)
            st.markdown(ui.preview_html(image), unsafe_allow_html=True)
        except VisionError as exc:
            st.markdown(ui.notice_html("error", "This image cannot be used.", str(exc)), unsafe_allow_html=True)
            data = None

    run_col, reset_col, _ = st.columns([1, 1, 4])
    with run_col:
        run = st.button("Run recognition", type="primary", disabled=data is None)
    with reset_col:
        if st.button("Reset"):
            reset()
            st.rerun()

    if run and data is not None:
        try:
            with st.spinner("Processing image..."):
                st.session_state.result = run_pipeline(data, name, settings)
                st.session_state.error = None
        except VisionError as exc:
            st.session_state.result, st.session_state.error = None, str(exc)
        except Exception:  # last resort: log the details, show a plain message
            logger.exception("Unexpected failure while processing %s", name)
            st.session_state.result = None
            st.session_state.error = "Something went wrong while processing the image. Details were written to the log."

    if st.session_state.error:
        st.markdown(ui.notice_html("error", "Recognition failed.", st.session_state.error), unsafe_allow_html=True)

    result = st.session_state.result
    if result is not None and data is not None:
        if result.settings != settings:
            st.markdown(ui.notice_html("warn", "Settings changed since this result.",
                                       "Select Run recognition to apply them."), unsafe_allow_html=True)
        show_result(result, name)


if __name__ == "__main__":
    main()
