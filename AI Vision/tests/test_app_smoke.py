"""Smoke tests for app.py using a stand-in for Streamlit.

IMPORTANT: this checks the app's own control flow and the HTML it emits. It does
NOT verify real Streamlit rendering; run `streamlit run app.py` for that.
"""

import importlib
import sys
import types
import unittest
from contextlib import contextmanager
from unittest import mock

from src.exceptions import TesseractNotFoundError
from tests.helpers import tesseract_available


class StopApp(Exception):
    pass


class FakeState(dict):
    __getattr__ = dict.get

    def __setattr__(self, key, value):
        self[key] = value


class Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def build_fake_streamlit(script):
    """script: {'upload': (bytes, name) | None, 'sample': str, 'run': bool}"""
    st = types.ModuleType("streamlit")
    st.out = []  # everything rendered
    st.codes = []
    st.session_state = FakeState()

    def selectbox(label, options, index=0, key=None, **kw):
        if key and key.startswith("sample_"):
            return script.get("sample", "None")
        return options[index]

    def file_uploader(label, key=None, **kw):
        upload = script.get("upload")
        if not upload:
            return None
        return types.SimpleNamespace(getvalue=lambda: upload[0], name=upload[1])

    def button(label, **kw):
        return bool(script.get("run")) if label == "Run recognition" else False

    def stop():
        raise StopApp

    def cache_data(*args, **kw):
        return (lambda f: f) if not args else args[0]

    sidebar = types.SimpleNamespace(
        markdown=lambda *a, **k: None, selectbox=selectbox,
        toggle=lambda label, value=False, **k: value, expander=lambda *a, **k: Ctx(),
    )
    st.set_page_config = lambda **k: None
    st.markdown = lambda body, **k: st.out.append(body)
    st.code = lambda body, **k: st.codes.append(body)
    st.columns = lambda spec: [Ctx() for _ in range(spec if isinstance(spec, int) else len(spec))]
    st.expander = lambda *a, **k: Ctx()
    st.spinner = lambda *a, **k: Ctx()
    st.selectbox, st.file_uploader, st.button = selectbox, file_uploader, button
    st.slider = lambda label, lo, hi, value, **k: value
    st.download_button = lambda *a, **k: False
    st.rerun = lambda: None
    st.stop, st.cache_data, st.sidebar = stop, cache_data, sidebar
    return st


@contextmanager
def loaded_app(script):
    fake = build_fake_streamlit(script)
    with mock.patch.dict(sys.modules, {"streamlit": fake}):
        sys.modules.pop("app", None)
        app = importlib.import_module("app")
        try:
            yield app, fake
        finally:
            sys.modules.pop("app", None)


class AppSmokeTests(unittest.TestCase):
    def test_empty_state_shows_prompt_and_idle_pipeline(self):
        with loaded_app({}) as (app, st):
            app.main()
        page = "".join(st.out)
        self.assertIn("No image selected", page)
        self.assertIn("Waiting for input", page)

    def test_missing_tesseract_shows_setup_help_and_stops(self):
        with loaded_app({}) as (app, st):
            with mock.patch.object(app, "get_tesseract_version", side_effect=TesseractNotFoundError("x")):
                with self.assertRaises(StopApp):
                    app.main()
        self.assertIn("Tesseract OCR could not be found", "".join(st.out))

    def test_corrupted_upload_shows_friendly_error_not_traceback(self):
        script = {"upload": (b"not an image", "broken.png")}
        with loaded_app(script) as (app, st):
            app.main()
        page = "".join(st.out)
        self.assertIn("This image cannot be used", page)
        self.assertNotIn("Traceback", page)

    def test_unsupported_file_type_message(self):
        script = {"upload": (b"abc", "doc.pdf"), "run": True}
        with loaded_app(script) as (app, st):
            app.main()
        self.assertIn("not supported", "".join(st.out))

    @unittest.skipUnless(tesseract_available(), "Tesseract is not installed")
    def test_full_run_on_sample_displays_validation_and_text(self):
        script = {"sample": "paragraph_clean.png", "run": True}
        with loaded_app(script) as (app, st):
            app.main()
        page = "".join(st.out)
        self.assertIn("PASS", page)
        self.assertIn("Required minimum: 80%", page)
        self.assertIn("Grayscale", page)
        self.assertIn("Thresholded", page)
        self.assertTrue(any("Optical character recognition" in c for c in st.codes))

    @unittest.skipUnless(tesseract_available(), "Tesseract is not installed")
    def test_low_quality_sample_shows_below_threshold(self):
        script = {"sample": "low_contrast_blurred.png", "run": True}
        with loaded_app(script) as (app, st):
            app.main()
        page = "".join(st.out)
        self.assertIn("BELOW THRESHOLD", page)
        self.assertNotIn(">PASS<", page)

    @unittest.skipUnless(tesseract_available(), "Tesseract is not installed")
    def test_blank_sample_shows_no_text_state(self):
        script = {"sample": "blank_page.png", "run": True}
        with loaded_app(script) as (app, st):
            app.main()
        self.assertIn("NO TEXT DETECTED", "".join(st.out))

    @unittest.skipUnless(tesseract_available(), "Tesseract is not installed")
    def test_reset_clears_result(self):
        script = {"sample": "paragraph_clean.png", "run": True}
        with loaded_app(script) as (app, st):
            app.main()
            self.assertIsNotNone(st.session_state.result)
            app.reset()
            self.assertIsNone(st.session_state.result)
            self.assertEqual(st.session_state.widget_n, 1)


if __name__ == "__main__":
    unittest.main()
