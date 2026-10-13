import unittest

import numpy as np

from src import ui_components as ui
from src.ocr_engine import WordResult
from tests.helpers import tesseract_available, text_image_bgr


class EscapingTests(unittest.TestCase):
    def test_ocr_text_and_filenames_are_escaped(self):
        words = [WordResult("<script>alert(1)</script>", 95, 0, 0, 5, 5)]
        table = ui.words_table_html(words)
        self.assertNotIn("<script>", table)
        self.assertIn("&lt;script&gt;", table)
        self.assertNotIn("<img", ui.notice_html("error", "<img src=x>", "<b>"))

    def test_compact_removes_newlines_and_indent(self):
        self.assertEqual(ui.compact("<div>\n    <p>a</p>\n\n</div>"), "<div><p>a</p></div>")


class MarkupTests(unittest.TestCase):
    def test_header_states(self):
        self.assertIn("Tesseract 5.3.4 ready", ui.header_html("5.3.4"))
        self.assertIn("Tesseract not found", ui.header_html(None))

    def test_idle_pipeline_lists_seven_stages(self):
        markup = ui.pipeline_html(None)
        self.assertEqual(markup.count("<li "), 7)
        for name in ("Original image", "Grayscale", "Gaussian blur", "Adaptive threshold",
                     "OCR", "Confidence validation", "Final text"):
            self.assertIn(name, markup)

    def test_data_uri_is_valid(self):
        for jpeg in (True, False):
            uri = ui.data_uri(text_image_bgr(), jpeg=jpeg)
            self.assertTrue(uri.startswith("data:image/"))
        gray = np.zeros((40, 40), np.uint8)
        self.assertTrue(ui.data_uri(gray).startswith("data:image/png;base64,"))

    def test_css_is_single_line_and_has_tokens(self):
        css = ui.load_css()
        self.assertNotIn("\n", css)
        self.assertIn("--accent", css)


@unittest.skipUnless(tesseract_available(), "Tesseract is not installed")
class ResultMarkupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.pipeline import run_pipeline_on_array
        cls.result = run_pipeline_on_array(text_image_bgr())

    def test_validation_panel_shows_score_requirement_and_status(self):
        markup = ui.validation_html(self.result)
        self.assertIn("PASS", markup)
        self.assertIn("Required minimum: 80%", markup)
        self.assertIn("avl-valid pass", markup)

    def test_validation_panel_failure_state(self):
        import dataclasses
        from src.confidence import Status, validate
        low = validate([WordResult("a", 40, 0, 0, 1, 1)])
        failing = dataclasses.replace(self.result, validation=low)
        markup = ui.validation_html(failing)
        self.assertIn("BELOW THRESHOLD", markup)
        self.assertIn("avl-valid fail", markup)
        self.assertNotIn(">PASS<", markup)
        self.assertIs(low.status, Status.BELOW_THRESHOLD)

    def test_gallery_has_four_labelled_stages(self):
        markup = ui.stage_gallery_html(self.result)
        for label in ("Original", "Grayscale", "Blurred", "Thresholded"):
            self.assertIn(f"<b>{label}</b>", markup)

    def test_metrics_and_details_render(self):
        self.assertIn("Words detected", ui.metrics_html(self.result))
        self.assertIn("PSM 3", ui.details_html(self.result))
        self.assertIn("Recognised words", ui.overlay_html(self.result))


if __name__ == "__main__":
    unittest.main()
