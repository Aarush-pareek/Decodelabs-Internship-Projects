import unittest
from unittest import mock

import pytesseract

from src import ocr_engine
from src.exceptions import OCRFailureError, TesseractNotFoundError
from src.ocr_engine import WordResult, build_text, parse_ocr_data
from tests.helpers import tesseract_available, text_image_bgr

import numpy as np


def fake_data(rows):
    """rows: (text, conf, block, par, line)"""
    n = len(rows)
    return {
        "text": [r[0] for r in rows], "conf": [r[1] for r in rows],
        "left": list(range(n)), "top": [0] * n, "width": [10] * n, "height": [10] * n,
        "block_num": [r[2] for r in rows], "par_num": [r[3] for r in rows],
        "line_num": [r[4] for r in rows],
    }


class ParseTests(unittest.TestCase):
    def test_minus_one_and_blank_entries_are_discarded(self):
        data = fake_data([("", "-1", 1, 0, 0), ("Hello", "96.5", 1, 1, 1),
                          ("   ", "88", 1, 1, 1), ("junk", "-1", 1, 1, 1)])
        words = parse_ocr_data(data)
        self.assertEqual([(w.text, w.confidence) for w in words], [("Hello", 96.5)])

    def test_confidence_comes_from_tesseract_values_unchanged(self):
        words = parse_ocr_data(fake_data([("a", "91.234", 1, 1, 1), ("b", 12, 1, 1, 1)]))
        self.assertEqual([w.confidence for w in words], [91.234, 12.0])

    def test_unparseable_confidence_skipped(self):
        self.assertEqual(parse_ocr_data(fake_data([("a", "n/a", 1, 1, 1)])), [])

    def test_empty_data(self):
        self.assertEqual(parse_ocr_data({"text": []}), [])


class BuildTextTests(unittest.TestCase):
    def w(self, text, block, par, line):
        return WordResult(text, 90, 0, 0, 1, 1, block, par, line)

    def test_lines_and_paragraphs(self):
        words = [self.w("Hello", 1, 1, 1), self.w("world", 1, 1, 1),
                 self.w("Second", 1, 1, 2), self.w("Para", 1, 2, 1)]
        self.assertEqual(build_text(words), "Hello world\nSecond\n\nPara")

    def test_empty(self):
        self.assertEqual(build_text([]), "")


class ErrorHandlingTests(unittest.TestCase):
    image = np.full((50, 50), 255, np.uint8)

    def test_missing_tesseract_gives_clear_error(self):
        with mock.patch("src.ocr_engine.config.find_tesseract_cmd", return_value=None):
            with self.assertRaises(TesseractNotFoundError) as ctx:
                ocr_engine.run_ocr(self.image)
        self.assertIn("TESSERACT_CMD", str(ctx.exception))

    def test_invalid_psm_rejected(self):
        with self.assertRaises(OCRFailureError):
            ocr_engine.run_ocr(self.image, psm=99)

    def test_invalid_language_string_rejected(self):
        for bad in ("", "eng;rm -rf", "eng --oem 0", "../x"):
            with self.assertRaises(OCRFailureError):
                ocr_engine.run_ocr(self.image, lang=bad)

    def _patched(self, side_effect):
        return mock.patch.multiple(
            "src.ocr_engine",
            configure_tesseract=mock.DEFAULT,
        ), mock.patch("src.ocr_engine.pytesseract.image_to_data", side_effect=side_effect)

    def test_tesseract_error_is_wrapped(self):
        a, b = self._patched(pytesseract.TesseractError(1, "Failed loading language 'xyz'"))
        with a, b:
            with self.assertRaises(OCRFailureError) as ctx:
                ocr_engine.run_ocr(self.image)
        self.assertIn("language", str(ctx.exception))

    def test_timeout_is_wrapped(self):
        a, b = self._patched(RuntimeError("Tesseract process timeout"))
        with a, b:
            with self.assertRaises(OCRFailureError) as ctx:
                ocr_engine.run_ocr(self.image)
        self.assertIn("too long", str(ctx.exception))

    def test_executable_vanishing_is_mapped_to_not_found(self):
        a, b = self._patched(pytesseract.TesseractNotFoundError())
        with a, b:
            with self.assertRaises(TesseractNotFoundError):
                ocr_engine.run_ocr(self.image)


@unittest.skipUnless(tesseract_available(), "Tesseract is not installed")
class RealTesseractTests(unittest.TestCase):
    """Integration tests: these call the real Tesseract executable."""

    def test_version_and_languages(self):
        self.assertTrue(ocr_engine.get_tesseract_version())
        self.assertIn("eng", ocr_engine.list_languages())

    def test_reads_known_text_with_real_confidences(self):
        from src.preprocessing import preprocess

        binary = preprocess(text_image_bgr()).binary
        words = ocr_engine.run_ocr(binary, psm=7)
        self.assertEqual(" ".join(w.text for w in words), "Welcome to DecodeLabs")
        self.assertTrue(all(0 <= w.confidence <= 100 for w in words))

    def test_blank_image_returns_no_words(self):
        self.assertEqual(ocr_engine.run_ocr(np.full((200, 400), 255, np.uint8)), [])

    def test_all_supported_psm_modes_run(self):
        from src import config
        from src.preprocessing import preprocess

        binary = preprocess(text_image_bgr()).binary
        for psm in config.PSM_MODES:
            self.assertIsInstance(ocr_engine.run_ocr(binary, psm=psm), list)


if __name__ == "__main__":
    unittest.main()
