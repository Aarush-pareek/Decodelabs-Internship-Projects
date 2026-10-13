import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src import config
from src.confidence import Status
from src.exceptions import InvalidImageError, TesseractNotFoundError, UnsupportedFormatError
from src.pipeline import PipelineSettings, run_pipeline, run_pipeline_on_array
from src.preprocessing import PreprocessSettings
from tests.helpers import tesseract_available, text_image_bgr, to_bytes

SAMPLES = Path(config.SAMPLES_DIR)


def sample(name):
    return (SAMPLES / name).read_bytes()


class InputErrorTests(unittest.TestCase):
    def test_no_image(self):
        with self.assertRaises(InvalidImageError):
            run_pipeline(b"", "x.png")

    def test_unsupported_type(self):
        with self.assertRaises(UnsupportedFormatError):
            run_pipeline(b"abc", "x.gif")

    def test_corrupted_image(self):
        with self.assertRaises(InvalidImageError):
            run_pipeline(b"not an image", "x.png")

    def test_tesseract_missing(self):
        with mock.patch("src.ocr_engine.config.find_tesseract_cmd", return_value=None):
            with self.assertRaises(TesseractNotFoundError):
                run_pipeline_on_array(text_image_bgr())


@unittest.skipUnless(tesseract_available(), "Tesseract is not installed")
class EndToEndTests(unittest.TestCase):
    """Run the real pipeline on the bundled SYNTHETIC samples."""

    def test_clean_text_is_validated_and_correct(self):
        result = run_pipeline(sample("welcome_clean.png"), "welcome_clean.png")
        self.assertEqual(result.text, "Welcome to DecodeLabs")
        self.assertEqual(result.validation.status, Status.VALIDATED)
        self.assertGreaterEqual(result.validation.average, 80)

    def test_result_exposes_every_preprocessing_stage(self):
        result = run_pipeline(sample("paragraph_clean.png"), "p.png")
        prep = result.preprocessing
        for image in (prep.original, prep.grayscale, prep.blurred, prep.binary):
            self.assertIsInstance(image, np.ndarray)
        self.assertEqual(result.annotated.ndim, 3)
        self.assertGreater(result.timings_ms["total"], 0)

    def test_skewed_sample_is_deskewed_and_read(self):
        result = run_pipeline(sample("text_skewed.png"), "s.png")
        self.assertLess(result.preprocessing.skew_angle, -4.0)
        self.assertEqual(result.validation.status, Status.VALIDATED)
        self.assertIn("Optical character recognition", result.text)

    def test_inverted_sample_is_handled(self):
        result = run_pipeline(sample("light_on_dark.png"), "d.png")
        self.assertTrue(result.preprocessing.inverted)
        self.assertEqual(result.validation.status, Status.VALIDATED)

    def test_strong_shadow_sample_reads_with_adaptive_threshold(self):
        result = run_pipeline(sample("document_noisy_shadow.png"), "n.png")
        self.assertEqual(result.validation.status, Status.VALIDATED)
        self.assertGreaterEqual(result.validation.total_words, 18)

    def test_sparse_invoice_with_psm_11(self):
        result = run_pipeline(sample("invoice_sparse.png"), "i.png", PipelineSettings(psm=11))
        self.assertIn("10234", result.text)
        self.assertEqual(result.validation.status, Status.VALIDATED)

    def test_blank_page_reports_no_text(self):
        result = run_pipeline(sample("blank_page.png"), "b.png")
        self.assertEqual(result.validation.status, Status.NO_TEXT)
        self.assertEqual(result.text, "")
        self.assertIsNone(result.validation.average)

    def test_low_quality_sample_is_not_falsely_validated(self):
        result = run_pipeline(sample("low_contrast_blurred.png"), "l.png")
        self.assertEqual(result.validation.status, Status.BELOW_THRESHOLD)
        self.assertLess(result.validation.average, 80)
        self.assertEqual(result.text, "")  # nothing reached the 80% word filter

    def test_otsu_recovers_the_low_contrast_sample(self):
        settings = PipelineSettings(preprocess=PreprocessSettings(threshold_method="otsu"))
        result = run_pipeline(sample("low_contrast_blurred.png"), "l.png", settings)
        self.assertGreater(result.validation.average, 70)

    def test_final_text_contains_only_words_at_or_above_threshold(self):
        result = run_pipeline(sample("paragraph_clean.png"), "p.png")
        self.assertTrue(all(w.confidence >= 80 for w in result.kept_words))
        self.assertTrue(all(w.confidence < 80 for w in result.dropped_words))
        self.assertEqual(len(result.words), len(result.kept_words) + len(result.dropped_words))

    def test_jpeg_upload_works(self):
        data = to_bytes(text_image_bgr(), ".jpg")
        result = run_pipeline(data, "photo.jpg")
        self.assertIn("Welcome", result.text)


if __name__ == "__main__":
    unittest.main()
