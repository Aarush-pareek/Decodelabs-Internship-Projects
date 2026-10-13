"""Tests for src/config.py (constants and Tesseract discovery)."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import config


class ConstantsTests(unittest.TestCase):
    def test_confidence_threshold_matches_brief(self):
        self.assertEqual(config.CONFIDENCE_THRESHOLD, 80.0)

    def test_psm_modes_match_brief(self):
        self.assertEqual(set(config.PSM_MODES), {3, 6, 7, 11})
        self.assertIn(config.DEFAULT_PSM, config.PSM_MODES)

    def test_odd_kernel_and_block_size(self):
        self.assertEqual(config.DEFAULT_BLUR_KERNEL % 2, 1)
        block = config.DEFAULT_ADAPTIVE_BLOCK_SIZE
        self.assertTrue(block is None or block % 2 == 1)

    def test_paths_are_inside_project(self):
        self.assertTrue(config.OUTPUT_DIR.is_relative_to(config.PROJECT_ROOT))
        self.assertTrue(config.SAMPLES_DIR.is_relative_to(config.PROJECT_ROOT))


class TesseractDiscoveryTests(unittest.TestCase):
    def test_env_var_pointing_to_real_file_is_used(self):
        with tempfile.NamedTemporaryFile(suffix=".exe") as fake:
            with mock.patch.dict(os.environ, {"TESSERACT_CMD": fake.name}):
                self.assertEqual(config.find_tesseract_cmd(), fake.name)

    def test_bad_env_var_is_reported_missing_not_ignored(self):
        bad = str(Path(tempfile.gettempdir()) / "definitely_not_tesseract.exe")
        with mock.patch.dict(os.environ, {"TESSERACT_CMD": bad}):
            self.assertIsNone(config.find_tesseract_cmd())

    def test_falls_back_to_path_lookup(self):
        with mock.patch.dict(os.environ, {"TESSERACT_CMD": ""}), mock.patch(
            "src.config.shutil.which", return_value="/usr/bin/tesseract"
        ):
            self.assertEqual(config.find_tesseract_cmd(), "/usr/bin/tesseract")

    def test_returns_none_when_nothing_found(self):
        with mock.patch.dict(os.environ, {"TESSERACT_CMD": ""}), mock.patch(
            "src.config.shutil.which", return_value=None
        ), mock.patch("src.config._windows_candidates", return_value=[]):
            self.assertIsNone(config.find_tesseract_cmd())


if __name__ == "__main__":
    unittest.main()
