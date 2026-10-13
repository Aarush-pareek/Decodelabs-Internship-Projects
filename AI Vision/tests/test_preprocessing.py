import unittest

import cv2
import numpy as np

from src import preprocessing as pp
from src.exceptions import PreprocessingError
from tests.helpers import text_image_bgr


def rotate_gray(gray, angle):
    h, w = gray.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, matrix, (w, h), borderValue=255)


class SizeAndGrayscaleTests(unittest.TestCase):
    def test_grayscale_is_single_channel_same_size(self):
        image = text_image_bgr()
        gray = pp.to_grayscale(image)
        self.assertEqual(gray.ndim, 2)
        self.assertEqual(gray.shape, image.shape[:2])
        self.assertEqual(gray.dtype, np.uint8)

    def test_grayscale_of_gray_input_is_copy(self):
        gray = np.full((20, 20), 7, np.uint8)
        out = pp.to_grayscale(gray)
        out[0, 0] = 99
        self.assertEqual(gray[0, 0], 7)

    def test_small_images_are_upscaled(self):
        small = np.full((200, 300, 3), 255, np.uint8)
        out, scale = pp.normalise_size(small, upscale=True)
        self.assertEqual(scale, 2.0)
        self.assertEqual(out.shape[:2], (400, 600))

    def test_upscale_can_be_disabled(self):
        small = np.full((200, 300, 3), 255, np.uint8)
        out, scale = pp.normalise_size(small, upscale=False)
        self.assertEqual((scale, out.shape[:2]), (1.0, (200, 300)))

    def test_huge_images_are_reduced(self):
        huge = np.full((3600, 4000, 3), 255, np.uint8)
        out, scale = pp.normalise_size(huge)
        self.assertLess(scale, 1.0)
        self.assertEqual(max(out.shape[:2]), 3500)


class BlurTests(unittest.TestCase):
    def test_blur_reduces_noise_variance(self):
        rng = np.random.default_rng(0)
        noisy = rng.integers(0, 255, (100, 100), dtype=np.uint8)
        self.assertLess(pp.apply_gaussian_blur(noisy, 5).var(), noisy.var())

    def test_kernel_one_or_zero_disables_blur(self):
        image = np.random.default_rng(1).integers(0, 255, (50, 50), dtype=np.uint8)
        for kernel in (0, 1):
            np.testing.assert_array_equal(pp.apply_gaussian_blur(image, kernel), image)


class ThresholdTests(unittest.TestCase):
    def setUp(self):
        self.gray = pp.to_grayscale(text_image_bgr())

    def test_adaptive_output_is_strictly_binary(self):
        for method in ("adaptive_gaussian", "adaptive_mean"):
            result = pp.apply_threshold(self.gray, method)
            self.assertEqual(set(np.unique(result.binary)), {0, 255})
            self.assertEqual(result.block_size % 2, 1)

    def test_otsu_returns_cutoff_in_range(self):
        result = pp.apply_threshold(self.gray, "otsu")
        self.assertEqual(set(np.unique(result.binary)), {0, 255})
        self.assertTrue(0 < result.otsu_cutoff < 255)

    def test_dark_text_on_light_background_stays_black_on_white(self):
        binary = pp.apply_threshold(self.gray).binary
        self.assertGreater((binary == 255).mean(), 0.5)

    def test_explicit_block_size_is_used(self):
        self.assertEqual(pp.apply_threshold(self.gray, block_size=21).block_size, 21)

    def test_unknown_method_raises(self):
        with self.assertRaises(PreprocessingError):
            pp.apply_threshold(self.gray, "magic")

    def test_auto_block_size_is_odd_and_clamped(self):
        for shape in ((50, 50), (800, 1000), (5000, 5000)):
            block = pp.auto_block_size(shape)
            self.assertEqual(block % 2, 1)
            self.assertTrue(15 <= block <= 101)


class SettingsTests(unittest.TestCase):
    def test_even_kernel_rejected(self):
        with self.assertRaises(PreprocessingError):
            pp.PreprocessSettings(blur_kernel=4)

    def test_even_block_size_rejected(self):
        with self.assertRaises(PreprocessingError):
            pp.PreprocessSettings(block_size=10)

    def test_unknown_method_rejected(self):
        with self.assertRaises(PreprocessingError):
            pp.PreprocessSettings(threshold_method="nope")


class PolarityTests(unittest.TestCase):
    def test_dark_background_detected(self):
        self.assertTrue(pp.needs_inversion(np.full((50, 50), 20, np.uint8)))
        self.assertFalse(pp.needs_inversion(np.full((50, 50), 240, np.uint8)))

    def test_light_on_dark_is_inverted_by_pipeline(self):
        image = 255 - text_image_bgr()
        result = pp.preprocess(image)
        self.assertTrue(result.inverted)
        self.assertGreater((result.binary == 255).mean(), 0.5)


class DeskewTests(unittest.TestCase):
    def setUp(self):
        self.gray = pp.to_grayscale(text_image_bgr(("Skew detection test line", "second line of text"), 40))

    def test_straight_image_is_left_untouched(self):
        self.assertEqual(pp.estimate_skew_angle(self.gray), 0.0)

    def test_blank_image_is_left_untouched(self):
        self.assertEqual(pp.estimate_skew_angle(np.full((300, 300), 255, np.uint8)), 0.0)

    def test_skew_is_detected_and_corrected(self):
        for applied in (6.0, -4.0):
            tilted = rotate_gray(self.gray, applied)
            angle = pp.estimate_skew_angle(tilted)
            self.assertAlmostEqual(angle, -applied, delta=1.0)

    def test_deskew_zero_returns_same_image(self):
        self.assertIs(pp.deskew(self.gray, 0.0), self.gray)

    def test_deskew_expands_canvas_without_cropping(self):
        out = pp.deskew(self.gray, 10.0)
        self.assertGreaterEqual(out.shape[0], self.gray.shape[0])

    def test_shadow_gradient_does_not_cause_false_skew(self):
        """Regression: a global threshold mistook a shadowed half-page for text."""
        ramp = np.linspace(1.0, 0.4, self.gray.shape[1], dtype=np.float32)
        shaded = (self.gray.astype(np.float32) * ramp[None, :]).astype(np.uint8)
        self.assertEqual(pp.estimate_skew_angle(shaded), 0.0)


class FullPreprocessTests(unittest.TestCase):
    def test_returns_all_stages_with_consistent_shapes(self):
        result = pp.preprocess(text_image_bgr())
        self.assertEqual(result.grayscale.ndim, 2)
        self.assertEqual(result.grayscale.shape, result.blurred.shape)
        self.assertEqual(set(np.unique(result.binary)), {0, 255})
        self.assertEqual(result.scale_factor, 2.0)

    def test_deskew_off_reports_zero_angle(self):
        settings = pp.PreprocessSettings(deskew=False)
        self.assertEqual(pp.preprocess(text_image_bgr(), settings).skew_angle, 0.0)


if __name__ == "__main__":
    unittest.main()
