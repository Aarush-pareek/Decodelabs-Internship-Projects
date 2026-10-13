import unittest

from src import config
from src.confidence import (Status, average_confidence, format_confidence,
                            split_by_confidence, validate)
from src.ocr_engine import WordResult


def word(text, conf):
    return WordResult(text, conf, 0, 0, 10, 10)


class SplitTests(unittest.TestCase):
    def test_boundary_values(self):
        words = [word("a", 79.9), word("b", 80.0), word("c", 80.1)]
        kept, dropped = split_by_confidence(words)
        self.assertEqual([w.text for w in kept], ["b", "c"])
        self.assertEqual([w.text for w in dropped], ["a"])

    def test_empty_input(self):
        self.assertEqual(split_by_confidence([]), ([], []))


class AverageTests(unittest.TestCase):
    def test_mean(self):
        self.assertEqual(average_confidence([word("a", 90), word("b", 70)]), 80.0)

    def test_empty_is_none_not_zero(self):
        self.assertIsNone(average_confidence([]))


class FormatTests(unittest.TestCase):
    def test_truncates_instead_of_rounding_up(self):
        self.assertEqual(format_confidence(79.96), "79.9%")
        self.assertEqual(format_confidence(92.45), "92.4%")
        self.assertEqual(format_confidence(80.0), "80.0%")

    def test_none(self):
        self.assertEqual(format_confidence(None), "n/a")


class ValidateTests(unittest.TestCase):
    def test_high_confidence_validated(self):
        result = validate([word("a", 95), word("b", 90)])
        self.assertEqual(result.status, Status.VALIDATED)
        self.assertTrue(result.passed)
        self.assertEqual(result.average, 92.5)
        self.assertEqual((result.total_words, result.kept_words, result.dropped_words), (2, 2, 0))

    def test_exactly_at_threshold_passes(self):
        self.assertEqual(validate([word("a", 80), word("b", 80)]).status, Status.VALIDATED)

    def test_just_below_threshold_fails(self):
        result = validate([word("a", 79.9)])
        self.assertEqual(result.status, Status.BELOW_THRESHOLD)
        self.assertFalse(result.passed)

    def test_low_confidence_output_fails_with_zero_kept(self):
        result = validate([word("a", 30), word("b", 45)])
        self.assertEqual(result.status, Status.BELOW_THRESHOLD)
        self.assertEqual(result.kept_words, 0)
        self.assertIn("below", result.message)

    def test_no_false_pass_from_filtered_average(self):
        """Mixed words: survivors average >= 80, but the full average does not."""
        words = [word("a", 95), word("b", 92), word("c", 20), word("d", 15)]
        kept, _ = split_by_confidence(words)
        self.assertGreaterEqual(average_confidence(kept), 80)  # the trap
        self.assertEqual(validate(words).status, Status.BELOW_THRESHOLD)

    def test_empty_ocr_output_is_no_text_not_pass_or_zero(self):
        result = validate([])
        self.assertEqual(result.status, Status.NO_TEXT)
        self.assertIsNone(result.average)
        self.assertFalse(result.passed)

    def test_min_max_reported(self):
        result = validate([word("a", 60), word("b", 100)])
        self.assertEqual((result.min_confidence, result.max_confidence), (60, 100))

    def test_uses_project_threshold_by_default(self):
        self.assertEqual(validate([word("a", 90)]).required, config.CONFIDENCE_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
