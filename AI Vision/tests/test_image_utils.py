import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src import config, image_utils
from src.exceptions import InvalidImageError, UnsupportedFormatError
from tests.helpers import pil_bytes, text_image_bgr, to_bytes


class ValidateUploadTests(unittest.TestCase):
    def test_none_and_empty_bytes_rejected(self):
        for data in (None, b""):
            with self.assertRaises(InvalidImageError):
                image_utils.validate_upload("a.png", data)

    def test_unsupported_extension_rejected(self):
        with self.assertRaises(UnsupportedFormatError) as ctx:
            image_utils.validate_upload("notes.pdf", b"123")
        self.assertIn("PNG", str(ctx.exception))

    def test_missing_extension_rejected(self):
        with self.assertRaises(UnsupportedFormatError):
            image_utils.validate_upload("image", b"123")

    def test_oversized_file_rejected(self):
        with self.assertRaises(InvalidImageError):
            image_utils.validate_upload("a.png", b"0" * (config.MAX_FILE_BYTES + 1))

    def test_valid_file_accepted(self):
        image_utils.validate_upload("A.PNG", b"123")  # case-insensitive extension


class DecodeImageTests(unittest.TestCase):
    def test_decodes_png_to_bgr_uint8(self):
        array = image_utils.decode_image(to_bytes(text_image_bgr()))
        self.assertEqual(array.dtype, np.uint8)
        self.assertEqual(array.ndim, 3)
        self.assertEqual(array.shape[2], 3)

    def test_corrupted_bytes_raise_invalid_image(self):
        with self.assertRaises(InvalidImageError):
            image_utils.decode_image(b"this is not an image at all")

    def test_truncated_png_raises_invalid_image(self):
        data = to_bytes(text_image_bgr())
        with self.assertRaises(InvalidImageError):
            image_utils.decode_image(data[: len(data) // 3])

    def test_transparent_png_is_composited_on_white(self):
        rgba = Image.new("RGBA", (64, 64), (0, 0, 0, 0))  # fully transparent
        array = image_utils.decode_image(pil_bytes(rgba))
        self.assertTrue((array == 255).all())

    def test_grayscale_image_becomes_three_channel(self):
        array = image_utils.decode_image(pil_bytes(Image.new("L", (64, 64), 128)))
        self.assertEqual(array.shape, (64, 64, 3))

    def test_too_small_image_rejected(self):
        with self.assertRaises(InvalidImageError):
            image_utils.decode_image(pil_bytes(Image.new("RGB", (4, 4), (255, 255, 255))))

    def test_too_large_image_rejected_from_header(self):
        side = config.MAX_DIMENSION_PX + 1
        with self.assertRaises(InvalidImageError):
            image_utils.decode_image(pil_bytes(Image.new("1", (side, 20))))


class FileAndHelperTests(unittest.TestCase):
    def test_load_missing_file(self):
        with self.assertRaises(InvalidImageError):
            image_utils.load_image_file(Path(tempfile.gettempdir()) / "no_such_image.png")

    def test_load_real_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "t.png"
            path.write_bytes(to_bytes(text_image_bgr()))
            self.assertEqual(image_utils.load_image_file(path).ndim, 3)

    def test_image_info(self):
        info = image_utils.image_info(np.zeros((10, 20, 3), np.uint8))
        self.assertEqual(info, {"height": 10, "width": 20, "channels": 3, "dtype": "uint8"})

    def test_preview_shrinks_but_never_enlarges(self):
        big = np.zeros((2000, 1000, 3), np.uint8)
        small = np.zeros((100, 50, 3), np.uint8)
        self.assertEqual(max(image_utils.make_preview(big, 900).shape[:2]), 900)
        self.assertEqual(image_utils.make_preview(small, 900).shape, small.shape)

    def test_png_encoding_round_trip(self):
        original = text_image_bgr()
        decoded = image_utils.decode_image(image_utils.encode_png(original))
        np.testing.assert_array_equal(decoded, original)


if __name__ == "__main__":
    unittest.main()
