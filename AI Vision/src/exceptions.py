"""Project-specific exceptions.

The UI catches these and shows the human-readable message. Anything that is
not a VisionError is logged in full and shown as a generic failure message.
"""


class VisionError(Exception):
    """Base class for all expected, user-presentable errors."""


class InvalidImageError(VisionError):
    """The input is missing, corrupted, empty or otherwise unusable."""


class UnsupportedFormatError(InvalidImageError):
    """The file extension is not one of the supported image formats."""


class PreprocessingError(VisionError):
    """An OpenCV preprocessing step failed."""


class TesseractNotFoundError(VisionError):
    """The Tesseract executable could not be located."""


class OCRFailureError(VisionError):
    """Tesseract was found but the recognition call itself failed."""
