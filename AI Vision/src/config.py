"""Central configuration: constants, paths, Tesseract discovery and logging.

Nothing in this module is machine-specific. The Tesseract location is resolved
at runtime from (in order): the TESSERACT_CMD environment variable / .env file,
the system PATH, then the standard Windows install folders.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------------
# Paths (all relative to the project root, never to a user's home directory)
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = PROJECT_ROOT / "assets" / "samples"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_FILE = OUTPUT_DIR / "app.log"

load_dotenv(PROJECT_ROOT / ".env")

# --------------------------------------------------------------------------
# Validation rule from the Project 4 brief (slides 16-17)
# --------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 80.0  # percent; Tesseract reports confidence on 0-100

# --------------------------------------------------------------------------
# Input rules
# --------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
MAX_FILE_BYTES = 20 * 1024 * 1024  # matches .streamlit/config.toml maxUploadSize
MIN_DIMENSION_PX = 16
MAX_DIMENSION_PX = 8000

# --------------------------------------------------------------------------
# OCR settings (PSM modes from slide 10 of the brief)
# --------------------------------------------------------------------------
PSM_MODES = {
    3: "Fully automatic - varied layouts (default)",
    6: "Single uniform block of text - book pages",
    7: "Single text line - plates, headers",
    11: "Sparse, scattered text - invoices",
}
DEFAULT_PSM = 3
DEFAULT_LANGUAGE = "eng"

# --------------------------------------------------------------------------
# Preprocessing defaults
# --------------------------------------------------------------------------
THRESHOLD_METHODS = ("adaptive_gaussian", "adaptive_mean", "otsu")
DEFAULT_THRESHOLD_METHOD = "adaptive_gaussian"
DEFAULT_BLUR_KERNEL = 3  # odd integer; 0 or 1 disables blur
DEFAULT_ADAPTIVE_BLOCK_SIZE = None  # None = derive from image size (see preprocessing)
DEFAULT_ADAPTIVE_C = 10  # constant subtracted from the local mean
UPSCALE_BELOW_PX = 1000  # images whose longest side is below this are enlarged
UPSCALE_FACTOR = 2.0
MAX_PROCESS_PX = 3500  # larger images are reduced to this longest side
DESKEW_MIN_ANGLE_DEG = 0.5  # below this the image is treated as straight
DESKEW_MAX_ANGLE_DEG = 15.0  # search range for the skew angle

# --------------------------------------------------------------------------
# Tesseract discovery
# --------------------------------------------------------------------------
_WINDOWS_DEFAULT_SUBPATH = Path("Tesseract-OCR") / "tesseract.exe"


def _windows_candidates() -> list[Path]:
    """Standard installer locations, built from system environment variables."""
    roots = [os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")]
    return [Path(root) / _WINDOWS_DEFAULT_SUBPATH for root in roots if root]


def find_tesseract_cmd() -> str | None:
    """Return the path to the Tesseract executable, or None if not found."""
    configured = os.environ.get("TESSERACT_CMD", "").strip().strip('"')
    if configured:
        # An explicit setting wins; if it is wrong we report it as missing
        # rather than silently falling back to a different installation.
        return configured if Path(configured).is_file() else None

    on_path = shutil.which("tesseract")
    if on_path:
        return on_path

    for candidate in _windows_candidates():
        if candidate.is_file():
            return str(candidate)
    return None


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure the 'aivision' logger once (console + outputs/app.log)."""
    logger = logging.getLogger("aivision")
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        logger.warning("Log file not writable; logging to console only.")

    logger.propagate = False
    return logger
