# AI Image & Text Recognition

A clean Python application that reads text from images using **OpenCV preprocessing** and **Tesseract OCR (pytesseract)**, reports the *real* OCR confidence for every word, and validates the result against the **80% minimum confidence** standard from the DecodeLabs *Artificial Intelligence - Project 4: Image or Text Recognition (Basic)* brief.

> Built as an internship submission. The OCR path (Path 1 of the brief) is fully implemented. The optional object-detection path (Path 2) is **not** implemented; see [Future Improvements](#future-improvements).

## Project Overview

AI Vision Lab lets you upload an image, watch it pass through a visible preprocessing pipeline (grayscale, Gaussian blur, deskew, adaptive threshold), run OCR, and see the extracted text with a confidence score and a clear **PASS / BELOW THRESHOLD** result.

## Problem Statement

Machines cannot "read" pixels directly. A photo of text is just a grid of numbers, and raw images are often noisy, unevenly lit or tilted. OCR engines also always return *something*, so without a confidence check a wrong guess looks identical to a correct reading.

## Objective

Integrate a pre-trained recognition engine (no model training) into a working Python pipeline that:

1. accepts an image, 2. preprocesses it, 3. recognises text, 4. attaches a real confidence value to each result, 5. enforces the 80% minimum, and 6. shows the evidence visually.

## Features

- Upload PNG, JPG, JPEG, BMP, TIFF or WEBP images (or pick a bundled sample)
- Visible stages: original, grayscale, blurred, thresholded, plus an overlay of recognised word boxes
- Pipeline strip showing all seven stages with real parameters after each run
- Per-word confidence from Tesseract (`image_to_data`); nothing is estimated or invented
- Validation panel: average confidence, required 80%, progress bar with 80% marker, PASS / BELOW THRESHOLD / NO TEXT DETECTED
- Words below 80% are dropped from the final text but remain visible in a word-level table
- Page segmentation modes 3, 6, 7, 11; language selector; threshold method selector; blur control
- Safe deskewing, handling of light-text-on-dark images, upscaling of small images
- Download extracted text (.txt) and the processed image; copy text with the built-in copy button
- Reset to process another image
- Clear, non-technical error messages; full details go to `outputs/app.log`
- A command-line runner (`scripts/run_ocr.py`)

## Technology Stack

| Purpose | Library |
|---|---|
| Language | Python 3.11+ |
| Image processing | OpenCV (`opencv-python-headless`), NumPy, Pillow |
| OCR | Tesseract OCR engine via `pytesseract` |
| UI | Streamlit |
| Configuration | `python-dotenv` |
| Tests | `unittest` (also runs under `pytest`) |

## Architecture

```
app.py (Streamlit layout + state only)
   |
   v
src/pipeline.py        run_pipeline() - the single entry point
   |-- image_utils.py     validate, decode (Pillow -> OpenCV BGR), helpers
   |-- preprocessing.py   grayscale -> blur -> deskew -> threshold
   |-- ocr_engine.py      Tesseract discovery, image_to_data, text rebuild
   |-- confidence.py      80% word filter + validation rule
src/ui_components.py   HTML builders (no Streamlit import, unit-tested)
src/config.py          constants, paths, Tesseract discovery, logging
```

The `src/` package never imports Streamlit, so the UI, the CLI and the tests all call the same pipeline. See [docs/architecture.md](docs/architecture.md).

## Recognition Pipeline

```
Original Image -> Grayscale -> Gaussian Blur -> Adaptive Threshold
              -> OCR -> Confidence Validation -> Final Text
```

(Deskewing runs between blur and threshold when a tilt is detected.)

## OCR Workflow

1. Validate the upload (exists, supported extension, size, decodable, sensible dimensions).
2. Preprocess (below).
3. `pytesseract.image_to_data` with the chosen page mode returns every word with its box and confidence. Entries with confidence `-1` (layout elements, not words) are discarded.
4. Words at or above 80% are kept for the final text; the rest are dropped.
5. Text is rebuilt in reading order (lines and paragraphs preserved).
6. The 80% validation rule is applied (next sections).

## Image Preprocessing

| Stage | OpenCV call | Purpose | Default |
|---|---|---|---|
| Size normalisation | `cv2.resize` | Enlarge small images 2x (longest side < 1000 px); reduce images above 3500 px | on |
| Grayscale | `cv2.cvtColor` | Discard colour; OCR needs only intensity | always |
| Polarity fix | `cv2.bitwise_not` | If the background is dark (median < 127), invert so text is dark on light | on |
| Gaussian blur | `cv2.GaussianBlur` | Suppress noise before thresholding | 3x3 kernel |
| Deskew | projection-profile search + `cv2.warpAffine` | Straighten tilted text (up to 15 degrees); skipped when the gain is negligible | on |
| Threshold | `cv2.adaptiveThreshold` | Force each pixel to black or white using a *local* average, so shadows do not matter | Gaussian, block size auto (about 3% of the image, odd, 15-101), C = 10 |

**About thresholding.** The brief's slide is titled "Adaptive Thresholding" but its worked example is a single global cutoff (Otsu's method). This project defaults to `cv2.adaptiveThreshold` (the method the milestone names) and offers Otsu as an option so both can be compared. See the measured results below.

**Deskew safety.** The skew angle is estimated on a local-threshold text mask (a global mask was found to be fooled by shadow gradients, and a regression test covers this). Angles below 0.5 degrees, or rotations that do not clearly improve line alignment, are ignored, so straight images are never touched.

## Confidence Validation

Tesseract reports a 0-100 certainty for each word. Two separate rules are applied, deliberately:

- **Word filter (brief, slide 16):** a word with confidence below 80 is dropped from the final text.
- **Validation (brief, slide 17):** the headline *average confidence* is the mean over **all** recognised words, before filtering. Averaging only the survivors would always give 80% or more, which would be a false pass.

| Condition | Status |
|---|---|
| Average confidence >= 80% | **PASS** |
| Average confidence < 80% | **BELOW THRESHOLD** |
| No words recognised | **NO TEXT DETECTED** (no score is shown; it is never reported as 0% or as a pass) |

The comparison uses the exact value, and the displayed percentage is truncated, never rounded up, so 79.96% cannot appear as "80.0%" beside a failing result.

**Limitation:** average confidence measures how sure Tesseract is about the words it found, not whether it found *all* the text. A method that recognises fewer words can still show a high average. The word counts are shown next to the score for this reason.

### Measured results on the bundled samples

These are **synthetic** images rendered with Pillow (`scripts/generate_samples.py`). They demonstrate behaviour; they are *not* a measure of real-world accuracy. Figures are from actual runs on Tesseract 5.3.4.

| Sample | Adaptive Gaussian (default) | Otsu |
|---|---|---|
| welcome_clean.png | 92.0% (3/3 words) - PASS | 93.3% (3/3) - PASS |
| paragraph_clean.png | 93.9% (18/19) - PASS | 95.4% (19/19) - PASS |
| text_skewed.png (6 degree tilt) | 95.3% (19/19) - PASS | 95.3% (19/19) - PASS |
| light_on_dark.png | 94.3% (19/19) - PASS | 95.3% (19/19) - PASS |
| invoice_sparse.png (use PSM 11 or 3) | 94.8% (12/12) - PASS | 95.1% (12/12) - PASS |
| document_noisy_shadow.png (strong shading) | 88.6% (16/20) - PASS, 20 words found | 88.8% (12/14) - PASS, only 14 words found |
| low_contrast_blurred.png | 9.1% (0/10) - **BELOW THRESHOLD** | 84.2% (12/16) - PASS |
| blank_page.png | no text | no text |

What this shows: adaptive thresholding recovers more of the text under strong uneven lighting; Otsu works better on the faint, evenly lit image; and a poor reading is reported honestly as BELOW THRESHOLD rather than passed.

## Project Structure

```
project-4-ai-vision/
├── app.py                  Streamlit interface
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example            optional TESSERACT_CMD setting
├── .streamlit/config.toml  theme and upload limit
├── src/
│   ├── config.py           constants, Tesseract discovery, logging
│   ├── exceptions.py       user-presentable error types
│   ├── image_utils.py      input validation and decoding
│   ├── preprocessing.py    grayscale, blur, deskew, threshold
│   ├── ocr_engine.py       pytesseract wrapper, per-word data
│   ├── confidence.py       80% filter and validation
│   ├── pipeline.py         end-to-end run + result object
│   ├── ui_components.py    HTML builders for the UI
│   └── style.css           design tokens and components
├── scripts/
│   ├── generate_samples.py synthetic sample images
│   └── run_ocr.py          command-line runner
├── assets/samples/         bundled synthetic samples
├── outputs/                logs and exports (git-ignored)
├── tests/
└── docs/
    ├── architecture.md
    └── presentation_notes.md
```

## Installation

Requires **Python 3.11 or newer** and the **Tesseract OCR** program (a separate install, not a pip package).

```bash
git clone <your-repository-url>
cd project-4-ai-vision
```

## Windows Tesseract Setup

1. Download the Windows installer from the UB Mannheim Tesseract page (search "Tesseract at UB Mannheim") and run it. Keep the default install folder, `C:\Program Files\Tesseract-OCR`. Select any extra languages you need during setup.
2. **Option A - PATH (recommended):** add `C:\Program Files\Tesseract-OCR` to your system `PATH`, then open a **new** terminal and run `tesseract --version`.
3. **Option B - `.env` file:** if you prefer not to change PATH, copy `.env.example` to `.env` and set:
   ```
   TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
   ```
4. The app looks for Tesseract in this order: `TESSERACT_CMD`, then `PATH`, then the standard `Program Files` folders. If `TESSERACT_CMD` is set but wrong, the app reports Tesseract as missing instead of silently using another copy.

The path is read from your machine at run time; nothing personal is stored in the code, and `.env` is git-ignored.

## Virtual Environment Setup

Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
If activation is blocked, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use `.venv\Scripts\activate.bat` in Command Prompt.

macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run app.py
```
Your browser opens at `http://localhost:8501`. Command-line alternative:
```bash
python scripts/run_ocr.py assets/samples/welcome_clean.png --psm 7
```
(exit code 0 = validated, 2 = below threshold or no text, 1 = error).

## Usage Instructions

1. Upload an image that contains text, or choose a bundled sample.
2. Check the preview and the image summary.
3. (Optional) adjust settings in the sidebar. Use PSM 6 for a block of text, 7 for one line, 11 for scattered text.
4. Select **Run recognition**.
5. Read the **Confidence validation** panel first, then the extracted text.
6. Review the preprocessing stages and the recognition overlay (green = kept, red = dropped).
7. Download the text or processed image, or select **Reset** to start again.

If the result shows BELOW THRESHOLD, try another threshold method or page mode, or use a sharper, better-lit image.

## Testing

```bash
python -m unittest discover -s tests -t . -v
# or, with pytest installed:
python -m pytest
```
The suite covers image loading and validation, grayscale, blur, thresholding, deskew (including a shadow-gradient regression), polarity handling, OCR output parsing, confidence maths and boundaries, the no-false-pass rule, empty and low-confidence output, error paths, the end-to-end pipeline on every sample, HTML escaping, and an app smoke test.

- Tests that call the real Tesseract are skipped automatically if it is not installed.
- `tests/test_app_smoke.py` uses a stand-in for Streamlit to check the app's control flow. It does **not** replace a real `streamlit run app.py` check.

## Common Errors

| Message | Cause | Fix |
|---|---|---|
| "Tesseract OCR could not be found" | Tesseract not installed or not discoverable | Follow *Windows Tesseract Setup* |
| "Files with .xyz are not supported" | Unsupported extension | Convert to PNG/JPG/BMP/TIFF/WEBP |
| "The image could not be read" | Corrupted or fake image file | Re-export the image |
| "The image is too small / too large" | Outside 16-8000 px per side | Resize the image |
| "Tesseract could not process the image. Check that the selected language is installed." | Language data missing | Install the language in Tesseract, or choose English |
| "Text recognition took too long" | Very large or complex image | Use a smaller image or PSM 6/7 |
| NO TEXT DETECTED | Nothing legible found | Check orientation, contrast and resolution |
| BELOW THRESHOLD | Average confidence under 80% | Try another threshold/PSM; improve the image |

## Troubleshooting

- **`streamlit` is not recognised:** activate the virtual environment, then `pip install -r requirements.txt`.
- **Tesseract installed but not found:** open a new terminal after editing PATH, or set `TESSERACT_CMD` in `.env`.
- **Poor results on photos:** photograph straight on in even light; handwriting and stylised fonts are weak spots for Tesseract.
- **Logs:** `outputs/app.log` holds details for any unexpected failure.

## Future Improvements

- **Object detection tab (Path 2 of the brief):** `cv2.dnn` with MobileNet-SSD, `blobFromImage` (300x300), bounding boxes, labels and an 80% confidence gate. Not implemented here because it needs separate model files and could not be verified in the development environment. The pipeline returns plain result objects, so a detector can be added as a second module and tab without touching OCR.
- PDF and multi-page input; batch processing
- Optional contrast enhancement (CLAHE) for faint scans
- Region-of-interest selection; export to searchable PDF
- Evaluation against a labelled real-world dataset

## Internship Project Requirements Mapping

Status key: **Complete** = implemented and covered by executed tests; **Complete (run locally)** = implemented, logic tested, but real Streamlit rendering must be confirmed on your machine; **Not implemented** = optional item left out.

| PDF requirement (slide) | Implementation | Where | Status |
|---|---|---|---|
| Use a pre-trained model or simple library (3) | Tesseract OCR via pytesseract; no training | `ocr_engine.py` | Complete |
| Perform recognition on sample input (3) | Upload or bundled samples | `app.py`, `assets/samples/` | Complete (run locally) |
| Display output clearly (3, 17.4) | Text panel, validation panel, downloads | `ui_components.py`, `app.py` | Complete (run locally) |
| Python script that ingests raw visual data (6) | `run_pipeline()` | `pipeline.py` | Complete |
| Image as an H x W x C array (7) | Pillow decode to NumPy BGR; shape/dtype shown | `image_utils.py` | Complete |
| pytesseract integration, error-free (9, 17.1) | Discovery, config, wrapped errors | `config.py`, `ocr_engine.py` | Complete |
| OpenCV preprocessing (9, 11) | Full pipeline | `preprocessing.py` | Complete |
| Grayscale (11, 17.2) | `cv2.cvtColor` | `preprocessing.py` | Complete |
| Gaussian blur (11) | `cv2.GaussianBlur`, adjustable | `preprocessing.py` | Complete |
| Deskewing (11) | Projection-profile estimate + guarded rotation | `preprocessing.py` | Complete |
| Adaptive thresholding (12, 17.2) | `cv2.adaptiveThreshold` (+ Otsu option) | `preprocessing.py` | Complete |
| PSM modes 3 / 6 / 7 / 11 (10) | Sidebar selector | `config.py`, `ocr_engine.py` | Complete |
| Formatted text output (9) | Lines and paragraphs rebuilt in order | `ocr_engine.py` | Complete |
| Confidence on every output (15) | Per-word values from `image_to_data` | `ocr_engine.py` | Complete |
| 80% gate: keep or drop (16) | `confidence >= 80` word filter | `confidence.py` | Complete |
| Minimum validated confidence 80% (17.3) | Mean over all words; PASS/BELOW/NO TEXT | `confidence.py` | Complete |
| Visual confirmation (17.4) | Original + stages + overlay + text + score + status | `ui_components.py`, `app.py` | Complete (run locally) |
| Object detection: cv2.dnn, MobileNet-SSD, blobFromImage, boxes (9, 13-14) | Optional Path 2 | - | Not implemented |

## Repository

Suggested name: **AI-Image-Text-Recognition**

Suggested description: *Python OCR application using OpenCV preprocessing (grayscale, Gaussian blur, adaptive thresholding, deskew) and Tesseract, with per-word confidence and an 80% validation gate. Streamlit interface.*
