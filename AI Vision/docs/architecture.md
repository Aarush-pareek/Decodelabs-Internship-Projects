# Architecture

## Goals

- Recognition logic is independent of the UI (testable, reusable from a CLI).
- Every confidence value comes from Tesseract; nothing is estimated.
- Failures are expected and presented as plain messages, never stack traces.
- Easy to extend (for example with an object-detection module) without touching OCR.

## Data flow

```
bytes + filename
   -> image_utils.validate_upload      extension, size, non-empty
   -> image_utils.decode_image         Pillow decode, EXIF rotation, alpha -> white, BGR uint8
   -> preprocessing.preprocess         PreprocessResult (all intermediate images)
        normalise_size -> to_grayscale -> [invert] -> apply_gaussian_blur
        -> estimate_skew_angle/deskew -> apply_threshold
   -> ocr_engine.run_ocr               list[WordResult] (text, confidence 0-100, box, line ids)
   -> confidence.split_by_confidence   kept / dropped at >= 80
   -> confidence.validate              ValidationResult (status, average, counts)
   -> pipeline.draw_word_boxes         annotated image
   -> PipelineResult                   everything the UI needs
```

## Modules

| Module | Responsibility |
|---|---|
| `config.py` | Constants (80% threshold, PSM modes, limits), project paths, Tesseract discovery, logging |
| `exceptions.py` | `VisionError` family: the only errors shown to users verbatim |
| `image_utils.py` | Upload validation, decoding, preview/PNG helpers |
| `preprocessing.py` | Pure functions per stage plus `preprocess()` and `PreprocessSettings` |
| `ocr_engine.py` | Tesseract configuration, `image_to_data` call, parsing, text rebuild |
| `confidence.py` | Word filter, average, `Status`, `ValidationResult` |
| `pipeline.py` | `run_pipeline()` / `run_pipeline_on_array()`, `PipelineResult` |
| `ui_components.py` | HTML string builders (escape all dynamic text); no Streamlit import |
| `app.py` | Layout, widgets, session state; calls only `run_pipeline()` |

## Key design decisions

1. **Validation uses all words.** The 80% word filter (slide 16) and the 80% validation (slide 17) are separate. The headline average is taken before filtering; otherwise the average of the survivors would always be at least 80% (a false pass).
2. **No-text is its own state.** Zero recognised words gives `NO_TEXT` with no score; never 0% and never a pass.
3. **Adaptive threshold by default, Otsu optional.** The brief names adaptive thresholding but illustrates a global cutoff; both are available and compared in the README.
4. **Local threshold inside the skew estimator.** A global threshold treated a shadowed half-page as text and produced a false 15.8 degree rotation (found during testing, covered by a regression test).
5. **Polarity handling.** Adaptive thresholding assumes dark text on a light background; dark-background images are inverted first.
6. **Adaptive block size scales with the image** (about 3% of the longest side, odd, 15-101); a fixed small block would hollow out thick strokes after upscaling.
7. **Pillow for decoding.** Handles EXIF orientation, transparency and 16-bit images; dimensions are checked from the header before the full decode.
8. **Settings are frozen dataclasses**, so the UI can detect that a stored result was produced with different settings.
9. **Security:** no secrets or personal paths; `TESSERACT_CMD` comes from the environment or `.env` (git-ignored); the OCR language string is validated before it reaches the Tesseract command line.

## Error handling

| Situation | Exception | Shown as |
|---|---|---|
| No / empty file | `InvalidImageError` | "The file is empty..." |
| Wrong extension | `UnsupportedFormatError` | Lists supported formats |
| Corrupt / unreadable / bad size | `InvalidImageError` | Plain explanation |
| Tesseract missing or wrong path | `TesseractNotFoundError` | Setup steps |
| Language missing, crash, timeout | `OCRFailureError` | Plain explanation + hint |
| OpenCV failure | `PreprocessingError` | "Image preprocessing failed" |
| Anything else | (logged with traceback) | Generic message; details in `outputs/app.log` |

## Extending with object detection (not implemented)

Add `src/detection_engine.py` returning a list of `(label, confidence, box)` objects, reuse the same 80% gate, and add a second tab in `app.py`. Model files (MobileNet-SSD prototxt and caffemodel) would live in a git-ignored `models/` folder, fetched by a small download script. The OCR modules do not need to change.

## Testing strategy

- Pure-logic modules (confidence, preprocessing, parsing) have direct unit tests, including boundaries (79.9 / 80.0) and failure inputs.
- Tesseract failure paths are tested with mocks; real-Tesseract tests are skipped automatically when it is absent.
- End-to-end tests run the pipeline on the bundled synthetic samples.
- `test_app_smoke.py` drives `app.main()` with a stand-in for Streamlit to check control flow. It does not validate Streamlit's own rendering.
