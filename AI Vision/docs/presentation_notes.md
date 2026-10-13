# Presentation Notes

Plain-language answers you can say out loud. Numbers below come from real runs of this project on its bundled **synthetic** samples.

## 1. What is the problem?
A computer sees an image as a grid of numbers, not as words. The task is to turn a picture of text into real, usable text, and to say how sure we are about it.

## 2. Why OCR?
OCR (Optical Character Recognition) is the standard way to extract text from images, and the brief allows a "basic" recognition project built on a pre-trained tool. It is a clear, demonstrable pipeline with visible stages and a measurable confidence.

## 3. What is pytesseract?
A small Python wrapper that lets my code call **Tesseract**, an open-source OCR engine. Tesseract does the reading; pytesseract sends it the image and returns results (text, positions, confidence).

## 4. What is OpenCV?
A computer-vision library. I use it to prepare the image: grayscale, blur, straighten, and black-and-white conversion.

## 5. Why grayscale?
Text recognition depends on light versus dark, not colour. Grayscale reduces three colour channels to one, which removes irrelevant information and speeds processing.

## 6. Why Gaussian blur?
It smooths random speckle noise so that noise is not mistaken for tiny letters. I keep it light (3x3) because heavy blur would also smear real letters. The kernel can be changed or turned off.

## 7. Why adaptive thresholding?
It turns every pixel black or white, but decides using the *local* surroundings rather than one cutoff for the whole image. So a shadow on one side of a page does not wipe out the text there.
*Honest detail:* on my strongest-shadow sample, adaptive found 20 words while a single global cutoff (Otsu) found only 14. On an evenly lit faint image, Otsu did better (84.2% vs 9.1%). That is why the method is selectable.

## 8. What is OCR, in one line?
Software that finds characters in an image and converts them into text.

## 9. How does confidence work?
For each word, Tesseract gives a 0-100 score of how sure its neural network is. I read it from `image_to_data`; I never make up values. A high score means "very likely correct", not a guarantee.

## 10. Why is 80% used?
The brief sets it as the minimum standard. Without a gate, a system treats every guess as equally certain and can produce confident-looking wrong answers. A higher bar removes more false results but may also discard some correct ones; 80% is the required balance for this project.

## 11. What happens when confidence is below 80%?
- Words under 80% are removed from the final text.
- The status shows **BELOW THRESHOLD** (never a pass).
- The raw guesses stay visible in an expandable table so nothing is hidden.
- The user is told to try another page mode or threshold method, or a better image.
Example: my faint, blurred sample scores 9.1% with the default settings and is correctly reported as BELOW THRESHOLD.

## 12. What is the input?
An image file (PNG, JPG, JPEG, BMP, TIFF or WEBP, up to 20 MB) that contains printed text, uploaded or chosen from the bundled samples.

## 13. What is the processing pipeline?
Original -> Grayscale -> Gaussian blur -> (deskew if tilted) -> Adaptive threshold -> OCR -> Confidence validation -> Final text. The app shows each stage and its parameters.

## 14. What is the output?
The extracted text (words at 80% or above), the average confidence, the required 80%, a PASS / BELOW THRESHOLD / NO TEXT result, word counts, processing time, the preprocessing images, and a box overlay (green = kept, red = dropped). Text and the processed image can be downloaded.

## 15. Why a pre-trained engine?
The goal is to *use* a trained model, not train one. Training an OCR model needs huge labelled datasets and compute; Tesseract already provides trained models, so the effort goes into integration, preprocessing and validation, which is what the brief assesses.

## 16. How does the project satisfy the internship PDF?
It follows Path 1 (OCR): pytesseract + OpenCV, grayscale, blur, deskew and adaptive thresholding, real confidence values, the 80% gate, and visual confirmation of original, processed image, text, score and status. The full slide-by-slide table is at the end of the README.

---

## Questions an evaluator may ask

**Why average over all words and not only the kept ones?** The kept words are by definition at least 80%, so their average could never fail. That would be a false pass. I validate on all recognised words and show kept/dropped counts separately.

**Does a high average mean all the text was found?** No. It measures certainty on the words that were found. That is why the word counts are shown beside the score.

**Did you implement object detection?** No. It is the brief's optional second path. It needs separate model files and I could not verify it, so I did not include untested code. The README explains how it would be added.

**How accurate is it in the real world?** I have only measured it on synthetic images I generated, so I cannot claim a real-world accuracy. Tesseract works best on clear printed text and is weak on handwriting and decorative fonts.

**What did testing find?** The deskew step produced a false 15.8 degree rotation on a page with a strong shadow, because its first version used a global threshold. I fixed it by using a local threshold and added a regression test.

## Two-minute demo script

1. Open the app; point out the Tesseract-ready chip and the empty pipeline.
2. Choose `welcome_clean.png`, Run recognition: show the text, ~92% confidence vs 80% required, **PASS**.
3. Walk through the four preprocessing images and the green overlay boxes.
4. Choose `text_skewed.png`: show "deskew -6.0 deg" and that it still passes.
5. Choose `low_contrast_blurred.png`: show **BELOW THRESHOLD**, then switch the threshold method to Otsu and run again to show the improvement.
6. Download the text, then Reset.
