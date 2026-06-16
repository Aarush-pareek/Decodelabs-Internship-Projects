import pytesseract
from .preprocessing import preprocess_ocr

def extract_text(image_bytes: bytes):
    preprocessed = preprocess_ocr(image_bytes)
    data = pytesseract.image_to_data(preprocessed, output_type=pytesseract.Output.DICT)
    text = pytesseract.image_to_string(preprocessed).strip()
    confs = [int(conf) for conf in data['conf'] if conf != '-1']
    avg_conf = sum(confs) / len(confs) if confs else 0
    return {
        "text": text,
        "confidence": round(avg_conf, 2),
        "word_details": data
    }