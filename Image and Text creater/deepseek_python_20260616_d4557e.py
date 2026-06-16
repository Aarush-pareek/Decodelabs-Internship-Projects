import cv2
import numpy as np

# Paths relative to backend/ai/
MODEL_PATH = "models/mobilenet_ssd/deploy.prototxt"
WEIGHTS_PATH = "models/mobilenet_ssd/mobilenet_iter_73000.caffemodel"

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

TARGET_CLASSES = {"person", "car", "bottle", "chair", "dog", "cat"}

net = cv2.dnn.readNetFromCaffe(MODEL_PATH, WEIGHTS_PATH)

def detect_objects(image_bytes: bytes, min_confidence=0.5):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843,
                                 (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    results = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > min_confidence:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]
            if label not in TARGET_CLASSES:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            results.append({
                "label": label,
                "confidence": round(float(confidence) * 100, 2),
                "bbox": [int(startX), int(startY), int(endX), int(endY)]
            })
    return results