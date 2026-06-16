import base64
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ai.ocr_engine import extract_text
from ai.object_detection_engine import detect_objects
from database import init_db, save_history, get_all_history

app = FastAPI(title="AI Vision Suite API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    await init_db()

def image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")

@app.post("/api/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only image files are accepted")
    contents = await file.read()
    result = extract_text(contents)
    confidence = result["confidence"]
    if confidence < 80:
        result["warning"] = "Low confidence – manual review recommended."
    else:
        result["warning"] = None
    # Save to history
    await save_history("ocr", file.filename, result["text"], confidence)
    return {
        "text": result["text"],
        "confidence": confidence,
        "warning": result["warning"]
    }

@app.post("/api/object-detection")
async def detection_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only image files are accepted")
    contents = await file.read()
    detections = detect_objects(contents)
    avg_conf = sum(d["confidence"] for d in detections) / len(detections) if detections else 0
    # Draw bounding boxes on image
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    for d in detections:
        x1,y1,x2,y2 = d["bbox"]
        cv2.rectangle(img, (x1,y1), (x2,y2), (99,102,241), 2)  # Indigo color
        cv2.putText(img, f"{d['label']} {d['confidence']:.1f}%", (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (99,102,241), 2)
    _, encoded = cv2.imencode(".jpg", img)
    annotated_image = base64.b64encode(encoded).decode()
    warning = None
    if any(d["confidence"] < 80 for d in detections):
        warning = "Some detections have low confidence."
    await save_history("object_detection", file.filename, str(detections), avg_conf)
    return {
        "detections": detections,
        "annotated_image": annotated_image,
        "average_confidence": round(avg_conf, 2),
        "warning": warning
    }

@app.get("/api/history")
async def history():
    return await get_all_history()

@app.get("/api/analytics")
async def analytics():
    # Aggregate stats from history table
    history_data = await get_all_history()
    ocr_total = sum(1 for h in history_data if h["type"]=="ocr")
    obj_total = sum(1 for h in history_data if h["type"]=="object_detection")
    ocr_confidences = [h["confidence"] for h in history_data if h["type"]=="ocr"]
    obj_confidences = [h["confidence"] for h in history_data if h["type"]=="object_detection"]
    return {
        "total_ocr": ocr_total,
        "total_object_detection": obj_total,
        "ocr_avg_confidence": round(sum(ocr_confidences)/len(ocr_confidences),2) if ocr_confidences else 0,
        "obj_avg_confidence": round(sum(obj_confidences)/len(obj_confidences),2) if obj_confidences else 0,
        "history": history_data
    }