import os
import sys
import time
import json
import base64
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from modules.detector import FaceDetector
from modules.preprocessor import crop_face_square, prepare_tensor
from modules.inferencer import CDCNInferencer, encode_bgr_to_base64

CONFIG_PATH = os.path.join(CURRENT_DIR, "config.json")

def load_config():
    default_config = {
        "model_type": "CDCNpp",
        "theta": 0.7,
        "weights_path": os.path.join(CURRENT_DIR, "weights", "best_cdcnpp.pth"),
        "threshold": 0.56,
        "best_accuracy": 0.8772,
        "input_size": [256, 256],
        "depth_map_size": [32, 32],
        "face_detector": "haar",
        "device": "cpu"
    }
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
                default_config.update(saved)
        except Exception as e:
            print(f"[Config] Error reading {CONFIG_PATH}: {e}")

    # Ensure weights path is absolute if relative to deployment 2 folder
    if not os.path.isabs(default_config["weights_path"]):
        default_config["weights_path"] = os.path.normpath(os.path.join(CURRENT_DIR, default_config["weights_path"]))

    default_config["model_type"] = "CDCNpp"
    default_config["face_detector"] = "haar"

    return default_config

def save_config(cfg):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        print(f"[Config] Error writing {CONFIG_PATH}: {e}")

# Initialize components
app_config = load_config()
detector = FaceDetector()
inferencer = CDCNInferencer(
    weights_path=app_config["weights_path"],
    theta=app_config.get("theta", 0.7),
    device=app_config.get("device", "cpu")
)

app = FastAPI(title="CDCN++ Face Anti-Spoofing Deployment 2 API (best_cdcnpp.pth)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConfigUpdatePayload(BaseModel):
    theta: float | None = None
    threshold: float | None = None

class PredictPayload(BaseModel):
    image_base64: str
    threshold: float | None = None

@app.get("/api/config")
def get_config():
    return {
        "status": "success",
        "config": app_config,
        "metadata": inferencer.metadata,
        "active_detector": "OpenCV Haar Cascade"
    }

@app.post("/api/config")
def update_config(payload: ConfigUpdatePayload):
    global app_config

    if payload.theta is not None and payload.theta != app_config.get("theta"):
        app_config["theta"] = float(payload.theta)
        inferencer.update_theta(app_config["theta"])

    if payload.threshold is not None:
        app_config["threshold"] = float(payload.threshold)

    save_config(app_config)

    return {
        "status": "success",
        "message": "Configuration updated successfully",
        "config": app_config
    }

@app.post("/api/predict")
def predict_frame(payload: PredictPayload):
    start_time = time.time()

    raw_b64 = payload.image_base64
    if "," in raw_b64:
        raw_b64 = raw_b64.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(raw_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Failed to decode image data")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    h_img, w_img = img_bgr.shape[:2]

    # 1. Multi-face detection with OpenCV Haar Cascade
    face_boxes = detector.detect_faces(img_bgr)

    # Resolve threshold (0.0 to 1.0)
    threshold = payload.threshold if payload.threshold is not None else float(app_config.get("threshold", 0.56))
    results = []
    target_size = tuple(app_config.get("input_size", [256, 256]))

    # 2. Process each detected face (direct input from camera)
    for idx, box in enumerate(face_boxes):
        crop_bgr, [x1, y1, x2, y2] = crop_face_square(img_bgr, box)
        if crop_bgr is None:
            continue

        # Direct camera input converted to normalized tensor
        tensor = prepare_tensor(crop_bgr, target_size=target_size, device=inferencer.device)

        # CDCN++ model inference
        pred = inferencer.predict_depth(tensor, threshold=threshold)

        results.append({
            "face_id": idx + 1,
            "box": [x1, y1, x2, y2],
            "score": pred["score"],
            "verdict": pred["verdict"],
            "threshold": threshold,
            "face_crop_base64": encode_bgr_to_base64(crop_bgr, quality=80),
            "depth_map_base64": encode_bgr_to_base64(pred["heatmap_bgr"], quality=80)
        })

    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "status": "success",
        "image_size": [w_img, h_img],
        "faces_detected": len(results),
        "results": results,
        "latency_ms": elapsed_ms
    }

# Mount static frontend
STATIC_DIR = os.path.join(CURRENT_DIR, "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return JSONResponse({"status": "API active. Static frontend not found."})
