import os
import sys
import base64

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from fastapi.testclient import TestClient
from backend import app

client = TestClient(app)

def test_pipeline():
    # 1. Test GET config
    cfg_resp = client.get("/api/config")
    assert cfg_resp.status_code == 200, f"Config failed: {cfg_resp.text}"
    cfg = cfg_resp.json()
    print("[TEST] /api/config loaded:")
    print(f"  Model Type: {cfg['config']['model_type']}")
    print(f"  Weights: {os.path.basename(cfg['config']['weights_path'])}")
    print(f"  Active Detector: {cfg['active_detector']}")
    print(f"  Checkpoint Metadata: {cfg.get('metadata')}")

    # 2. Test POST predict with direct camera frame
    sample_img = os.path.join(CURRENT_DIR, "..", "data", "test", "0_20230127005314.jpg")
    if not os.path.exists(sample_img):
        print(f"[TEST] Warning: Sample image not found at {sample_img}")
        return

    with open(sample_img, "rb") as f:
        b64_str = base64.b64encode(f.read()).decode("utf-8")

    # Test with default optimal threshold (0.56)
    resp = client.post("/api/predict", json={
        "image_base64": b64_str,
        "threshold": 0.56
    })
    assert resp.status_code == 200, f"Predict failed: {resp.text}"
    data = resp.json()
    print("\n[TEST] Direct Camera Prediction (Threshold=0.56):")
    print(f"  Faces detected: {data.get('faces_detected')}")
    print(f"  Latency: {data.get('latency_ms')} ms")
    for face in data.get("results", []):
        print(f"  Face #{face['face_id']}: Score={face['score']}, Verdict={face['verdict']}, Box={face['box']}")

    # 3. Test threshold slider update (e.g. threshold=0.18)
    resp2 = client.post("/api/predict", json={
        "image_base64": b64_str,
        "threshold": 0.18
    })
    assert resp2.status_code == 200
    data2 = resp2.json()
    print("\n[TEST] Prediction with Custom Threshold=0.18:")
    for face in data2.get("results", []):
        print(f"  Face #{face['face_id']}: Score={face['score']}, Verdict={face['verdict']}")

    print("\nAll tests passed successfully for Deployment 2 (best_cdcnpp.pth)!")

if __name__ == "__main__":
    test_pipeline()
