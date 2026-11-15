# CDCN++ Face Anti-Spoofing Web Deployment 2 (`best_cdcnpp.pth`)

An end-to-end, multi-face Presentation Attack Detection (PAD) web application powered by your trained **`best_cdcnpp.pth`** model checkpoint (CVPR 2020: *"Searching Central Difference Convolutional Networks for Face Anti-Spoofing"*).

---

## Model & Checkpoint Overview

- **Checkpoint Path**: `deployment 2/weights/best_cdcnpp.pth`
- **Architecture**: `CDCN++` (Central Difference Convolutional Network with Multi-scale Attention Fusion Module)
- **Central Difference Parameter ($\theta$)**: `0.7`
- **Validation Accuracy**: `87.72%`
- **Trained Epoch**: `2`
- **Optimal Threshold**: `0.56` (exact: `0.5614`)
- **Input Size**: `256 × 256`
- **Depth Map Size**: `32 × 32`

---

## Modular File Structure

The `deployment 2` folder is completely standalone:

```
deployment 2/
├── config.json                  # Paper defaults & runtime configuration (threshold: 0.56)
├── backend.py                   # FastAPI REST API endpoints & key normalizer
├── run_server.py                # Server launcher script (runs on http://127.0.0.1:8001)
├── test_pipeline.py             # Automated end-to-end verification script
├── models/
│   ├── CDCNs.py                 # CDCN++ & Central Difference Convolution (CDC) definitions
│   └── __init__.py
├── weights/
│   └── best_cdcnpp.pth          # 27 MB trained model checkpoint
├── modules/
│   ├── detector.py              # OpenCV Haar Cascade multi-face detector
│   ├── preprocessor.py          # Direct camera crop & tensor normalization
│   ├── inferencer.py            # CDCN++ depth inference, score & JET heatmap generation
│   └── __init__.py
└── static/
    ├── index.html               # Clean dark-mode dashboard with accuracy & threshold badges
    ├── app.js                   # Webcam capture, dynamic thresholding & canvas rendering
    └── style.css                # Polished dashboard styling
```

---

## Key Features

1. **Dedicated to `best_cdcnpp.pth`**:
   - Automatically loads and maps weights from `best_cdcnpp.pth` with full layer matching.
   - Preserves training metadata: Epoch 2, Validation Accuracy 87.72%, Optimal Threshold 0.56.
2. **Direct Camera Feed & Multi-Face Detection**:
   - Takes raw webcam frames directly from HTML5 video.
   - Detects all faces using OpenCV Haar Cascade and runs CDCN++ on each face independently.
3. **Interactive Threshold Slider (0.00 to 1.00)**:
   - Defaults to the optimal trained threshold (`0.56`).
   - Can be adjusted in real-time between `0.00` and `1.00` with instant verdict updates.
4. **Visual 3D Depth Map Reconstruction**:
   - Produces estimated 3D facial depth maps rendered as colorized JET heatmaps.
   - Real faces yield prominent convex depth structures; 2D spoof attacks produce flat or zero-depth maps.

---

## How to Run

1. Open a terminal in the root workspace directory and run:
   ```bash
   python "deployment 2/run_server.py"
   ```
   *(By default, this server runs on port **8001** to avoid any conflicts with previous deployments on 8000).*

2. Open your web browser:
   **[http://127.0.0.1:8001](http://127.0.0.1:8001)**

3. Click **"Start Camera"**, face the camera, and click **"Capture & Analyze"**.

4. Adjust the **Threshold** slider as needed.
