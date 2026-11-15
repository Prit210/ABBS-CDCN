import os
import base64
import cv2
import numpy as np
import torch
from models.CDCNs import CDCNpp, Conv2d_cd

class CDCNInferencer:
    def __init__(self, weights_path, theta=0.7, device="cpu"):
        self.device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        self.theta = float(theta)
        self.weights_path = weights_path
        self.model = None
        self.metadata = {}
        self._load_model()

    def _load_model(self):
        print(f"[Model] Initializing CDCN++ with theta={self.theta} on {self.device}...")
        net = CDCNpp(basic_conv=Conv2d_cd, theta=self.theta)

        if os.path.exists(self.weights_path):
            print(f"[Model] Loading weights from {self.weights_path}...")
            checkpoint = torch.load(self.weights_path, map_location=self.device)
            state_dict = checkpoint.get("state_dict", checkpoint)

            # Store checkpoint training metadata if present
            if isinstance(checkpoint, dict):
                self.metadata = {
                    "epoch": checkpoint.get("epoch"),
                    "best_acc": checkpoint.get("best_acc"),
                    "best_threshold": checkpoint.get("best_threshold"),
                    "checkpoint_theta": checkpoint.get("theta")
                }
                print(f"[Model] Checkpoint metadata: {self.metadata}")

            # Remap layer keys seamlessly if checkpoint uses stem/head notation
            cleaned_sd = {}
            for k, v in state_dict.items():
                nk = k[len("module."):] if k.startswith("module.") else k
                if nk.startswith("stem."):
                    nk = "conv1." + nk[len("stem."):]
                elif nk.startswith("head."):
                    nk = "lastconv1." + nk[len("head."):]
                cleaned_sd[nk] = v

            net.load_state_dict(cleaned_sd, strict=True)
            print("[Model] Weights loaded successfully (All layers matched).")
        else:
            print(f"[Model] Error: Weights file not found at {self.weights_path}!")

        net.to(self.device)
        net.eval()
        self.model = net

    def update_theta(self, new_theta):
        if float(new_theta) != self.theta:
            self.theta = float(new_theta)
            self._load_model()

    def predict_depth(self, tensor, threshold=0.56):
        """
        Executes model inference on a normalized tensor (1, 3, 256, 256).
        Returns:
            score: float (mean depth map value between 0.0 and 1.0)
            verdict: 'Real' | 'Spoof'
            depth_map: 32x32 numpy array
            heatmap_bgr: 128x128 colorized JET depth map
        """
        with torch.no_grad():
            outputs = self.model(tensor)
            depth_raw = outputs[0].squeeze(0).cpu().numpy()
            depth_map = np.clip(depth_raw, 0.0, 1.0)
            score = float(np.mean(depth_map))

        verdict = "Real" if score >= threshold else "Spoof"

        # Colorize depth map (32x32 -> 128x128 JET heatmap)
        depth_vis = (depth_map * 255.0).astype(np.uint8)
        depth_vis_resized = cv2.resize(depth_vis, (128, 128), interpolation=cv2.INTER_NEAREST)
        heatmap_bgr = cv2.applyColorMap(depth_vis_resized, cv2.COLORMAP_JET)

        return {
            "score": round(score, 4),
            "verdict": verdict,
            "depth_map": depth_map,
            "heatmap_bgr": heatmap_bgr
        }

def encode_bgr_to_base64(img_bgr, quality=85):
    if img_bgr is None:
        return ""
    success, buffer = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")
