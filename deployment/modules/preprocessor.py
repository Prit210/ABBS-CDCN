import cv2
import numpy as np
import torch

def crop_face_square(img_bgr, box):
    """
    Crops face region directly from camera image with 1:1 aspect ratio and boundary clamping.
    box: [x, y, w, h]
    Returns: cropped BGR image, [x1, y1, x2, y2]
    """
    h_img, w_img = img_bgr.shape[:2]
    x, y, w, h = box
    edge = max(w, h)

    center_x = x + w / 2.0
    center_y = y + h / 2.0

    x1 = max(int(center_x - edge / 2.0), 0)
    y1 = max(int(center_y - edge / 2.0), 0)
    x2 = min(int(center_x + edge / 2.0), w_img)
    y2 = min(int(center_y + edge / 2.0), h_img)

    if x2 <= x1 or y2 <= y1:
        return None, [0, 0, 0, 0]

    crop = img_bgr[y1:y2, x1:x2]
    return crop, [x1, y1, x2, y2]

def prepare_tensor(crop_bgr, target_size=(256, 256), device="cpu"):
    """
    Direct input from camera: resizes to target_size (256, 256), converts BGR to RGB,
    normalizes to [-1, 1], and returns a PyTorch tensor (1, 3, H, W).
    """
    resized = cv2.resize(crop_bgr, target_size, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
    tensor = (tensor - 0.5) / 0.5
    tensor = tensor.unsqueeze(0).to(device)
    return tensor
