import os
import cv2

class FaceDetector:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Haar cascade XML not found at {cascade_path}")
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        print(f"[Detector] Loaded OpenCV Haar Cascade Face Detector from {cascade_path}")

    def detect_faces(self, img_bgr):
        """
        Detects all faces in BGR image using OpenCV Haar Cascade.
        Returns: list of [x, y, w, h] boxes
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        detected = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return [[int(x), int(y), int(w), int(h)] for (x, y, w, h) in detected]
