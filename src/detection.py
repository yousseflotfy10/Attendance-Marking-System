from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

from .config import DEFAULT_DETECTION_CONFIDENCE, YOLO_WEIGHTS_PATH
from .entities import FaceBox


class FaceDetector:
    def __init__(self, weights_path: Path = YOLO_WEIGHTS_PATH, confidence: float = DEFAULT_DETECTION_CONFIDENCE) -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV is required for face detection. Install requirements.txt before using detection.")
        self.weights_path = Path(weights_path)
        self.confidence = confidence
        self._yolo_model = None
        self._cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self._load_model()

    def _load_model(self) -> None:
        if not self.weights_path.exists():
            return
        try:
            from ultralytics import YOLO

            self._yolo_model = YOLO(str(self.weights_path))
        except Exception:
            self._yolo_model = None

    def detect(self, frame: np.ndarray) -> List[FaceBox]:
        if self._yolo_model is not None:
            return self._detect_with_yolo(frame)
        return self._detect_with_haar(frame)

    def _detect_with_yolo(self, frame: np.ndarray) -> List[FaceBox]:
        assert self._yolo_model is not None
        detections = self._yolo_model.predict(frame, conf=self.confidence, verbose=False)
        if not detections:
            return []
        result = detections[0]
        if result.boxes is None:
            return []
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        face_boxes: List[FaceBox] = []
        for coords, score in zip(boxes, confidences):
            x1, y1, x2, y2 = [int(value) for value in coords]
            face_boxes.append(FaceBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=float(score)))
        return face_boxes

    def _detect_with_haar(self, frame: np.ndarray) -> List[FaceBox]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        face_boxes: List[FaceBox] = []
        for (x, y, width, height) in faces:
            face_boxes.append(FaceBox(x1=int(x), y1=int(y), x2=int(x + width), y2=int(y + height), confidence=0.5))
        return face_boxes
