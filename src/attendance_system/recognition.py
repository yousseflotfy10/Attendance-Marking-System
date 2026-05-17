from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

try:
    from tensorflow import keras
    from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess_input
    from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
except Exception:
    keras = None

    def mobilenet_preprocess_input(value):
        return value

    def efficientnet_preprocess_input(value):
        return value

    def vgg16_preprocess_input(value):
        return value

from .config import FACE_INPUT_SIZE, LABEL_MAP_PATH, RECOGNITION_METADATA_PATH, RECOGNITION_MODEL_PATH
from .entities import RecognitionResult


PREPROCESSORS = {
    "mobilenetv2": mobilenet_preprocess_input,
    "efficientnetb0": efficientnet_preprocess_input,
    "vgg16": vgg16_preprocess_input,
}


class MobileNetV2Recognizer:
    def __init__(self, model_path: Path = RECOGNITION_MODEL_PATH, label_map_path: Path = LABEL_MAP_PATH, metadata_path: Path = RECOGNITION_METADATA_PATH) -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV is required for recognition preprocessing. Install requirements.txt first.")
        self.model_path = Path(model_path)
        self.label_map_path = Path(label_map_path)
        self.metadata_path = Path(metadata_path)
        self.backbone_name = "mobilenetv2"
        self.model = self._load_model()
        self.index_to_label = self._load_label_map()
        self.preprocess = PREPROCESSORS[self.backbone_name]

    def _load_metadata(self) -> dict:
        if not self.metadata_path.exists():
            return {}
        with self.metadata_path.open("r", encoding="utf-8") as file_handle:
            return json.load(file_handle)

    def _load_model(self):
        if keras is None:
            return None
        if not self.model_path.exists():
            return None
        metadata = self._load_metadata()
        self.backbone_name = str(metadata.get("backbone", "mobilenetv2")).lower()
        self.preprocess = PREPROCESSORS.get(self.backbone_name, mobilenet_preprocess_input)
        return keras.models.load_model(self.model_path)

    def _load_label_map(self) -> dict[int, str]:
        if not self.label_map_path.exists():
            return {}
        with self.label_map_path.open("r", encoding="utf-8") as file_handle:
            raw_map = json.load(file_handle)
        return {int(index): label for index, label in raw_map.items()}

    def preprocess_face(self, face_bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(face_bgr, FACE_INPUT_SIZE)
        rgb_face = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        face_array = np.asarray(rgb_face, dtype=np.float32)
        face_array = np.expand_dims(face_array, axis=0)
        return self.preprocess(face_array)

    def predict(self, face_bgr: np.ndarray) -> RecognitionResult:
        if self.model is None or keras is None:
            return RecognitionResult(student_name="Unknown", confidence=0.0)

        prepared_face = self.preprocess_face(face_bgr)
        probabilities = self.model.predict(prepared_face, verbose=0)[0]
        class_index = int(np.argmax(probabilities))
        confidence = float(probabilities[class_index])
        label = self.index_to_label.get(class_index, "Unknown")
        return RecognitionResult(student_name=label, confidence=confidence)

