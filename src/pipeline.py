from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

from .config import DEFAULT_RECOGNITION_CONFIDENCE
from .db import AttendanceDatabase
from .detection import FaceDetector
from .entities import FaceBox, RecognitionResult
from .recognition import MobileNetV2Recognizer


@dataclass(frozen=True)
class AttendanceEvent:
    student_name: str
    confidence: float
    marked: bool


class AttendancePipeline:
    def __init__(
        self,
        database: AttendanceDatabase,
        detector: FaceDetector,
        recognizer: MobileNetV2Recognizer,
        recognition_threshold: float = DEFAULT_RECOGNITION_CONFIDENCE,
    ) -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV is required for the live attendance pipeline. Install requirements.txt first.")
        self.database = database
        self.detector = detector
        self.recognizer = recognizer
        self.recognition_threshold = recognition_threshold
        self.marked_today = self.database.get_marked_student_names()

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[AttendanceEvent]]:
        events: List[AttendanceEvent] = []
        face_boxes = self.detector.detect(frame)

        for face_box in face_boxes:
            clipped_box = self._clip_box(frame, face_box)
            if clipped_box is None:
                continue
            x1, y1, x2, y2 = clipped_box
            face_crop = frame[y1:y2, x1:x2]
            recognition = self.recognizer.predict(face_crop)

            marked = False
            student = self.database.get_student_by_name(recognition.student_name)
            if (
                student is not None
                and recognition.student_name != "Unknown"
                and recognition.confidence >= self.recognition_threshold
                and recognition.student_name not in self.marked_today
            ):
                marked = self.database.mark_attendance(student.id)
                if marked:
                    self.marked_today.add(student.name)

            self._draw_face_annotation(frame, clipped_box, recognition, marked)
            events.append(
                AttendanceEvent(
                    student_name=recognition.student_name,
                    confidence=recognition.confidence,
                    marked=marked,
                )
            )

        return frame, events

    def _clip_box(self, frame: np.ndarray, face_box: FaceBox) -> tuple[int, int, int, int] | None:
        height, width = frame.shape[:2]
        x1 = max(0, min(face_box.x1, width - 1))
        y1 = max(0, min(face_box.y1, height - 1))
        x2 = max(0, min(face_box.x2, width))
        y2 = max(0, min(face_box.y2, height))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _draw_face_annotation(
        self,
        frame: np.ndarray,
        box: tuple[int, int, int, int],
        recognition: RecognitionResult,
        marked: bool,
    ) -> None:
        x1, y1, x2, y2 = box
        color = (0, 200, 0) if marked else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{recognition.student_name} {recognition.confidence:.2f}"
        cv2.putText(frame, label, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

