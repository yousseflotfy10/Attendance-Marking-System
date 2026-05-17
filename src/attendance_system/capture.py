from __future__ import annotations

from pathlib import Path

try:
    import cv2
except Exception:
    cv2 = None

from .config import (
    CAPTURE_MAX_FACE_AREA_RATIO,
    CAPTURE_MIN_EDGE_MARGIN_RATIO,
    CAPTURE_MIN_FACE_AREA_RATIO,
    DATASET_DIR,
    DEFAULT_CAPTURE_SAMPLE_COUNT,
)
from .detection import FaceDetector


def _open_camera(camera_index: int):
    if cv2 is None:
        return None
    capture = cv2.VideoCapture(camera_index)
    if capture.isOpened():
        success, _ = capture.read()
        if success:
            return capture
        capture.release()
    else:
        capture.release()
    return None


def _face_capture_eligible(frame, face_box) -> tuple[bool, str]:
    frame_height, frame_width = frame.shape[:2]
    face_width = max(0, face_box.x2 - face_box.x1)
    face_height = max(0, face_box.y2 - face_box.y1)
    if face_width == 0 or face_height == 0:
        return False, "No face area"

    face_area_ratio = (face_width * face_height) / float(frame_width * frame_height)
    if face_area_ratio < CAPTURE_MIN_FACE_AREA_RATIO:
        return False, "Move closer until your full face is visible"
    if face_area_ratio > CAPTURE_MAX_FACE_AREA_RATIO:
        return False, "Move a little farther back"

    edge_margin_x = int(frame_width * CAPTURE_MIN_EDGE_MARGIN_RATIO)
    edge_margin_y = int(frame_height * CAPTURE_MIN_EDGE_MARGIN_RATIO)
    touches_edge = (
        face_box.x1 <= edge_margin_x
        or face_box.y1 <= edge_margin_y
        or face_box.x2 >= frame_width - edge_margin_x
        or face_box.y2 >= frame_height - edge_margin_y
    )
    if touches_edge:
        return False, "Center your full face in the frame"

    return True, "Full face detected"


def collect_dataset(
    student_name: str,
    department: str,
    sample_count: int = DEFAULT_CAPTURE_SAMPLE_COUNT,
    camera_index: int = 0,
    dataset_dir: Path = DATASET_DIR,
    detector: FaceDetector | None = None,
) -> Path:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for dataset capture. Install requirements.txt before using collect.")
    dataset_dir = Path(dataset_dir)
    student_dir = dataset_dir / student_name
    student_dir.mkdir(parents=True, exist_ok=True)

    detector = detector or FaceDetector()
    capture = _open_camera(camera_index)
    if capture is None:
        raise RuntimeError("No working camera was found. Check your camera driver and camera permissions.")
    captured_samples = 0

    try:
        while captured_samples < sample_count:
            success, frame = capture.read()
            if not success:
                continue

            face_boxes = detector.detect(frame)
            if not face_boxes:
                cv2.imshow("Dataset Capture", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            primary_face = max(face_boxes, key=lambda box: (box.x2 - box.x1) * (box.y2 - box.y1))
            x1 = max(primary_face.x1, 0)
            y1 = max(primary_face.y1, 0)
            x2 = max(primary_face.x2, 0)
            y2 = max(primary_face.y2, 0)
            cropped_face = frame[y1:y2, x1:x2]
            if cropped_face.size == 0:
                continue

            eligible, guidance = _face_capture_eligible(frame, primary_face)
            if not eligible:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(
                    frame,
                    guidance,
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Captured {captured_samples}/{sample_count}",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Dataset Capture", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            image_path = student_dir / f"{captured_samples + 1}.jpg"
            cv2.imwrite(str(image_path), cropped_face)
            captured_samples += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{student_name} {captured_samples}/{sample_count}",
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            progress_pct = int((captured_samples / sample_count) * 100)
            cv2.putText(frame, f"Progress: {progress_pct}%", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            rotation = (captured_samples % 4)
            if rotation == 0:
                tip = "Move: CENTER"
            elif rotation == 1:
                tip = "Move: LEFT"
            elif rotation == 2:
                tip = "Move: RIGHT"
            else:
                tip = "Move: UP / DOWN"
            cv2.putText(frame, tip, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            cv2.putText(frame, "Save only when your full face is visible", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 1)
            
            cv2.imshow("Dataset Capture", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return student_dir

