from __future__ import annotations

from pathlib import Path

try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None


DEFAULT_STUDENT_NAMES = ["Youssef", "Ahmed", "Sara", "Mona", "Omar", "Nour"]


def _require_cv_stack() -> None:
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required for synthetic data generation.")


def _create_synthetic_face(student_id: int, variation: int, size: tuple[int, int] = (224, 224)):
    _require_cv_stack()
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)

    np.random.seed(student_id * 1000 + variation)
    color = tuple(np.random.randint(50, 200, 3).tolist())

    cv2.rectangle(img, (20, 20), (size[0] - 20, size[1] - 20), color, -1)
    cv2.putText(img, f"Student {student_id + 1}", (35, size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"Var {variation + 1}", (35, size[1] // 2 + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
    return img


def create_synthetic_dataset(output_dir: Path, num_students: int = 3, images_per_student: int = 15) -> int:
    _require_cv_stack()

    if num_students < 1:
        raise RuntimeError("--students must be at least 1")
    if images_per_student < 1:
        raise RuntimeError("--images must be at least 1")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    names = DEFAULT_STUDENT_NAMES[:num_students]
    if len(names) < num_students:
        names.extend([f"Student_{index + 1}" for index in range(len(names), num_students)])

    created = 0
    for student_id, student_name in enumerate(names):
        student_dir = output_dir / student_name
        student_dir.mkdir(parents=True, exist_ok=True)

        for image_idx in range(images_per_student):
            img = _create_synthetic_face(student_id, image_idx)
            img_path = student_dir / f"{image_idx + 1:03d}.jpg"
            cv2.imwrite(str(img_path), img)
            created += 1

    return created
