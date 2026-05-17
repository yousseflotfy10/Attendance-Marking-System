from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from attendance_system.capture import collect_dataset
from attendance_system.db import AttendanceDatabase
from attendance_system.config import DEFAULT_CAPTURE_SAMPLE_COUNT


def _prompt(text: str) -> str:
    return input(text).strip()


def main() -> int:
    print("Face Collection (separate runner)")
    print("Press q in the camera window to stop early.")

    name = _prompt("Student name: ")
    if not name:
        print("Student name is required.")
        return 1

    department = _prompt("Department: ")
    if not department:
        print("Department is required.")
        return 1

    camera_text = _prompt("Camera index [default 0]: ")

    camera_index = 0 if not camera_text else int(camera_text)

    AttendanceDatabase().upsert_student(name, department)
    student_dir = collect_dataset(
        student_name=name,
        department=department,
        sample_count=DEFAULT_CAPTURE_SAMPLE_COUNT,
        camera_index=camera_index,
    )

    sample_word = "sample" if DEFAULT_CAPTURE_SAMPLE_COUNT == 1 else "samples"
    print(f"Collection finished for {name}. Saved {DEFAULT_CAPTURE_SAMPLE_COUNT} face {sample_word} in {student_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
