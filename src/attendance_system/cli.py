from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .config import DATABASE_PATH
from .db import AttendanceDatabase


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Real-time attendance marking system")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init-db", help="Create the SQLite schema")

    collect_parser = subparsers.add_parser("collect", help="Capture dataset samples for one student")
    collect_parser.add_argument("--name", required=True, help="Student name")
    collect_parser.add_argument("--department", required=True, help="Student department")
    collect_parser.add_argument("--samples", type=int, default=100, help="Number of samples to capture")
    collect_parser.add_argument("--camera", type=int, default=0, help="Camera index")

    train_parser = subparsers.add_parser("train", help="Train and compare recognition backbones")
    train_parser.add_argument("--dataset", type=Path, default=Path("dataset"), help="Dataset root directory")
    train_parser.add_argument("--epochs", type=int, default=15, help="Training epochs")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    train_parser.add_argument(
        "--backbone",
        choices=["auto", "mobilenetv2", "efficientnetb0", "vgg16"],
        default="auto",
        help="Recognition backbone to train; auto compares all 3 and keeps the best",
    )

    synthetic_parser = subparsers.add_parser("synthetic", help="Generate a synthetic dataset for quick testing")
    synthetic_parser.add_argument("--dataset", type=Path, default=Path("dataset"), help="Dataset root directory")
    synthetic_parser.add_argument("--students", type=int, default=3, help="Number of synthetic students")
    synthetic_parser.add_argument("--images", type=int, default=15, help="Images per synthetic student")

    subparsers.add_parser("run", help="Launch the GUI attendance dashboard")
    subparsers.add_parser("watch", help="Run the webcam pipeline in a console loop")
    return parser


def run_console_pipeline() -> int:
    import cv2

    from .detection import FaceDetector
    from .pipeline import AttendancePipeline
    from .recognition import MobileNetV2Recognizer

    database = AttendanceDatabase()
    pipeline = AttendancePipeline(database, FaceDetector(), MobileNetV2Recognizer())
    capture = cv2.VideoCapture(0)
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            processed_frame, _ = pipeline.process_frame(frame)
            cv2.imshow("Attendance", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    if argv is None and len(sys.argv) == 1:
        argv = ["run"]
    args = parser.parse_args(argv)

    if args.command == "init-db":
        AttendanceDatabase(DATABASE_PATH)
        print(f"Initialized database at {DATABASE_PATH}")
        return 0

    if args.command == "collect":
        from .capture import collect_dataset

        AttendanceDatabase(DATABASE_PATH).upsert_student(args.name, args.department)
        collect_dataset(
            student_name=args.name,
            department=args.department,
            sample_count=args.samples,
            camera_index=args.camera,
        )
        print(f"Collected samples for {args.name}")
        return 0

    if args.command == "train":
        from .training import train_recognition_model

        start_time = time.perf_counter()
        print("Training in process... please wait.")
        _, _, metadata = train_recognition_model(
            args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            backbone=args.backbone,
        )
        elapsed_seconds = time.perf_counter() - start_time
        print(
            f"Training complete. Best backbone: {metadata['backbone']} "
            f"(val_accuracy={metadata['validation_accuracy']:.4f}, duration={elapsed_seconds:.1f}s)"
        )
        return 0

    if args.command == "synthetic":
        from .synthetic import create_synthetic_dataset

        total = create_synthetic_dataset(
            output_dir=args.dataset,
            num_students=args.students,
            images_per_student=args.images,
        )
        print(f"Synthetic dataset created: {total} images")
        return 0

    if args.command == "run":
        from .ui import launch_ui

        try:
            return launch_ui()
        except RuntimeError as exc:
            print(str(exc))
            return 1

    if args.command == "watch":
        return run_console_pipeline()

    parser.error("Unknown command")
    return 1

