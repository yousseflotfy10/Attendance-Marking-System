# Attendance Marking System

Real-time academic attendance system built around the course progression:

CNN fundamentals -> YOLO face detection -> MobileNetV2 transfer learning -> regularization -> database-backed attendance.

## Architecture

- Face detection: YOLOv8 when weights are available, with a built-in OpenCV Haar fallback so the app can still run during setup.
- Face recognition: MobileNetV2 transfer learning classifier.
- Regularization: dropout plus on-the-fly augmentation during training.
- Runtime: OpenCV webcam pipeline.
- Storage: SQLite database for students and attendance rows.
- UI: PyQt5 dashboard.

## Project Layout

- `src/attendance_system/detection.py` - face detection wrapper.
- `src/attendance_system/recognition.py` - MobileNetV2 inference helper.
- `src/attendance_system/training.py` - transfer learning training utilities.
- `src/attendance_system/db.py` - SQLite attendance storage.
- `src/attendance_system/pipeline.py` - real-time attendance orchestration.
- `src/attendance_system/ui.py` - PyQt5 dashboard.
- `src/attendance_system/cli.py` - command line entry points.

## Setup

1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

3. Initialize the database:

```bash
python -m attendance_system.cli init-db
```

4. Collect student samples into `dataset/<student_name>/`.
5. Train the recognition model with transfer learning.
6. Launch the dashboard or webcam loop.

## Commands

```bash
python -m attendance_system.cli collect --name Ahmed --department CSE
python -m attendance_system.cli synthetic --students 3 --images 15
python -m attendance_system.cli train --dataset dataset
python -m attendance_system.cli run
```

## Separate Runners

If you want separate files for each task:

```bash
python train_model.py --dataset dataset --epochs 15 --batch-size 32 --backbone auto
```

Or use the root entry point:

```bash
python main.py collect --name Ahmed --department CSE
python main.py synthetic --students 3 --images 15
python main.py train --dataset dataset
python main.py run
```

## Academic Framing

This project is intentionally split into two deep learning stages:

1. YOLO detects faces in the classroom stream in real time.
2. MobileNetV2 extracts discriminative identity features through transfer learning.

That separation makes the project easier to justify in a presentation and matches the course sequence directly.
