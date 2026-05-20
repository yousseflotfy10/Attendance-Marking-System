from pathlib import Path
import shutil
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from attendance_system.training import train_recognition_model
from attendance_system.config import DEFAULT_RECOGNITION_BACKBONE


def _tensorflow_available() -> bool:
    try:
        import tensorflow  # noqa: F401

        return True
    except Exception:
        return False


def _delegate_if_needed() -> int | None:
    if _tensorflow_available():
        return None

    # Auto-switch from unsupported runtime to Python 3.13 if available.
    if sys.version_info >= (3, 14):
        py_launcher = shutil.which("py")
        if py_launcher is None:
            return None

        command = [py_launcher, "-3.13", str(Path(__file__).resolve()), *sys.argv[1:]]
        print("TensorFlow not available in this interpreter. Retrying with Python 3.13...")
        completed = subprocess.run(command)
        return completed.returncode

    return None


def _parse_args(argv: list[str]) -> tuple[Path, int, int, str]:
    dataset = Path("dataset")
    epochs = 15
    batch_size = 32
    backbone = DEFAULT_RECOGNITION_BACKBONE

    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--dataset" and index + 1 < len(argv):
            dataset = Path(argv[index + 1])
            index += 2
            continue
        if token == "--epochs" and index + 1 < len(argv):
            epochs = int(argv[index + 1])
            index += 2
            continue
        if token == "--batch-size" and index + 1 < len(argv):
            batch_size = int(argv[index + 1])
            index += 2
            continue
        if token == "--backbone" and index + 1 < len(argv):
            backbone = argv[index + 1]
            index += 2
            continue
        raise RuntimeError(
            "Usage: python train_model.py [--dataset dataset] [--epochs 15] [--batch-size 32] [--backbone efficientnetb0|auto|mobilenetv2|vgg16]"
        )

    return dataset, epochs, batch_size, backbone


def main() -> int:
    delegated_exit = _delegate_if_needed()
    if delegated_exit is not None:
        return delegated_exit

    try:
        dataset, epochs, batch_size, backbone = _parse_args(sys.argv[1:])
    except Exception as exc:
        print(str(exc))
        return 1

    start_time = time.perf_counter()
    print("Training in process... please wait.")
    try:
        _, _, metadata = train_recognition_model(
            dataset,
            epochs=epochs,
            batch_size=batch_size,
            backbone=backbone,
        )
    except RuntimeError as exc:
        print(f"Training failed: {exc}")
        return 1
    elapsed_seconds = time.perf_counter() - start_time
    print(
        f"Training complete. Best backbone: {metadata['backbone']} "
        f"(val_accuracy={metadata['validation_accuracy']:.4f}, duration={elapsed_seconds:.1f}s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
