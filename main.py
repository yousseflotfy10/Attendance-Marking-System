from pathlib import Path
from importlib import import_module
import shutil
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _tensorflow_available() -> bool:
    try:
        import_module("tensorflow")

        return True
    except Exception:
        return False


def _should_delegate(argv: list[str]) -> bool:
    if not argv:
        # Default path is `run`, which needs TensorFlow for recognition.
        return True
    return argv[0] in {"run", "watch", "train"}


def _delegate_to_python313_if_needed() -> int | None:
    args = sys.argv[1:]
    if _tensorflow_available() or not _should_delegate(args):
        return None

    if sys.version_info < (3, 14):
        return None

    py_launcher = shutil.which("py")
    if py_launcher is None:
        return None

    command = [py_launcher, "-3.13", str(Path(__file__).resolve()), *args]
    print("TensorFlow not available in this interpreter. Retrying with Python 3.13...")
    completed = subprocess.run(command)
    return completed.returncode


from attendance_system.cli import main


if __name__ == "__main__":
    delegated_exit = _delegate_to_python313_if_needed()
    if delegated_exit is not None:
        raise SystemExit(delegated_exit)
    raise SystemExit(main())
