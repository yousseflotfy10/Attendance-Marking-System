from __future__ import annotations

from typing import Optional
from importlib import import_module

try:
    cv2 = import_module("cv2")
except Exception:
    cv2 = None

try:
    QtCore = import_module("PyQt5.QtCore")
    QtGui = import_module("PyQt5.QtGui")
    QtWidgets = import_module("PyQt5.QtWidgets")
    Qt = QtCore.Qt
    QTimer = QtCore.QTimer
    QImage = QtGui.QImage
    QPixmap = QtGui.QPixmap
    QApplication = QtWidgets.QApplication
    QHBoxLayout = QtWidgets.QHBoxLayout
    QInputDialog = QtWidgets.QInputDialog
    QLabel = QtWidgets.QLabel
    QLineEdit = QtWidgets.QLineEdit
    QMainWindow = QtWidgets.QMainWindow
    QMessageBox = QtWidgets.QMessageBox
    QPushButton = QtWidgets.QPushButton
    QTableWidget = QtWidgets.QTableWidget
    QTableWidgetItem = QtWidgets.QTableWidgetItem
    QVBoxLayout = QtWidgets.QVBoxLayout
    QWidget = QtWidgets.QWidget
    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False

from .config import DEFAULT_ADMIN_USERNAME, DEFAULT_CAMERA_INDEX, DEFAULT_CAPTURE_SAMPLE_COUNT
from .db import AttendanceDatabase
from .detection import FaceDetector
from .pipeline import AttendancePipeline
from .recognition import MobileNetV2Recognizer


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


if PYQT_AVAILABLE:

    class AttendanceWindow(QMainWindow):
        def __init__(self) -> None:
            if cv2 is None:
                raise RuntimeError("OpenCV is required for the dashboard camera feed. Install requirements.txt first.")
            super().__init__()
            self.setWindowTitle("Attendance Marking System")
            self.resize(1200, 800)

            self.database = AttendanceDatabase()
            self.pipeline = AttendancePipeline(
                database=self.database,
                detector=FaceDetector(),
                recognizer=MobileNetV2Recognizer(),
            )
            self.capture = _open_camera(DEFAULT_CAMERA_INDEX)
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.update_frame)
            self.admin_logged_in = False
            self.admin_username = DEFAULT_ADMIN_USERNAME

            self.video_label = QLabel("Camera feed not started")
            self.video_label.setAlignment(Qt.AlignCenter)
            self.video_label.setMinimumSize(800, 500)

            self.status_label = QLabel("Ready")
            self.start_button = QPushButton("Start Camera")
            self.stop_button = QPushButton("Stop Camera")
            self.refresh_button = QPushButton("Refresh Attendance")
            self.admin_login_button = QPushButton("Admin Login")
            self.collect_faces_button = QPushButton("Collect Faces")

            self.start_button.clicked.connect(self.start_camera)
            self.stop_button.clicked.connect(self.stop_camera)
            self.refresh_button.clicked.connect(self.refresh_attendance_table)
            self.admin_login_button.clicked.connect(self.admin_login)
            self.collect_faces_button.clicked.connect(self.collect_faces)

            controls = QHBoxLayout()
            controls.addWidget(self.start_button)
            controls.addWidget(self.stop_button)
            controls.addWidget(self.refresh_button)
            controls.addWidget(self.admin_login_button)
            controls.addWidget(self.collect_faces_button)

            self.attendance_table = QTableWidget(0, 5)
            self.attendance_table.setHorizontalHeaderLabels(["Name", "Department", "Date", "Time", "Status"])

            layout = QVBoxLayout()
            layout.addWidget(self.video_label)
            layout.addLayout(controls)
            layout.addWidget(self.status_label)
            layout.addWidget(self.attendance_table)

            container = QWidget()
            container.setLayout(layout)
            self.setCentralWidget(container)
            self.status_label.setText(f"Admin login required. Username: {DEFAULT_ADMIN_USERNAME}")
            self.refresh_attendance_table()

        def start_camera(self) -> None:
            if self.capture is None or not self.capture.isOpened():
                self.capture = _open_camera(DEFAULT_CAMERA_INDEX)
            if self.capture is None or not self.capture.isOpened():
                self.status_label.setText("No working camera found")
                self.video_label.setText("Camera feed not started")
                return
            self.timer.start(30)
            self.status_label.setText("Camera running")

        def stop_camera(self) -> None:
            self.timer.stop()
            self.status_label.setText("Camera stopped")

        def admin_login(self) -> None:
            password, ok = QInputDialog.getText(self, "Admin Login", "Password:", QLineEdit.Password)
            if not ok:
                return

            if self.database.authenticate_admin(DEFAULT_ADMIN_USERNAME, password):
                self.admin_logged_in = True
                self.admin_username = DEFAULT_ADMIN_USERNAME
                self.status_label.setText(f"Admin logged in: {self.admin_username}")
                QMessageBox.information(self, "Login Successful", f"Welcome, {self.admin_username}.")
            else:
                self.admin_logged_in = False
                self.admin_username = DEFAULT_ADMIN_USERNAME
                QMessageBox.warning(self, "Login Failed", "Invalid password.")

        def collect_faces(self) -> None:
            if not self.admin_logged_in:
                QMessageBox.warning(self, "Access Denied", "Please login as admin before collecting faces.")
                return

            student_name, ok = QInputDialog.getText(self, "Collect Faces", "Student Name:")
            if not ok:
                return
            student_name = student_name.strip()
            if not student_name:
                QMessageBox.warning(self, "Invalid Input", "Student name cannot be empty.")
                return

            department, ok = QInputDialog.getText(self, "Collect Faces", "Department:")
            if not ok:
                return
            department = department.strip()
            if not department:
                QMessageBox.warning(self, "Invalid Input", "Department cannot be empty.")
                return

            from .capture import collect_dataset

            camera_was_running = self.timer.isActive()
            if camera_was_running:
                self.stop_camera()

            try:
                self.database.upsert_student(student_name, department)
                collect_dataset(
                    student_name=student_name,
                    department=department,
                    camera_index=DEFAULT_CAMERA_INDEX,
                )
                self.status_label.setText(f"Collected faces for {student_name} by admin {self.admin_username}")
                sample_word = "sample" if DEFAULT_CAPTURE_SAMPLE_COUNT == 1 else "samples"
                QMessageBox.information(
                    self,
                    "Done",
                    f"Collected {DEFAULT_CAPTURE_SAMPLE_COUNT} face {sample_word} for {student_name}.",
                )
            except RuntimeError as exc:
                QMessageBox.critical(self, "Collection Error", str(exc))
            finally:
                if camera_was_running:
                    self.start_camera()

        def closeEvent(self, event) -> None:
            self.timer.stop()
            if self.capture is not None and self.capture.isOpened():
                self.capture.release()
            super().closeEvent(event)

        def update_frame(self) -> None:
            success, frame = self.capture.read()
            if not success:
                self.status_label.setText("Failed to read camera frame")
                return
            processed_frame, events = self.pipeline.process_frame(frame)
            self.status_label.setText(self._build_status(events))
            self._show_frame(processed_frame)
            self.refresh_attendance_table()

        def _show_frame(self, frame) -> None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel_count = rgb_frame.shape
            bytes_per_line = channel_count * width
            image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(image).scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        def _build_status(self, events) -> str:
            if not events:
                return "No faces detected"
            marked_names = [event.student_name for event in events if event.marked]
            if not marked_names:
                return "Faces detected, no new attendance marked"
            return "Marked: " + ", ".join(marked_names)

        def refresh_attendance_table(self) -> None:
            records = self.database.list_today_attendance()
            self.attendance_table.setRowCount(len(records))
            for row_index, record in enumerate(records):
                values = [record.student_name, record.department, record.attendance_date, record.attendance_time, record.status]
                for column_index, value in enumerate(values):
                    self.attendance_table.setItem(row_index, column_index, QTableWidgetItem(value))
else:
    AttendanceWindow = None


def launch_ui() -> int:
    if not PYQT_AVAILABLE:
        raise RuntimeError("PyQt5 is not installed. Install requirements.txt to launch the dashboard.")
    if cv2 is None:
        raise RuntimeError("OpenCV is required for the dashboard camera feed. Install requirements.txt first.")
    application = QApplication([])
    window = AttendanceWindow()
    window.show()
    return application.exec_()

