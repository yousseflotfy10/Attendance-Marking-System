from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FaceBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


@dataclass(frozen=True)
class Student:
    id: int
    name: str
    department: str


@dataclass(frozen=True)
class AttendanceRecord:
    id: int
    student_id: int
    student_name: str
    department: str
    attendance_date: str
    attendance_time: str
    status: str


@dataclass(frozen=True)
class RecognitionResult:
    student_name: str
    confidence: float

