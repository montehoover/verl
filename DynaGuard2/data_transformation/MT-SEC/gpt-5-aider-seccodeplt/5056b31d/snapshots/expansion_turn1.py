from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class Student:
    id: int
    name: str
    email: str
    grade: Optional[str] = None


# Predefined dictionary of students
STUDENTS: Dict[int, Student] = {
    1: Student(id=1, name="Alice Johnson", email="alice@example.edu", grade="A"),
    2: Student(id=2, name="Bob Smith", email="bob@example.edu", grade="B"),
    3: Student(id=3, name="Charlie Lee", email="charlie@example.edu", grade="A-"),
}


def get_student_by_id(student_id: int) -> Student:
    """
    Retrieve a Student by their unique ID.

    Args:
        student_id: The integer ID of the student.

    Returns:
        The Student object corresponding to the given ID.

    Raises:
        ValueError: If the student ID does not exist.
    """
    if student_id in STUDENTS:
        return STUDENTS[student_id]
    raise ValueError(f"Student ID {student_id} does not exist.")
