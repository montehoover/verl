from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Student:
    id: str
    name: str
    age: int
    major: str


# Predefined dictionary of students keyed by their string ID
STUDENTS_BY_ID: Dict[str, Student] = {
    "S001": Student(id="S001", name="Alice Johnson", age=20, major="Computer Science"),
    "S002": Student(id="S002", name="Bob Smith", age=21, major="Mathematics"),
    "S003": Student(id="S003", name="Carla Gomez", age=22, major="Physics"),
    "S004": Student(id="S004", name="David Lee", age=19, major="Chemistry"),
    "S005": Student(id="S005", name="Evelyn Chen", age=23, major="Biology"),
}


def get_student_by_id(student_id: str) -> Student:
    """
    Retrieve a Student by their string ID.

    Args:
        student_id: The student ID as a string.

    Returns:
        The corresponding Student object.

    Raises:
        ValueError: If the student ID does not exist in the predefined dictionary.
    """
    try:
        return STUDENTS_BY_ID[student_id]
    except KeyError:
        raise ValueError(f"Student ID '{student_id}' does not exist.")
