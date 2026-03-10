from dataclasses import dataclass

@dataclass(frozen=True)
class Student:
    id: str
    name: str
    major: str
    year: int

# Predefined dictionary of students
STUDENTS: dict[str, Student] = {
    "1001": Student(id="1001", name="Alice Johnson", major="Computer Science", year=2),
    "1002": Student(id="1002", name="Bob Smith", major="Mathematics", year=3),
    "1003": Student(id="1003", name="Charlie Brown", major="Physics", year=1),
}

def get_student_by_id(student_id) -> Student:
    """
    Retrieve a Student object by their ID.

    Args:
        student_id: The ID of the student (int or str).

    Returns:
        Student: The student object corresponding to the given ID.

    Raises:
        ValueError: If the student ID does not exist.
    """
    sid = str(student_id)
    try:
        return STUDENTS[sid]
    except KeyError:
        raise ValueError(f"Student ID '{sid}' does not exist.") from None
