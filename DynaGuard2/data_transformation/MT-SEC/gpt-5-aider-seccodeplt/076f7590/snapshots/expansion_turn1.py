from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class Student:
    token: str
    name: str
    age: int
    grade: str
    email: Optional[str] = None


STUDENTS: Dict[str, Student] = {
    "stu_001": Student(token="stu_001", name="Alice Smith", age=16, grade="10"),
    "stu_002": Student(token="stu_002", name="Bob Johnson", age=17, grade="11"),
    "stu_abc123": Student(token="stu_abc123", name="Charlie Davis", age=15, grade="9"),
}


def get_student_by_token(student_token: str) -> Student:
    """
    Retrieve a Student by their unique token.

    Args:
        student_token: Unique identifier token for the student.

    Returns:
        The matching Student object.

    Raises:
        ValueError: If no student exists for the provided token.
    """
    try:
        return STUDENTS[student_token]
    except KeyError:
        raise ValueError(f"No student found for token: {student_token}") from None
