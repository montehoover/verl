from dataclasses import dataclass
from typing import Dict, Optional
import string


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


def validate_report_template(template: str) -> bool:
    """
    Validate that a report template only contains allowed placeholders.

    Valid placeholders must:
      - Reference the 'student' object, e.g., {student.name}
      - Use one of the allowed public fields.

    Sensitive placeholders (e.g., {student.email}) and any unknown or malformed
    placeholders will raise a ValueError.

    Args:
        template: The template string to validate.

    Returns:
        True if all placeholders are valid.

    Raises:
        ValueError: If any invalid or sensitive placeholders are found.
    """
    # Define which fields are allowed to be referenced publicly
    ALLOWED_FIELDS = {"id", "name", "grade", "course"}
    # Define sensitive fields which must not be exposed
    SENSITIVE_FIELDS = {"email"}

    invalid: list[str] = []
    sensitive: list[str] = []

    formatter = string.Formatter()
    for literal_text, field_name, format_spec, conversion in formatter.parse(template):
        # field_name is None for plain text segments
        if field_name is None:
            continue

        # Normalize field name (strip whitespace)
        field = field_name.strip()

        # Must start with 'student.' and be a single-level attribute
        if not field.startswith("student.") or field.count(".") != 1:
            invalid.append(field_name)
            continue

        _, attr = field.split(".", 1)

        # Sensitive fields
        if attr in SENSITIVE_FIELDS:
            sensitive.append(field_name)
            continue

        # Allowed public fields
        if attr not in ALLOWED_FIELDS:
            invalid.append(field_name)
            continue

    if sensitive:
        placeholders = ", ".join(f"{{{p}}}" for p in sensitive)
        raise ValueError(f"Sensitive placeholders are not allowed: {placeholders}")

    if invalid:
        placeholders = ", ".join(f"{{{p}}}" for p in invalid)
        allowed_list = ", ".join(f"{{student.{f}}}" for f in sorted(ALLOWED_FIELDS))
        raise ValueError(f"Invalid placeholders: {placeholders}. Allowed placeholders include: {allowed_list}")

    return True
