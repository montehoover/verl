from dataclasses import dataclass
from typing import Dict, Optional
import re


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


def validate_report_placeholders(template: str) -> bool:
    """
    Validate placeholders within a report template.

    A valid placeholder must be of the form {student.<field>}, where <field> is one of:
      - Allowed: name, age, grade, course
      - Sensitive (disallowed): token, email

    Args:
        template: The template string containing placeholders.

    Returns:
        True if all placeholders are valid.

    Raises:
        ValueError: If any placeholder is invalid, sensitive, or braces are unbalanced.
    """
    if template is None:
        raise ValueError("Template cannot be None.")

    # Quick check for balanced braces
    if template.count("{") != template.count("}"):
        raise ValueError("Unbalanced braces in template.")

    allowed_fields = {"name", "age", "grade", "course"}
    sensitive_fields = {"token", "email"}

    # Find all {...} occurrences
    matches = list(re.finditer(r"\{([^{}]+)\}", template))

    for match in matches:
        inner = match.group(1).strip()
        if not inner:
            raise ValueError("Empty placeholder {} is not allowed.")

        if not inner.startswith("student."):
            raise ValueError(
                f"Invalid placeholder '{{{inner}}}'. Placeholders must be of the form '{{student.<field>}}'."
            )

        parts = inner.split(".", 1)
        if len(parts) != 2 or not parts[1]:
            raise ValueError(
                f"Invalid placeholder '{{{inner}}}'. Missing field after 'student.'."
            )

        field = parts[1]

        # Field must be a simple identifier
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", field):
            raise ValueError(
                f"Invalid field name in placeholder '{{{inner}}}'."
            )

        if field in sensitive_fields:
            raise ValueError(
                f"Sensitive placeholder '{{student.{field}}}' is not allowed."
            )

        if field not in allowed_fields:
            raise ValueError(
                f"Unknown placeholder '{{student.{field}}}'."
            )

    return True
