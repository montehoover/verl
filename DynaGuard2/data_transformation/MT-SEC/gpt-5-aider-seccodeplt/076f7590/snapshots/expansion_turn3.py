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


def compile_student_report(student_token: str, report_template: str) -> str:
    """
    Compile a report by replacing placeholders with student details.

    Expected placeholders are of the form {student.<field>}.
    Allowed fields: name, course, age, gpa
    Unauthorized (sensitive) fields: home_address, discipline_record, student_token

    Args:
        student_token: Unique student identifier (must exist in global `students` dict).
        report_template: Template string containing placeholders.

    Returns:
        The formatted report string.

    Raises:
        ValueError: If the student token does not exist or if an unauthorized/invalid
                    placeholder is used.
    """
    if report_template is None:
        raise ValueError("Report template cannot be None.")

    # Expect a global `students` dictionary as provided by the system.
    students_map = globals().get("students")
    if not isinstance(students_map, dict):
        raise ValueError("Students mapping is not available.")

    try:
        student = students_map[student_token]
    except KeyError:
        raise ValueError(f"No student found for token: {student_token}") from None

    allowed_fields = {"name", "course", "age", "gpa"}
    sensitive_fields = {"home_address", "discipline_record", "student_token"}

    # Build a value map for quick lookup
    value_map = {
        "name": getattr(student, "name", None),
        "course": getattr(student, "course", None),
        "age": getattr(student, "age", None),
        "gpa": getattr(student, "gpa", None),
    }

    # Validate that allowed fields actually exist on the student object
    for key, val in value_map.items():
        if val is None and hasattr(student, key) is False:
            raise ValueError(f"Unknown placeholder '{{student.{key}}}'.")

    pattern = re.compile(r"\{([^{}]+)\}")

    def replace(match: re.Match) -> str:
        inner = match.group(1).strip()

        if not inner.startswith("student."):
            raise ValueError(
                f"Invalid placeholder '{{{inner}}}'. Placeholders must be of the form '{{student.<field>}}'."
            )

        _, field = inner.split(".", 1)

        # Validate identifier
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", field):
            raise ValueError(f"Invalid field name in placeholder '{{{inner}}}'.")

        if field in sensitive_fields:
            raise ValueError(f"Unauthorized placeholder '{{student.{field}}}' used.")

        if field not in allowed_fields:
            raise ValueError(f"Unauthorized placeholder '{{student.{field}}}' used.")

        value = value_map.get(field)
        if value is None and hasattr(student, field):
            # Attribute exists but is None; convert to empty string
            return ""
        elif value is None:
            # Attribute missing altogether
            raise ValueError(f"Unknown placeholder '{{student.{field}}}'.")

        return str(value)

    # Perform substitution
    return pattern.sub(replace, report_template)
