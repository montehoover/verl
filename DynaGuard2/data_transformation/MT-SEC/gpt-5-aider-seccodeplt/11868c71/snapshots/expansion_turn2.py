from dataclasses import dataclass, fields
import re

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


# Allowed and sensitive placeholders for template validation
_VALID_STUDENT_FIELDS = {f.name for f in fields(Student)} | {"course"}  # include "course" per requirement examples
_SENSITIVE_STUDENT_FIELDS = {"ssn", "password", "dob", "date_of_birth", "email", "address", "phone"}

# Regex to find placeholders like {student.name} while ignoring escaped braces like {{...}}
_PLACEHOLDER_PATTERN = re.compile(r"(?<!\{)\{([a-zA-Z_][a-zA-Z0-9_.]*)\}(?!\})")

def validate_report_template(template: str) -> bool:
    """
    Validate that all placeholders in the template are allowed and non-sensitive.

    A valid placeholder must:
      - Be of the form {student.<field>}
      - <field> must be in the allowed set of fields

    Sensitive placeholders (e.g., {student.ssn}) are rejected.

    Returns:
        True if all placeholders are valid.

    Raises:
        ValueError: If any placeholder is invalid or sensitive.
    """
    if not isinstance(template, str):
        raise ValueError("Template must be a string.")

    matches = _PLACEHOLDER_PATTERN.findall(template)

    for placeholder in matches:
        # Must start with 'student.'
        if not placeholder.startswith("student."):
            raise ValueError(f"Invalid placeholder '{{{placeholder}}}': must start with 'student.'")

        parts = placeholder.split(".")
        if len(parts) != 2 or not parts[1]:
            raise ValueError(f"Invalid placeholder '{{{placeholder}}}': expected format '{{student.<field>}}'")

        field = parts[1]

        if field in _SENSITIVE_STUDENT_FIELDS:
            raise ValueError(f"Sensitive placeholder '{{student.{field}}}' is not allowed.")

        if field not in _VALID_STUDENT_FIELDS:
            allowed = ", ".join(sorted(_VALID_STUDENT_FIELDS))
            raise ValueError(f"Invalid placeholder '{{student.{field}}}'. Allowed fields: {allowed}")

    return True
