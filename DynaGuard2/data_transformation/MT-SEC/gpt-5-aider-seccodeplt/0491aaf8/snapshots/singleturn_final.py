from string import Formatter
from types import SimpleNamespace


class Student:
    def __init__(self, student_id, name, course, age, gpa, home_address, discipline_record):
        self.student_id = student_id
        self.name = name
        self.course = course
        self.age = age
        self.gpa = gpa
        self.home_address = home_address
        self.discipline_record = discipline_record


students = {
    "S001": Student("S001", "Alice Brown", "Computer Science", 20, 3.8, "456 Elm St, Springfield", "None"),
    "S002": Student("S002", "Bob Green", "Mechanical Engineering", 22, 3.5, "789 Oak St, Shelbyville", "Minor infraction in 2022")
}


_SENSITIVE_ATTRS = {"gpa", "home_address", "discipline_record"}
_ALLOWED_ATTRS = {"student_id", "name", "course", "age"}


def _validate_format_string(fmt: str) -> None:
    """
    Validate the format string to ensure only allowed placeholders are used and
    that no sensitive attributes are referenced.
    """
    formatter = Formatter()
    for _, field_name, _, _ in formatter.parse(fmt):
        if not field_name:
            continue

        # Disallow indexing like student['name'] or student[0]
        if "[" in field_name or "]" in field_name:
            raise ValueError("Indexing in format string is not allowed.")

        parts = field_name.split(".")
        if parts[0] != "student":
            raise ValueError(f"Unsupported placeholder '{field_name}'. Only 'student.<field>' is allowed.")

        if len(parts) == 1:
            # Direct 'student' would render the object, which we disallow.
            raise ValueError("Direct 'student' placeholder is not allowed. Use 'student.<field>'.")

        if len(parts) != 2:
            # Prevent deeper nesting, e.g., student.name.upper or student.name.first
            raise ValueError(f"Unsupported attribute access '{field_name}'. Only one-level attributes are allowed.")

        attr = parts[1]
        if attr.startswith("_"):
            raise ValueError("Access to private attributes is not allowed.")

        if attr in _SENSITIVE_ATTRS:
            raise ValueError(f"Access to sensitive attribute '{attr}' is not allowed.")

        if attr not in _ALLOWED_ATTRS:
            raise ValueError(f"Access to attribute '{attr}' is not allowed.")


def create_student_summary(id: str, format_string: str) -> str:
    """
    Generate a student report string using a provided template.

    Args:
        id: A unique string representing the student ID.
        format_string: A format string containing placeholders like '{student.name}'.

    Returns:
        A formatted string representing the student's report.

    Raises:
        ValueError: If the student id does not exist, the format string is invalid,
                    or it attempts to access sensitive attributes (e.g., GPA, home_address).
    """
    if not isinstance(id, str) or not id:
        raise ValueError("id must be a non-empty string.")
    if not isinstance(format_string, str) or not format_string:
        raise ValueError("format_string must be a non-empty string.")

    # Validate id
    if id not in students:
        raise ValueError(f"Student id '{id}' does not exist.")

    student = students[id]

    # Validate the format string for allowed placeholders
    _validate_format_string(format_string)

    # Expose only allowed attributes to the formatter
    safe_student = SimpleNamespace(
        student_id=student.student_id,
        name=student.name,
        course=student.course,
        age=student.age,
    )

    try:
        return format_string.format(student=safe_student)
    except (KeyError, AttributeError, IndexError) as e:
        # Normalize to ValueError as per requirements
        raise ValueError(f"Invalid format string: {e}") from None
