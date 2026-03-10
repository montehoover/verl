import re

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
        return students[sid]
    except KeyError:
        raise ValueError(f"Student ID '{sid}' does not exist.") from None


# Allowed and sensitive placeholders for template validation
_STUDENT_ALLOWED_FIELDS = {
    "student_id",
    "name",
    "course",
    "age",
    "gpa",
    "home_address",
    "discipline_record",
}
_STUDENT_SENSITIVE_FIELDS = {
    "home_address",
    "discipline_record",
}

# Regex to find placeholders like {student.name} while ignoring escaped braces like {{...}}
_PLACEHOLDER_PATTERN = re.compile(r"(?<!\{)\{([a-zA-Z_][a-zA-Z0-9_.]*)\}(?!\})")

def validate_report_template(template: str) -> bool:
    """
    Validate that all placeholders in the template are allowed and non-sensitive.

    A valid placeholder must:
      - Be of the form {student.<field>}
      - <field> must be in the allowed set of fields

    Sensitive placeholders (e.g., {student.home_address}) are rejected.

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

        if field in _STUDENT_SENSITIVE_FIELDS:
            raise ValueError(f"Sensitive placeholder '{{student.{field}}}' is not allowed.")

        if field not in _STUDENT_ALLOWED_FIELDS:
            allowed = ", ".join(sorted(_STUDENT_ALLOWED_FIELDS))
            raise ValueError(f"Invalid placeholder '{{student.{field}}}'. Allowed fields: {allowed}")

    return True


def build_student_profile(unique_id: str, report_template: str) -> str:
    """
    Build a formatted student report from a template.

    Args:
        unique_id: The student's unique ID (e.g., "S001").
        report_template: Template string with placeholders like {student.name}, {student.course}, etc.

    Returns:
        The formatted report string.

    Raises:
        ValueError: If the student ID does not exist or if the template contains sensitive/invalid placeholders.
    """
    # Fetch student or raise if not found
    student = get_student_by_id(unique_id)

    # Validate template (raises ValueError on invalid or sensitive placeholders)
    validate_report_template(report_template)

    # Perform formatting using the student object for attribute access
    return report_template.format(student=student)
