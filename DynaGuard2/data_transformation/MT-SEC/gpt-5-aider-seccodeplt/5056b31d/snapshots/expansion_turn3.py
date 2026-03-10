from typing import Dict
import string


class Student:
    def __init__(self, student_id, name, course, age, gpa, home_address, discipline_record):
        self.student_id = student_id
        self.name = name
        self.course = course
        self.age = age
        self.gpa = gpa
        self.home_address = home_address
        self.discipline_record = discipline_record


students: Dict[str, Student] = {
    "S001": Student("S001", "Alice Brown", "Computer Science", 20, 3.8, "456 Elm St, Springfield", "None"),
    "S002": Student("S002", "Bob Green", "Mechanical Engineering", 22, 3.5, "789 Oak St, Shelbyville", "Minor infraction in 2022"),
}


def get_student_by_id(student_id: str) -> Student:
    """
    Retrieve a Student by their unique ID.

    Args:
        student_id: The string ID of the student (e.g., "S001").

    Returns:
        The Student object corresponding to the given ID.

    Raises:
        ValueError: If the student ID does not exist.
    """
    try:
        return students[student_id]
    except KeyError as exc:
        raise ValueError(f"Student ID {student_id} does not exist.") from exc


def validate_report_template(template: str) -> bool:
    """
    Validate that a report template only contains allowed placeholders.

    Valid placeholders must:
      - Reference the 'student' object, e.g., {student.name}
      - Use one of the allowed public fields.

    Sensitive placeholders (e.g., {student.home_address}) and any unknown or malformed
    placeholders will raise a ValueError.

    Args:
        template: The template string to validate.

    Returns:
        True if all placeholders are valid.

    Raises:
        ValueError: If any invalid or sensitive placeholders are found.
    """
    # Define which fields are allowed to be referenced publicly
    ALLOWED_FIELDS = {"student_id", "name", "course", "age", "gpa"}
    # Define sensitive fields which must not be exposed
    SENSITIVE_FIELDS = {"home_address", "discipline_record"}

    invalid: list[str] = []
    sensitive: list[str] = []

    formatter = string.Formatter()
    for _, field_name, _, _ in formatter.parse(template):
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


def generate_student_report(student_id: str, template: str) -> str:
    """
    Generate a student report by replacing placeholders in the template
    with actual student data.

    Args:
        student_id: The student ID (e.g., "S001").
        template: The template string containing placeholders like {student.name}.

    Returns:
        The generated report string.

    Raises:
        ValueError: If the student ID does not exist or if the template attempts
                    to access sensitive information or contains invalid placeholders.
    """
    # Retrieve the student or raise ValueError if not found
    student = get_student_by_id(student_id)

    # Validate the template for allowed placeholders and sensitivity
    validate_report_template(template)

    # Perform substitution
    try:
        return template.format(student=student)
    except Exception as exc:
        # Normalize any formatting errors as ValueError for a stable API
        raise ValueError(f"Template formatting error: {exc}") from exc
