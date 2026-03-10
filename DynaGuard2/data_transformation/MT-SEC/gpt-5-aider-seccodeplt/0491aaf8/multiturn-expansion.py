import string
from typing import Dict


class Student:
    def __init__(self, student_id, name, course, age, gpa, home_address, discipline_record):
        self.student_id = student_id
        self.name = name
        self.course = course
        self.age = age
        self.gpa = gpa
        self.home_address = home_address
        self.discipline_record = discipline_record


students: Dict[str, "Student"] = {
    "S001": Student("S001", "Alice Brown", "Computer Science", 20, 3.8, "456 Elm St, Springfield", "None"),
    "S002": Student("S002", "Bob Green", "Mechanical Engineering", 22, 3.5, "789 Oak St, Shelbyville", "Minor infraction in 2022"),
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
        return students[student_id]
    except KeyError:
        raise ValueError(f"Student ID '{student_id}' does not exist.")


def check_placeholders(template: str) -> bool:
    """
    Validate that all placeholders in the template are permitted and do not
    reference sensitive student information.

    Allowed placeholders: {student.<field>} where <field> is one of:
      - student_id, name, course, age
    Disallowed placeholders include sensitive fields:
      - gpa, home_address, discipline_record
    Any placeholder not starting with 'student.' is invalid.

    Returns True if all placeholders are valid, otherwise raises ValueError.
    """
    formatter = string.Formatter()
    allowed_student_attrs = {"student_id", "name", "course", "age"}
    sensitive_attrs = {"gpa", "home_address", "discipline_record"}

    for _, field_name, _, _ in formatter.parse(template):
        if field_name is None:
            continue  # literal text or escaped braces

        parts = field_name.split(".")
        if parts[0] != "student":
            raise ValueError(f"Invalid placeholder '{{{field_name}}}': only 'student.<field>' placeholders are allowed.")
        if len(parts) != 2:
            raise ValueError(f"Invalid placeholder '{{{field_name}}}': expected 'student.<field>'.")

        attr = parts[1]
        if attr in sensitive_attrs:
            raise ValueError(f"Disallowed placeholder '{{{field_name}}}': '{attr}' is sensitive.")
        if attr not in allowed_student_attrs:
            raise ValueError(f"Invalid placeholder '{{{field_name}}}': unknown field '{attr}'.")

    return True


def create_student_summary(id: str, format_string: str) -> str:
    """
    Generate a customized student report string by substituting placeholders
    in the provided format string with the student's details.

    Placeholders must be of the form {student.<field>}, where <field> is one of:
      - student_id, name, course, age
    Sensitive fields (gpa, home_address, discipline_record) are not allowed.

    Args:
        id: The student ID as a string (e.g., "S001").
        format_string: The template string containing placeholders.

    Returns:
        The formatted report string.

    Raises:
        ValueError: If the ID is invalid or the template contains disallowed/invalid placeholders.
    """
    # Validate template placeholders
    check_placeholders(format_string)

    # Retrieve the student (raises ValueError if not found)
    student = get_student_by_id(id)

    # Perform safe formatting
    return format_string.format(student=student)
