from string import Formatter
from typing import Set


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
    "S001": Student(
        "S001",
        "Alice Brown",
        "Computer Science",
        20,
        3.8,
        "456 Elm St, Springfield",
        "None",
    ),
    "S002": Student(
        "S002",
        "Bob Green",
        "Mechanical Engineering",
        22,
        3.5,
        "789 Oak St, Shelbyville",
        "Minor infraction in 2022",
    ),
}


def create_student_summary(id: str, format_string: str) -> str:
    """
    Generate a student report string using a customizable format template.

    Args:
        id: The unique student ID.
        format_string: A format string (e.g.,
            "Name: {student.name}, Course: {student.course}, Age: {student.age}")
            Placeholders may reference allowed attributes via {student.<attr>}
            or directly as {<attr>}.

    Returns:
        A formatted string representing the student's report.

    Raises:
        ValueError: If the student ID does not exist, or if the format string
        attempts to access sensitive or unsupported attributes (e.g., gpa,
        home_address, discipline_record).
    """
    if id not in students:
        raise ValueError(f"Student ID '{id}' does not exist.")

    student_obj = students[id]

    # Define allowed and sensitive attributes
    sensitive_attributes: Set[str] = {"gpa", "home_address", "discipline_record"}
    allowed_attributes: Set[str] = {"student_id", "name", "course", "age"}

    # Validate placeholders to ensure no sensitive or unsupported attributes are accessed
    format_parser = Formatter()

    for _, field_name, _, _ in format_parser.parse(format_string):
        if not field_name:
            continue

        # Disallow any bracket-based indexing to avoid bypass (e.g., student['__dict__'])
        if "[" in field_name or "]" in field_name:
            raise ValueError("Invalid placeholder usage with indexing is not allowed.")

        # Support both "{student.attr}" and "{attr}" forms
        field_parts = field_name.split(".")
        if field_parts[0] == "student":
            field_parts = field_parts[1:]

        # Referencing the whole student object (i.e., "{student}") is not useful; disallow to be strict.
        if len(field_parts) == 0:
            raise ValueError(
                "Referencing the entire student object is not allowed; use specific attributes."
            )

        # Validate each attribute referenced is allowed and not sensitive
        for attr_name in field_parts:
            if attr_name in sensitive_attributes:
                raise ValueError(f"Access to sensitive attribute '{attr_name}' is not allowed.")
            if attr_name not in allowed_attributes:
                raise ValueError(f"Access to unsupported attribute '{attr_name}' is not allowed.")

    # Build a safe context for formatting
    format_context = {"student": student_obj}

    for attr_name in allowed_attributes:
        format_context[attr_name] = getattr(student_obj, attr_name)

    return format_string.format(**format_context)
