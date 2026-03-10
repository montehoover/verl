from dataclasses import dataclass, fields
from typing import Dict
import string


@dataclass(frozen=True)
class Student:
    id: str
    name: str
    age: int
    major: str


# Predefined dictionary of students keyed by their string ID
STUDENTS_BY_ID: Dict[str, Student] = {
    "S001": Student(id="S001", name="Alice Johnson", age=20, major="Computer Science"),
    "S002": Student(id="S002", name="Bob Smith", age=21, major="Mathematics"),
    "S003": Student(id="S003", name="Carla Gomez", age=22, major="Physics"),
    "S004": Student(id="S004", name="David Lee", age=19, major="Chemistry"),
    "S005": Student(id="S005", name="Evelyn Chen", age=23, major="Biology"),
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
        return STUDENTS_BY_ID[student_id]
    except KeyError:
        raise ValueError(f"Student ID '{student_id}' does not exist.")


def check_placeholders(template: str) -> bool:
    """
    Validate that all placeholders in the template are permitted and do not
    reference sensitive student information.

    Allowed placeholders: {student.<field>} where <field> is one of the fields
    defined on the Student dataclass (id, name, age, major).
    Disallowed examples include: {student.gpa}, {student.home_address},
    {student}, {student.name.extra}, or any placeholder not starting with 'student.'.

    Returns True if all placeholders are valid, otherwise raises ValueError.
    """
    formatter = string.Formatter()
    allowed_student_attrs = {f.name for f in fields(Student)}
    sensitive_attrs = {"gpa", "home_address"}

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
