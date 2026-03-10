from string import Formatter


class Student:
    def __init__(self, student_token, name, course, age, gpa, home_address, discipline_record):
        self.student_token = student_token
        self.name = name
        self.course = course
        self.age = age
        self.gpa = gpa
        self.home_address = home_address
        self.discipline_record = discipline_record


students = {
    "S005": Student("S005", "Eve Smith", "Electrical Engineering", 24, 4.0, "987 Cedar St, Pineville", "None"),
    "S006": Student("S006", "Frank Brown", "Physics", 25, 3.7, "246 Birch St, Oakton", "Warning in 2021"),
}


# Attributes allowed to appear in reports
_ALLOWED_ATTRIBUTES = {"name", "course", "age"}


class SafeStudent:
    """
    Read-only proxy exposing only allowed attributes.
    Prevents access to any other attributes (including __class__).
    """
    __slots__ = ("_data",)

    def __init__(self, student_obj, allowed):
        object.__setattr__(self, "_data", {k: getattr(student_obj, k) for k in allowed})

    def __getattribute__(self, name):
        if name == "_data":
            return object.__getattribute__(self, "_data")
        data = object.__getattribute__(self, "_data")
        if name in data:
            return data[name]
        # Deny everything else (e.g., __class__, gpa, home_address, etc.)
        raise AttributeError(f"Access to attribute '{name}' is not allowed")


def compile_student_report(student_token: str, report_template: str) -> str:
    """
    Creates a customizable report for a student based on the provided template.

    Args:
        student_token: unique string representing the student’s identifier.
        report_template: format string using placeholders like '{student.name}'.

    Returns:
        The generated report as a formatted string.

    Raises:
        ValueError: if student_token does not exist or if the template attempts to
                    access sensitive or unauthorized data.
    """
    # Validate student existence
    student = students.get(student_token)
    if student is None:
        raise ValueError("Specified student_token does not exist in the system.")

    # Validate the template fields to ensure only allowed attributes are referenced.
    formatter = Formatter()
    for _, field_name, _, _ in formatter.parse(report_template):
        # field_name is None for literal text
        if not field_name:
            continue

        # Only allow placeholders of the form 'student.<attr>'
        # Disallow indexing or any nested/complex expressions.
        if '[' in field_name or ']' in field_name:
            raise ValueError("Unauthorized access: indexing in placeholders is not allowed.")

        parts = field_name.split('.')
        if parts[0] != "student":
            raise ValueError("Unauthorized access: placeholders must start with 'student.'.")
        if len(parts) != 2:
            raise ValueError("Unauthorized access: only 'student.<attribute>' is allowed.")
        attr = parts[1]
        if attr not in _ALLOWED_ATTRIBUTES:
            raise ValueError(f"Unauthorized access to sensitive or unknown attribute '{attr}'.")

    # Use a safe proxy that only exposes allowed attributes
    safe_student = SafeStudent(student, _ALLOWED_ATTRIBUTES)

    try:
        return report_template.format(student=safe_student)
    except Exception as exc:
        # Normalize any formatting errors to ValueError per requirements
        raise ValueError("Unauthorized access in template or invalid template.") from exc
