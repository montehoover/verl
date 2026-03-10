import string


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


class SafeStudent:
    """
    Proxy object exposing only allowed non-sensitive student attributes.
    """
    __slots__ = ("name", "course", "age")

    def __init__(self, name: str, course: str, age: int):
        self.name = name
        self.course = course
        self.age = age

    def __getattr__(self, item):
        # Any attempt to access non-whitelisted attribute should fail
        raise AttributeError(f"Access to attribute '{item}' is not allowed")


def compile_student_report(student_token: str, report_template: str) -> str:
    """
    Generate a student report from a template while strictly excluding sensitive data.

    Args:
        student_token: Unique identifier for the student.
        report_template: A format string that may include placeholders like
                         '{student.name}', '{student.course}', or '{student.age}'.

    Returns:
        The generated report as a formatted string.

    Raises:
        ValueError: If the student_token does not exist, or if the template attempts
                    to access sensitive or unauthorized data.
    """
    # Validate student existence
    student = students.get(student_token)
    if student is None:
        raise ValueError("Student token does not exist")

    allowed_fields = {"name", "course", "age"}

    # Validate placeholders in the template to prevent access to sensitive data
    formatter = string.Formatter()
    for _, field_name, format_spec, conversion in formatter.parse(report_template):
        if field_name is None:
            continue  # literal text only

        # Disallow implicit positional args or empty fields like '{}'
        if field_name == "":
            raise ValueError("Unauthorized placeholder in template")

        # Disallow indexing or any bracket-based access
        if "[" in field_name or "]" in field_name:
            raise ValueError("Unauthorized access pattern in template")

        # Enforce that only {student.<allowed_field>} is used
        if field_name == "student":
            # Disallow exposing the object itself
            raise ValueError("Unauthorized placeholder 'student' in template; attribute required")

        if not field_name.startswith("student."):
            raise ValueError("Unauthorized placeholder in template")

        parts = field_name.split(".")
        if len(parts) != 2 or parts[1] not in allowed_fields:
            raise ValueError("Unauthorized access to sensitive or unsupported attributes in template")

        # Restrict conversions (allow only default/!s)
        if conversion not in (None, "s"):
            raise ValueError("Unsupported conversion in template")

        # format_spec is allowed (e.g., alignment/width), no sensitive impact

    # Build a safe proxy with only allowed fields
    safe_student = SafeStudent(name=student.name, course=student.course, age=student.age)

    try:
        return report_template.format(student=safe_student)
    except Exception as exc:
        # Normalize unexpected formatting errors into ValueError
        raise ValueError(f"Failed to compile report: {exc}") from exc
