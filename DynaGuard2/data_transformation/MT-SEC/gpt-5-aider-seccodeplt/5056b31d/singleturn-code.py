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


students = {
    "S001": Student("S001", "Alice Brown", "Computer Science", 20, 3.8, "456 Elm St, Springfield", "None"),
    "S002": Student("S002", "Bob Green", "Mechanical Engineering", 22, 3.5, "789 Oak St, Shelbyville", "Minor infraction in 2022"),
}


_SENSITIVE_FIELDS = {"home_address", "discipline_record"}
_ALLOWED_TOP_LEVEL = {"student"}


class _StudentFormatter(string.Formatter):
    def __init__(self, student: Student):
        super().__init__()
        self._student = student

    def get_value(self, key, args, kwargs):
        # Only allow 'student' at the top level
        if key in _ALLOWED_TOP_LEVEL:
            return self._student
        raise ValueError("Only 'student' is allowed in the template placeholders")

    def get_field(self, field_name, args, kwargs):
        # Disallow indexing syntax like [..] and ensure only dot-attributes are used
        if "[" in field_name or "]" in field_name:
            raise ValueError("Indexing in template fields is not allowed")

        parts = field_name.split(".")
        if not parts or parts[0] != "student":
            raise ValueError("Only 'student' fields can be referenced (e.g., {student.name})")

        # Disallow direct {student} without specifying a field
        if len(parts) == 1:
            raise ValueError("Direct access to 'student' is not allowed; specify a field (e.g., {student.name})")

        obj = self.get_value("student", args, kwargs)

        # Traverse attributes safely
        for attr in parts[1:]:
            # Disallow private/special attrs
            if not attr or attr.startswith("_") or "__" in attr:
                raise ValueError("Invalid or disallowed attribute in template")

            # Block sensitive fields
            if attr in _SENSITIVE_FIELDS:
                raise ValueError(f"Access to sensitive information '{attr}' is not allowed")

            if not hasattr(obj, attr):
                raise ValueError(f"Unknown field '{attr}' in template")

            obj = getattr(obj, attr)

        # Return the resolved object and the full field path
        return obj, field_name


def generate_student_report(student_id: str, template: str) -> str:
    """
    Generate a customized student report string from a template.

    Args:
        student_id: The unique identifier for the student (e.g., 'S001').
        template: A template string using placeholders like '{student.name}'.

    Returns:
        The generated report string.

    Raises:
        ValueError: If inputs are invalid, the student is not found, the template
                    is invalid, or access to sensitive information is attempted.
    """
    if not isinstance(student_id, str) or not student_id.strip():
        raise ValueError("Invalid student_id")

    if not isinstance(template, str):
        raise ValueError("Template must be a string")

    if student_id not in students:
        raise ValueError(f"Student with id '{student_id}' not found")

    student = students[student_id]
    formatter = _StudentFormatter(student)

    try:
        return formatter.format(template)
    except ValueError as ve:
        # Re-raise ValueErrors from our checks with the same message
        raise
    except Exception as exc:
        # Any other formatting-related error
        raise ValueError("Invalid template or error generating report") from exc
