import re


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


_SENSITIVE_FIELDS = {"gpa", "home_address", "discipline_record"}
_ALLOWED_OUTPUT_FIELDS = ("name", "course", "age")

# Placeholders mapping for allowed, non-sensitive fields
_ALLOWED_PLACEHOLDER_TO_ATTR = {
    "student_name": "name",
    "student_course": "course",
    "student_age": "age",
}

# Any attempt to reference these placeholders in the template should raise an error
_SENSITIVE_PLACEHOLDER_NAMES = {
    "student_gpa",
    "student_home_address",
    "student_discipline_record",
    # Also guard against direct attribute names without the 'student_' prefix
    "gpa",
    "home_address",
    "discipline_record",
}


def print_student_details(student_token: str, format_template: str) -> str:
    """
    Returns a formatted string of non-sensitive student details based on the provided template.

    The template may include placeholders:
      - {student_name}, {student_course}, {student_age}

    Any attempt to include sensitive placeholders such as:
      - {student_gpa}, {student_home_address}, {student_discipline_record}
    (or their direct attribute counterparts without 'student_' prefix) will raise ValueError.

    Missing or unknown placeholders are left unchanged in the output.

    Raises:
        ValueError: If the student_token does not exist or if an unauthorized access
                    to sensitive data is attempted.
    """
    # Validate token existence
    student = students.get(student_token)
    if student is None:
        raise ValueError("Student not found for the provided token.")

    # Extract placeholders in the template
    placeholders = set(m.group(1) for m in re.finditer(r"\{([a-zA-Z0-9_]+)\}", format_template or ""))

    # Check for sensitive placeholders
    if placeholders & _SENSITIVE_PLACEHOLDER_NAMES:
        raise ValueError("Unauthorized access to sensitive data is attempted.")

    # Perform safe, selective replacement for allowed placeholders only
    result = format_template
    for ph, attr in _ALLOWED_PLACEHOLDER_TO_ATTR.items():
        if ph in placeholders:
            result = result.replace("{" + ph + "}", str(getattr(student, attr)))

    return result


def compile_student_report(student_token: str, report_template: str) -> str:
    """
    Compiles a report by replacing placeholders like {student.name} or {student.course}
    with actual non-sensitive data from a student object.

    Rules:
    - Allowed fields: name, course, age
    - Sensitive fields (gpa, home_address, discipline_record) will raise ValueError if referenced.
    - Unknown placeholders are left unchanged.
    - Only placeholders prefixed with 'student.' are processed; others are left unchanged.

    Args:
        student_token: The token identifying the student.
        report_template: The template string containing placeholders.

    Returns:
        The generated report string.

    Raises:
        ValueError: If the student_token does not exist or if an attempt to access
                    sensitive data is detected.
    """
    student = students.get(student_token)
    if student is None:
        raise ValueError("Student not found for the provided token.")

    pattern = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_\.]*)\}")
    text = report_template or ""

    def _replace(match: re.Match) -> str:
        key = match.group(1)

        # Only process placeholders that start with 'student.'
        if not key.startswith("student."):
            return match.group(0)

        # Extract the first attribute after 'student.'
        tail = key.split(".", 1)[1] if "." in key else ""
        primary_attr = tail.split(".", 1)[0] if tail else ""

        if not primary_attr:
            # Nothing meaningful to replace; leave as is
            return match.group(0)

        # Block sensitive fields explicitly
        if primary_attr in _SENSITIVE_FIELDS:
            raise ValueError("Unauthorized access to sensitive data is attempted.")

        # Replace allowed attributes; leave unknowns unchanged
        if primary_attr in _ALLOWED_OUTPUT_FIELDS:
            return str(getattr(student, primary_attr))

        return match.group(0)

    return pattern.sub(_replace, text)
