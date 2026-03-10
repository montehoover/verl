from string import Formatter
from types import SimpleNamespace
import logging


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

# Define allowed and sensitive fields as module-level constants for maintainability
SENSITIVE_FIELDS = frozenset({"home_address", "discipline_record"})
ALLOWED_FIELDS = frozenset({"student_id", "name", "course", "age", "gpa"})


def _render_student_template(student: Student, template: str) -> str:
    """
    Pure function that validates and renders the given template using the provided student.
    Only placeholders of the form 'student.<field>' are allowed, and sensitive fields are blocked.
    """
    if not isinstance(template, str):
        raise ValueError("Invalid arguments: template must be a string.")

    formatter = Formatter()
    for _, field_name, _, _ in formatter.parse(template):
        if field_name is None:
            continue

        # Only allow placeholders of the form 'student.<field>'
        if field_name.startswith("student"):
            rest = field_name[len("student"):]
            if not rest:
                # Direct 'student' reference is not allowed
                raise ValueError("Invalid template: direct 'student' reference is not allowed.")
            if rest[0] != ".":
                # Disallow indexing or other forms like 'student[field]' or 'student__dict__'
                raise ValueError("Invalid template: only attribute access 'student.<field>' is allowed.")
            attr = rest[1:]

            # Disallow nested attributes or indexing
            if not attr or "." in attr or "[" in attr or "]" in attr:
                raise ValueError("Invalid template: unsupported attribute path.")

            if attr in SENSITIVE_FIELDS:
                raise ValueError("Access to sensitive student information is not allowed.")
            if attr not in ALLOWED_FIELDS:
                raise ValueError(f"Invalid template: unknown field '{attr}'.")
        else:
            # Disallow any other top-level placeholders
            raise ValueError("Invalid template: only 'student.<field>' placeholders are allowed.")

    safe_student = SimpleNamespace(
        student_id=student.student_id,
        name=student.name,
        course=student.course,
        age=student.age,
        gpa=student.gpa,
    )

    try:
        return template.format(student=safe_student)
    except Exception as exc:
        raise ValueError("Failed to generate report from the provided template.") from exc


def generate_student_report(student_id: str, template: str) -> str:
    """
    Generate a student report string from a template.

    Args:
        student_id: The unique identifier of the student.
        template: A string template that may contain placeholders like '{student.name}'.

    Returns:
        The generated report string.

    Raises:
        ValueError: If the student is not found, the template is invalid,
                    or if access to sensitive information is attempted.
    """
    if not isinstance(student_id, str) or not isinstance(template, str):
        raise ValueError("Invalid arguments: student_id and template must be strings.")

    if student_id not in students:
        raise ValueError("Student not found.")

    # Initialize logging within the function for human-readable logs
    logger = logging.getLogger("student_report")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt="(%(levelname)s) %(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    logger.info("Generating report for student_id=%s", student_id)
    logger.info("Using template: %s", template)

    student = students[student_id]
    report = _render_student_template(student, template)

    logger.info("Generated report: %s", report)
    return report
