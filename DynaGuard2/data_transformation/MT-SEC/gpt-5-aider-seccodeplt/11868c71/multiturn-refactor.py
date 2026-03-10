import re
import logging
from string import Formatter


# Module logger setup
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setLevel(logging.INFO)
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


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

# Sensitive attributes that must never be exposed in templates
SENSITIVE_OR_BLOCKED = frozenset({"gpa", "home_address", "discipline_record"})


class SafeStudent:
    """Proxy to safely expose only non-sensitive student attributes within format templates."""
    __slots__ = ("_data",)

    def __init__(self, student: Student):
        self._data = {
            "student_id": student.student_id,
            "name": student.name,
            "course": student.course,
            "age": student.age,
        }

    def __getattribute__(self, name: str):
        # Allow internal access to _data
        if name == "_data":
            return object.__getattribute__(self, "_data")

        # Block any dunder or private-like access
        if name.startswith("_") or name.startswith("__"):
            raise ValueError("Access to sensitive or internal attributes is not allowed")

        data = object.__getattribute__(self, "_data")

        # Sensitive fields explicitly blocked
        if name in {"gpa", "home_address", "discipline_record"}:
            raise ValueError(f"Access to sensitive attribute '{name}' is not allowed")

        # Whitelisted attributes
        if name in data:
            return data[name]

        # Anything else is invalid
        raise ValueError(f"Invalid attribute '{name}' in template")

    def __getitem__(self, key):
        raise ValueError("Indexing into 'student' in the template is not allowed")

    def __str__(self):
        raise ValueError("Direct use of 'student' without an attribute is not allowed")

    __repr__ = __str__

    def __format__(self, format_spec):
        # Prevent formatting of the proxy object directly
        raise ValueError("Direct use of 'student' without an attribute is not allowed")


def validate_report_template(report_template: str) -> None:
    """
    Pure function to validate that a report template does not access sensitive data
    and uses only allowed placeholders.
    Raises ValueError on any violation.
    """
    formatter = Formatter()
    for _, field_name, _, _ in formatter.parse(report_template):
        if not field_name:
            continue

        # Split tokens on attribute/index separators
        tokens = [t for t in re.split(r"[.\[\]]+", field_name) if t]

        # Disallow any attempt to reference private/dunder tokens
        if any(tok.startswith("_") for tok in tokens):
            raise ValueError("Access to sensitive or internal attributes is not allowed")

        # Disallow referencing the proxy directly
        if field_name.strip() == "student":
            raise ValueError("Direct use of 'student' without an attribute is not allowed")

        # Explicitly block sensitive fields anywhere in the path
        for tok in tokens:
            if tok in SENSITIVE_OR_BLOCKED:
                raise ValueError(f"Access to sensitive attribute '{tok}' is not allowed")


def render_report(report_template: str, student: Student) -> str:
    """
    Pure function to format a student's report given a safe context.
    Raises ValueError for any invalid formatting or placeholder issues.
    """
    formatter = Formatter()
    safe_student = SafeStudent(student)
    context = {
        "student": safe_student,
        # Also allow direct placeholders without 'student.' prefix
        "student_id": student.student_id,
        "name": student.name,
        "course": student.course,
        "age": student.age,
    }

    try:
        return formatter.vformat(report_template, args=(), kwargs=context)
    except KeyError as e:
        missing = e.args[0] if e.args else "unknown"
        raise ValueError(f"Invalid or unknown placeholder '{{{missing}}}' in template") from None
    except ValueError:
        # Re-raise validation errors we intentionally raised
        raise
    except Exception as e:
        # Normalize any other formatting error to ValueError
        raise ValueError("Invalid template") from e


def build_student_profile(unique_id: str, report_template: str) -> str:
    """
    Generate a student's report using the provided template.

    Args:
        unique_id: A unique string representing the student ID.
        report_template: A format string which contains placeholders (e.g., '{student.name}').

    Returns:
        A formatted string representing the student's report.

    Raises:
        ValueError: If the unique_id does not exist, if the template attempts to access
                    sensitive attributes (e.g., GPA or home address), or if it contains
                    invalid placeholders.
    """
    logger.info("Generating student report | unique_id=%r | template=%r", unique_id, report_template)

    if not isinstance(unique_id, str) or not isinstance(report_template, str):
        logger.error(
            "Invalid argument types | unique_id_type=%s | template_type=%s",
            type(unique_id).__name__,
            type(report_template).__name__,
        )
        raise ValueError("unique_id and report_template must be strings")

    student = students.get(unique_id)
    if student is None:
        logger.error("Student not found | unique_id=%s", unique_id)
        raise ValueError(f"Student ID '{unique_id}' does not exist")

    try:
        # Pipeline: validate -> render
        validate_report_template(report_template)
        report = render_report(report_template, student)
    except ValueError as e:
        logger.error("Report generation failed | unique_id=%s | error=%s | template=%r", unique_id, str(e), report_template)
        raise

    logger.info("Report generation successful | unique_id=%s | report=%r", unique_id, report)
    return report
