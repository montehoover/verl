from typing import Any


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


class _SafeStudentProxy:
    """
    A proxy around a Student that exposes only non-sensitive attributes.
    Any attempt to access sensitive or unsupported attributes raises ValueError.
    """
    __slots__ = ("_student",)

    # Attributes explicitly allowed in templates
    _ALLOWED_ATTRS = {"name", "course", "age"}

    # Attributes considered sensitive and therefore forbidden
    _SENSITIVE_ATTRS = {"gpa", "home_address", "discipline_record"}

    def __init__(self, student: Student) -> None:
        self._student = student

    def __getattr__(self, attr: str) -> Any:
        # Block private/special attributes up front
        if attr.startswith("_"):
            raise ValueError(f"Access to attribute '{attr}' is not allowed")

        if attr in self._SENSITIVE_ATTRS:
            raise ValueError(f"Access to sensitive attribute '{attr}' is prohibited")

        if attr in self._ALLOWED_ATTRS:
            return getattr(self._student, attr)

        # Explicitly block anything not in the allow-list
        raise ValueError(f"Unknown or unsupported attribute '{attr}' in template")


def build_student_profile(unique_id: str, report_template: str) -> str:
    """
    Generate a formatted student report from a template.

    Args:
        unique_id: A unique string representing the student ID.
        report_template: A format string containing placeholders such as
                         '{student.name}', '{student.course}', '{student.age}'.

    Returns:
        A formatted string representing the student's report.

    Raises:
        ValueError: If the unique_id does not exist, or if the template attempts
                    to access sensitive or unsupported attributes.
    """
    if unique_id not in students:
        raise ValueError(f"Student with ID '{unique_id}' does not exist")

    student = students[unique_id]
    proxy = _SafeStudentProxy(student)

    # Perform formatting with a restricted proxy to prevent access to sensitive data
    try:
        # Only 'student' is provided to the template; access control is enforced by the proxy.
        return report_template.format(student=proxy)
    except (KeyError, AttributeError, ValueError) as exc:
        # Normalize to ValueError per requirements
        raise ValueError(str(exc)) from None
