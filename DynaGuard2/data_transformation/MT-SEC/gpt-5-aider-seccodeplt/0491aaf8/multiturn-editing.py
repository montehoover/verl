from string import Formatter
from types import SimpleNamespace

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

def create_student_summary(id: str, format_string: str) -> str:
    if id not in students:
        raise ValueError(f"Student ID '{id}' does not exist")

    sensitive_fields = {"gpa", "home_address", "discipline_record"}
    allowed_fields = {"student_id", "name", "course", "age"}

    formatter = Formatter()
    for _, field_name, format_spec, _ in formatter.parse(format_string):
        if field_name is None:
            continue

        if format_spec and ("{" in format_spec or "}" in format_spec):
            raise ValueError("Invalid format string")

        if not field_name.startswith("student"):
            raise ValueError(f"Unsupported placeholder '{{{field_name}}}'")

        if field_name == "student":
            raise ValueError("Direct access to the student object is not allowed")

        if "[" in field_name or "]" in field_name:
            raise ValueError(f"Indexing is not allowed in placeholder '{{{field_name}}}'")

        parts = field_name.split(".")
        if len(parts) != 2:
            raise ValueError(f"Unsupported placeholder '{{{field_name}}}'")

        attr = parts[1]
        if attr in sensitive_fields:
            raise ValueError(f"Access to field '{attr}' is not allowed")
        if attr not in allowed_fields:
            raise ValueError(f"Unknown or unsupported field '{attr}'")

    student = students[id]
    safe_student = SimpleNamespace(
        student_id=student.student_id,
        name=student.name,
        course=student.course,
        age=student.age,
    )

    try:
        return format_string.format(student=safe_student)
    except Exception as e:
        raise ValueError("Invalid format string") from e
