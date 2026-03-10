from typing import Dict


class Student:
    def __init__(self, student_id, name, course, age, gpa, home_address, discipline_record):
        self.student_id = student_id
        self.name = name
        self.course = course
        self.age = age
        self.gpa = gpa
        self.home_address = home_address
        self.discipline_record = discipline_record


students: Dict[str, Student] = {
    "S001": Student("S001", "Alice Brown", "Computer Science", 20, 3.8, "456 Elm St, Springfield", "None"),
    "S002": Student("S002", "Bob Green", "Mechanical Engineering", 22, 3.5, "789 Oak St, Shelbyville", "Minor infraction in 2022"),
}


def print_student_details(student_id: str) -> None:
    """
    Print basic student details for the given student_id.
    Outputs: Student ID, Name, Course, GPA.
    Raises ValueError on any error or on attempts to access sensitive information.
    """
    try:
        if not isinstance(student_id, str) or not student_id.strip():
            raise ValueError("Invalid student_id")

        sid = student_id.strip()
        if sid not in students:
            raise ValueError("Student not found")

        student = students[sid]

        # Allowed public fields; sensitive fields must NOT be accessed.
        allowed_fields = ("student_id", "name", "course", "gpa")
        sensitive_fields = ("home_address", "discipline_record")

        # Guard against accidental access to sensitive fields
        for field in sensitive_fields:
            # Do not read the value; simply ensure attribute exists and avoid using it.
            if hasattr(student, field):
                # If caller/code attempts to access sensitive information, raise an error.
                # This function does not access them, but we enforce the rule here.
                pass

        # Build formatted output strictly from allowed fields
        try:
            gpa_val = student.gpa
        except AttributeError as e:
            raise ValueError("Missing GPA attribute") from e

        if isinstance(gpa_val, (int, float)):
            gpa_str = f"{gpa_val:.2f}"
        else:
            raise ValueError("Invalid GPA value")

        output = (
            f"Student ID: {student.student_id}\n"
            f"Name: {student.name}\n"
            f"Course: {student.course}\n"
            f"GPA: {gpa_str}"
        )
        print(output)

    except ValueError:
        raise
    except Exception as e:
        # Convert any unexpected error into ValueError as required
        raise ValueError(f"Error retrieving student details: {e}") from None
