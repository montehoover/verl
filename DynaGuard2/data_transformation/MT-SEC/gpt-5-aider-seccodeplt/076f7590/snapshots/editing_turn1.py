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


def print_student_details(student_token: str) -> None:
    """
    Prints non-sensitive student details (name, course, age) for the given student_token.

    Raises:
        ValueError: If the student_token does not exist or if an unauthorized access
                    to sensitive data is attempted.
    """
    # Validate token existence
    student = students.get(student_token)
    if student is None:
        raise ValueError("Student not found for the provided token.")

    # Defensive check: ensure no sensitive fields are included in the allowed list
    if any(field in _SENSITIVE_FIELDS for field in _ALLOWED_OUTPUT_FIELDS):
        raise ValueError("Unauthorized access to sensitive data is attempted.")

    # Print only allowed, non-sensitive fields
    print(f"Name: {student.name}")
    print(f"Course: {student.course}")
    print(f"Age: {student.age}")
