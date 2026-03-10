class Student:
    def __init__(self, student_id, name, course, age, gpa, home_address, discipline_record):
        self.student_id = student_id
        self.name = name
        self.course = course
        self.age = age
        self.gpa = gpa  # Sensitive
        self.home_address = home_address  # Sensitive
        self.discipline_record = discipline_record # Sensitive

students = {
    "S001": Student("S001", "Alice Brown", "Computer Science", 20, 3.8, "456 Elm St, Springfield", "None"),
    "S002": Student("S002", "Bob Green", "Mechanical Engineering", 22, 3.5, "789 Oak St, Shelbyville", "Minor infraction in 2022")
}

SENSITIVE_ATTRIBUTES = {"gpa", "home_address", "discipline_record"}
ALLOWED_ATTRIBUTES = {"student_id", "name", "course", "age"}

class SafeStudentView:
    def __init__(self, student_obj: Student):
        self._student_obj = student_obj

    def __getattr__(self, name: str):
        if name in SENSITIVE_ATTRIBUTES:
            raise ValueError(f"Access to sensitive attribute '{name}' is not allowed.")
        if name in ALLOWED_ATTRIBUTES:
            return getattr(self._student_obj, name)
        
        # Attribute is not sensitive and not explicitly allowed for the summary.
        # Standard behavior for str.format() is to raise AttributeError if a field is missing.
        raise AttributeError(f"Attribute '{name}' is not allowed for student summary.")

def create_student_summary(id: str, format_string: str) -> str:
    """
    Generates a student report using a customizable format string.

    Args:
        id: The unique student ID.
        format_string: A format string with placeholders (e.g., '{student.name}').

    Returns:
        A formatted string representing the student's report.

    Raises:
        ValueError: If the student ID does not exist, or if there's an attempt
                    to access sensitive attributes (gpa, home_address, discipline_record)
                    via the format_string.
    """
    if id not in students:
        raise ValueError(f"Student ID '{id}' does not exist.")
    
    student = students[id]
    safe_view = SafeStudentView(student)
    
    try:
        # The format_string is expected to use `student` as the object name,
        # e.g., "{student.name}".
        summary = format_string.format(student=safe_view)
    except ValueError:
        # Catches ValueErrors from SafeStudentView for sensitive attributes.
        raise
    except AttributeError:
        # Catches AttributeErrors from SafeStudentView for non-allowed/non-sensitive attributes.
        # Or if an allowed attribute doesn't actually exist on the Student object.
        # This is standard behavior for .format() with missing attributes.
        raise
        
    return summary
