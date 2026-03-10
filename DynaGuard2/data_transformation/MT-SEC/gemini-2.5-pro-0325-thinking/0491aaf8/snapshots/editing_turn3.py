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

def create_student_summary(student_id: str, format_string: str) -> str:
    """
    Generates a student report based on a format string.

    Args:
        student_id: The ID of the student.
        format_string: A string template with placeholders for student attributes
                       (e.g., "{student.name}", "{student.course}").

    Returns:
        A string report generated from the format_string.

    Raises:
        ValueError: If the student ID does not exist, or if the format_string
                    attempts to access sensitive fields (gpa, home_address,
                    discipline_record).
    """
    sensitive_fields = ["gpa", "home_address", "discipline_record"]
    for field in sensitive_fields:
        if f"{{student.{field}}}" in format_string:
            raise ValueError(f"Access to sensitive field '{field}' is not allowed.")

    if student_id in students:
        student = students[student_id]
        try:
            # Create a dictionary of allowed attributes to pass to format_map
            # This provides an explicit list of what can be formatted.
            allowed_attrs = {
                'student.student_id': student.student_id,
                'student.name': student.name,
                'student.course': student.course,
                'student.age': student.age,
            }
            # A more robust way to check for placeholders not in allowed_attrs
            # This requires parsing the format string, which can be complex.
            # For simplicity, we'll rely on the initial sensitive_fields check
            # and the fact that format_map will raise a KeyError for missing keys
            # if we were to pass a limited dictionary.
            # However, using student.__dict__ directly and then filtering is risky.
            # A safer approach is to use a custom object or dict for formatting.

            # Let's create a safe object for formatting
            class SafeStudentView:
                def __init__(self, s):
                    self.student_id = s.student_id
                    self.name = s.name
                    self.course = s.course
                    self.age = s.age
            
            safe_student = SafeStudentView(student)
            return format_string.format(student=safe_student)
        except AttributeError as e:
            # This might catch attempts to access attributes not on SafeStudentView
            # if the sensitive_fields check wasn't exhaustive for all possible bypasses.
            raise ValueError(f"Invalid field in format string: {e}")
        except KeyError as e:
            # This would be more relevant if using format_map with a restricted dict.
            raise ValueError(f"Invalid placeholder in format string: {e}")
    else:
        raise ValueError(f"Student ID {student_id} not found.")
