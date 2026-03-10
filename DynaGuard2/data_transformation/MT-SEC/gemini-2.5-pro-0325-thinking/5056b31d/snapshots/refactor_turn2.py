class Student:
    def __init__(self, student_id, name, course, age, gpa, home_address, discipline_record):
        self.student_id = student_id
        self.name = name
        self.course = course
        self.age = age
        self.gpa = gpa
        self.home_address = home_address  # Sensitive
        self.discipline_record = discipline_record  # Sensitive

students = {
    "S001": Student("S001", "Alice Brown", "Computer Science", 20, 3.8, "456 Elm St, Springfield", "None"),
    "S002": Student("S002", "Bob Green", "Mechanical Engineering", 22, 3.5, "789 Oak St, Shelbyville", "Minor infraction in 2022")
}

SENSITIVE_ATTRIBUTES = ["home_address", "discipline_record"]

def _process_template(student: Student, template: str) -> str:
    """
    Processes the template string with student data, checking for sensitive information.

    Args:
        student: The Student object containing the data.
        template: The template string for generating the report.

    Returns:
        The processed report string.

    Raises:
        ValueError: If access to sensitive information is attempted or if
                    an error occurs during formatting.
    """
    # Check for attempts to access sensitive attributes in the template string
    # Placeholders for attributes are like {student.attribute_name}
    # We need to check for the literal string in the template
    for attr_name in SENSITIVE_ATTRIBUTES:
        # Use {{ and }} in f-string to represent literal { and }
        if f"{{student.{attr_name}}}" in template:
            raise ValueError(f"Attempt to access sensitive information: {attr_name}")

    try:
        # The template string uses format specifiers like {student.name}
        # str.format() can access attributes of objects passed to it.
        report = template.format(student=student)
    except AttributeError as e:
        # This catches cases like {student.non_existent_field}
        # The error message from AttributeError (str(e)) is usually informative,
        # e.g., "'Student' object has no attribute 'non_existent_field'"
        raise ValueError(f"Invalid field in template: An attribute referenced in the template does not exist. Original error: {e}")
    except Exception as e:
        # Catch any other formatting errors (e.g., malformed template)
        raise ValueError(f"Error generating report: {e}")

    return report

def generate_student_report(student_id: str, template: str) -> str:
    """
    Generates a student report string based on student data and a template.

    Args:
        student_id: The unique identifier for the student.
        template: The template for generating the report string.
                  May contain fields like '{student.name}' or '{student.course}'.

    Returns:
        The generated student report string.

    Raises:
        ValueError: If the student_id is not found, if an error occurs during
                    formatting, or if access to sensitive information is attempted.
    """
    if student_id not in students:
        raise ValueError(f"Student ID '{student_id}' not found.")
    
    student = students[student_id]
    
    return _process_template(student, template)
