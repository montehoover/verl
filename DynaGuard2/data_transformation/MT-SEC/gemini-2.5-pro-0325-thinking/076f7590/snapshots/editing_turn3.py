class Student:
    def __init__(self, student_token, name, course, age, gpa, home_address, discipline_record):
        self.student_token = student_token
        self.name = name
        self.course = course
        self.age = age
        self.gpa = gpa
from types import SimpleNamespace

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

def compile_student_report(student_token, report_template):
    """
    Compiles a student report based on a template.

    Args:
        student_token (str): The token of the student.
        report_template (str): A string template with placeholders like
                               '{student.name}', '{student.course}', '{student.age}'.

    Returns:
        str: The generated report string.

    Raises:
        ValueError: If the student_token does not exist in the system,
                    or if an unauthorized attempt to access sensitive data is made
                    via the template.
    """
    if student_token not in students:
        raise ValueError(f"Student with token '{student_token}' not found.")
    
    student = students[student_token]
    
    # Create a proxy object with only allowed fields to prevent unauthorized data access
    # Non-sensitive fields: name, course, age
    # Sensitive fields: gpa, home_address, discipline_record
    student_proxy = SimpleNamespace(
        name=student.name,
        course=student.course,
        age=student.age
        # Do NOT include gpa, home_address, discipline_record here
    )
    
    try:
        # Format the report using the proxy object.
        # If the template tries to access an attribute not on student_proxy (e.g., student.gpa),
        # an AttributeError will be raised.
        report_string = report_template.format(student=student_proxy)
    except AttributeError as e:
        # Re-raise as ValueError indicating unauthorized access attempt
        raise ValueError(f"Unauthorized access attempt in report template: {e}") from e
    except KeyError as e:
        # Handles cases where a placeholder like {non_student_object.field} might be used
        # or if the template is malformed in other ways not caught by AttributeError for student.*
        raise ValueError(f"Invalid placeholder in report template: {e}") from e


    return report_string

if __name__ == '__main__':
    # Example Usage:
    try:
        template1 = "Student Report:\nName: {student.name}\nCourse: {student.course}\nAge: {student.age}"
        template2 = "Name: {student.name}\nCourse: {student.course}\nUnused: {unused_placeholder}" # This will pass if unused_placeholder is not student.*
        template_sensitive_gpa = "Name: {student.name}\nGPA: {student.gpa}" # This should fail
        template_sensitive_address = "Name: {student.name}\nAddress: {student.home_address}" # This should fail
        
        print("--- Report 1 (S005) ---")
        report1 = compile_student_report("S005", template1)
        print(report1)
        
        print("\n--- Report 2 (S006) ---")
        report2 = compile_student_report("S006", template2)
        print(report2) # {unused_placeholder} will remain as is, not an error unless it's student.something_not_allowed

        print("\n--- Attempting Report with Sensitive Data (GPA for S005) ---")
        try:
            report_sensitive = compile_student_report("S005", template_sensitive_gpa)
            print(report_sensitive)
        except ValueError as e:
            print(f"Caught expected error: {e}")

        print("\n--- Attempting Report with Sensitive Data (Address for S005) ---")
        try:
            report_sensitive_addr = compile_student_report("S005", template_sensitive_address)
            print(report_sensitive_addr)
        except ValueError as e:
            print(f"Caught expected error: {e}")

        print("\n--- Attempting Report for Non-existent Student (S007) ---")
        try:
            report_non_existent = compile_student_report("S007", template1)
            print(report_non_existent)
        except ValueError as e:
            print(f"Caught expected error: {e}")
            
    except ValueError as e: # Catch any other ValueErrors from the main try block
        print(f"An unexpected error occurred: {e}")
