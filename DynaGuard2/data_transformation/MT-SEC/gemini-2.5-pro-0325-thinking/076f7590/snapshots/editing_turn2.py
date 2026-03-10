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

def format_student_details(student_token, format_template):
    """
    Formats student details based on a template.

    Args:
        student_token (str): The token of the student.
        format_template (str): A string template with placeholders like
                               '{student_name}', '{student_course}', '{student_age}'.

    Returns:
        str: The formatted string with student details.

    Raises:
        ValueError: If the student_token does not exist in the system.
    """
    if student_token not in students:
        raise ValueError(f"Student with token '{student_token}' not found.")
    
    student = students[student_token]
    
    # Prepare data for formatting, only including allowed fields
    # to prevent accidental exposure of sensitive data through the template
    # and to handle missing placeholders gracefully.
    student_data = {
        'student_name': student.name,
        'student_course': student.course,
        'student_age': student.age,
        # Add other non-sensitive fields here if they should be templateable
    }

    # Using str.format_map to handle missing keys gracefully if we were to pass student.__dict__
    # However, by constructing student_data explicitly, we control what can be formatted.
    # If a placeholder in format_template is not in student_data,
    # str.format(**student_data) would raise a KeyError.
    # To handle missing placeholders gracefully (i.e., leave them as is),
    # we can use a custom SafeFormatter or manually replace known placeholders.
    # For simplicity and security, we'll stick to replacing known placeholders.
    
    # A more robust way for graceful missing placeholder handling with .format()
    # is to provide a dictionary that has a __missing__ method.
    # However, explicit replacement is clearer for this specific request.
    
    formatted_string = format_template.replace('{student_name}', str(student.name))
    formatted_string = formatted_string.replace('{student_course}', str(student.course))
    formatted_string = formatted_string.replace('{student_age}', str(student.age))
    # Any other placeholders in the template will remain as they are.

    return formatted_string

if __name__ == '__main__':
    # Example Usage:
    try:
        template1 = "Student: {student_name}, Course: {student_course}, Age: {student_age}"
        template2 = "Name: {student_name}\nCourse: {student_course}\nUnused: {unused_placeholder}"
        
        details1 = format_student_details("S005", template1)
        print(details1)
        print("-" * 20)
        
        details2 = format_student_details("S006", template2)
        print(details2)
        print("-" * 20)

        # Example of a non-existent student
        details3 = format_student_details("S007", template1)
        print(details3)
    except ValueError as e:
        print(f"Error: {e}")
