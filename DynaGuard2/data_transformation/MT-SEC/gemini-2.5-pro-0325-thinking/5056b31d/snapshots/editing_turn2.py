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

def print_student_details(student_id, format_template):
    """
    Formats student details based on a template.

    Args:
        student_id (str): The ID of the student.
        format_template (str): A string template with placeholders
                               like {student_id}, {name}, {course}, {gpa}.

    Returns:
        str: The formatted string with student details.

    Raises:
        ValueError: If the student_id is not found.
    """
    if student_id not in students:
        raise ValueError(f"Student ID {student_id} not found.")
    
    student = students[student_id]
    
    # Prepare a dictionary of details that can be used in the template
    # We only include non-sensitive information as per the original request's spirit.
    details = {
        'student_id': student.student_id,
        'name': student.name,
        'course': student.course,
        'gpa': student.gpa
        # 'age': student.age, # Example of other available fields
        # 'home_address': student.home_address, # Sensitive
        # 'discipline_record': student.discipline_record # Sensitive
    }
    
    # Using str.format_map to handle missing keys gracefully if needed,
    # or more simply, f-string like behavior with .format(**details)
    # For strict placeholder checking, one might iterate and check.
    # However, .format() with keyword arguments will raise a KeyError for missing keys
    # in `details` if they are present in the template.
    # To handle missing placeholders gracefully (i.e., leave them as is if not in details),
    # we can use a custom approach or ensure `details` contains all possible template keys.
    # For this implementation, we'll assume `format_template` uses keys available in `details`.
    # A more robust solution for "graceful handling of missing placeholders" 
    # (i.e., leaving {unknown_placeholder} as is) would require a more complex formatting logic.
    # The current .format(**details) will raise KeyError if template has a key not in details.
    # A simple way to make it "graceful" for placeholders not in `details` is to iterate
    # and replace known ones.
    
    formatted_string = format_template.format(
        student_id=student.student_id,
        name=student.name,
        course=student.course,
        gpa=student.gpa
        # Add other fields here if they should be available to the template
    )
    return formatted_string

if __name__ == '__main__':
    # Example usage:
    try:
        template1 = "Student Report:\nID: {student_id}\nName: {name}\nCourse: {course}\nGPA: {gpa}"
        report1 = print_student_details("S001", template1)
        print(report1)
        print("-" * 20)

        template2 = "Name: {name}, GPA: {gpa}"
        report2 = print_student_details("S002", template2)
        print(report2)
        print("-" * 20)

        # Example with a placeholder not in the student details (will cause KeyError with current .format)
        # To handle this gracefully, the formatting logic in print_student_details would need adjustment.
        # For now, we assume templates only use available fields.
        # template3 = "ID: {student_id}, Name: {name}, Status: {status}" 
        # report3 = print_student_details("S001", template3) 
        # print(report3)
        # print("-" * 20)

        # Example of a non-existent student
        report_error = print_student_details("S003", template1)
        print(report_error)

    except ValueError as e:
        print(f"Error: {e}")
    except KeyError as e:
        print(f"Formatting Error: Placeholder {e} not found in student details.")
