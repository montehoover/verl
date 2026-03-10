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
    Formats student details (name, course, age) given their ID, using a template.

    Args:
        student_id (str): The ID of the student.
        format_template (str): A string template with placeholders like
                               '{student_id}', '{student_name}', '{student_course}',
                               '{student_age}'.

    Returns:
        str: The formatted string with student details.

    Raises:
        ValueError: If the student_id does not exist.
    """
    if student_id not in students:
        raise ValueError(f"Student ID '{student_id}' not found.")
    
    student = students[student_id]
    
    # Prepare data for formatting, ensuring only non-sensitive attributes are used.
    # Placeholders in the template not matching these keys will be ignored by str.format()
    # if they are not present in the data dictionary.
    data = {
        'student_id': student.student_id,
        'student_name': student.name,
        'student_course': student.course,
        'student_age': student.age
        # Sensitive attributes like gpa or home_address are intentionally omitted.
    }
    
    # Using **data allows for graceful handling of missing placeholders in the template.
    # If a placeholder in format_template (e.g., {unknown_field}) is not in data,
    # it will remain in the string as is, rather than causing a KeyError.
    return format_template.format(**data)

if __name__ == '__main__':
    # Example usage:
    template1 = "Student: {student_name} (ID: {student_id}), Age: {student_age}, Course: {student_course}"
    template2 = "Name: {student_name}\nID: {student_id}\nDetails: {student_course}, {student_age} years old. Extra info: {extra_info}"

    print("Details for S001 (Template 1):")
    try:
        details_s001 = print_student_details("S001", template1)
        print(details_s001)
    except ValueError as e:
        print(f"Error: {e}")

    print("\nDetails for S002 (Template 1):")
    try:
        details_s002 = print_student_details("S002", template1)
        print(details_s002)
    except ValueError as e:
        print(f"Error: {e}")

    print("\nDetails for S001 (Template 2 - with missing placeholder):")
    try:
        details_s001_t2 = print_student_details("S001", template2)
        print(details_s001_t2)
    except ValueError as e:
        print(f"Error: {e}")

    print("\nDetails for S003 (non-existent):")
    try:
        details_s003 = print_student_details("S003", template1)
        print(details_s003)
    except ValueError as e:
        print(f"Error: {e}")
