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

def generate_student_report(student_id, template):
    """
    Generates a student report string based on a template.

    Args:
        student_id (str): The ID of the student.
        template (str): A string template with placeholders like
                        '{student.name}', '{student.course}', etc.

    Returns:
        str: The generated student report string.

    Raises:
        ValueError: If the student_id is not found.
        AttributeError: If the template references an attribute not present
                        on the Student object.
        KeyError: If the template has other formatting issues (e.g. unmatched placeholders).
    """
    if student_id not in students:
        raise ValueError(f"Student ID {student_id} not found.")
    
    student = students[student_id]
    
    # The template can access any attribute of the student object.
    # e.g., template = "Name: {student.name}, GPA: {student.gpa}"
    # This will also allow access to sensitive fields if they are part of the template
    # e.g., template = "Address: {student.home_address}"
    # The responsibility for crafting a safe template lies with the caller.
    
    return template.format(student=student)

if __name__ == '__main__':
    # Example usage:
    try:
        template1 = "Student Report:\nID: {student.student_id}\nName: {student.name}\nCourse: {student.course}\nGPA: {student.gpa}"
        report1 = generate_student_report("S001", template1)
        print(report1)
        print("-" * 20)

        template2 = "Name: {student.name}, GPA: {student.gpa}, Age: {student.age}"
        report2 = generate_student_report("S002", template2)
        print(report2)
        print("-" * 20)

        # Example with a template accessing a sensitive attribute
        template_sensitive = "Sensitive Info Test:\nName: {student.name}\nHome Address: {student.home_address}"
        report_sensitive = generate_student_report("S001", template_sensitive)
        print(report_sensitive)
        print("-" * 20)

        # Example with a template accessing a non-existent attribute
        template_attr_error = "Name: {student.name}, Status: {student.status}"
        try:
            report_attr_error = generate_student_report("S001", template_attr_error)
            print(report_attr_error)
        except AttributeError as e:
            print(f"Attribute Error in template: {e}")
        print("-" * 20)
        
        # Example of a non-existent student
        report_error = generate_student_report("S003", template1)
        print(report_error) # This line won't be reached if S003 raises ValueError

    except ValueError as e:
        print(f"Value Error: {e}")
    except KeyError as e:
        print(f"Key Error in template: Malformed placeholder or missing key {e}")
    except AttributeError as e: # This might catch AttributeErrors from the main block if not handled by inner try-except
        print(f"Attribute Error: {e}")
