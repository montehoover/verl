import re

class Student:
    def __init__(self, id, name, age, grade):
        self.id = id
        self.name = name
        self.age = age
        self.grade = grade

# Predefined dictionary of students
students = {
    1001: Student(1001, "Alice Johnson", 20, "A"),
    1002: Student(1002, "Bob Smith", 19, "B"),
    1003: Student(1003, "Charlie Brown", 21, "A"),
    1004: Student(1004, "Diana Prince", 20, "C"),
    1005: Student(1005, "Ethan Hunt", 22, "B")
}

def get_student_by_id(student_id):
    """
    Retrieve a student by their ID.
    
    Args:
        student_id: The ID of the student to retrieve
        
    Returns:
        Student object corresponding to the given ID
        
    Raises:
        ValueError: If the student ID does not exist
    """
    if student_id not in students:
        raise ValueError(f"Student with ID {student_id} does not exist")
    
    return students[student_id]

def validate_report_template(template):
    """
    Validate that a report template only contains valid placeholders.
    
    Args:
        template: The template string to validate
        
    Returns:
        bool: True if all placeholders are valid, False otherwise
        
    Raises:
        ValueError: If any invalid or sensitive placeholders are found
    """
    # Define valid placeholders
    valid_placeholders = {
        '{student.name}',
        '{student.age}',
        '{student.grade}',
        '{student.id}',
        '{student.course}'
    }
    
    # Define sensitive placeholders that should raise an error
    sensitive_placeholders = {
        '{student.password}',
        '{student.ssn}',
        '{student.credit_card}',
        '{student.bank_account}'
    }
    
    # Find all placeholders in the template
    placeholder_pattern = r'\{student\.\w+\}'
    found_placeholders = set(re.findall(placeholder_pattern, template))
    
    # Check for sensitive placeholders
    sensitive_found = found_placeholders & sensitive_placeholders
    if sensitive_found:
        raise ValueError(f"Template contains sensitive placeholders: {sensitive_found}")
    
    # Check if all placeholders are valid
    invalid_placeholders = found_placeholders - valid_placeholders
    if invalid_placeholders:
        raise ValueError(f"Template contains invalid placeholders: {invalid_placeholders}")
    
    return True
