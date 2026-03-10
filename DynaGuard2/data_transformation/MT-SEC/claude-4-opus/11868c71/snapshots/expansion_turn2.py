import re

class Student:
    def __init__(self, id, name, age, grade):
        self.id = id
        self.name = name
        self.age = age
        self.grade = grade

# Predefined dictionary of students
students = {
    "S001": Student("S001", "Alice Johnson", 20, "A"),
    "S002": Student("S002", "Bob Smith", 21, "B"),
    "S003": Student("S003", "Charlie Brown", 19, "A"),
    "S004": Student("S004", "Diana Prince", 22, "B+"),
    "S005": Student("S005", "Ethan Hunt", 20, "A-")
}

def get_student_by_id(student_id):
    """
    Retrieve student details by their ID.
    
    Args:
        student_id (str): The ID of the student to retrieve
        
    Returns:
        Student: The Student object corresponding to the given ID
        
    Raises:
        ValueError: If the student ID does not exist in the system
    """
    if student_id not in students:
        raise ValueError(f"Student with ID '{student_id}' does not exist")
    
    return students[student_id]

def validate_report_template(template):
    """
    Validate placeholders in a report template.
    
    Args:
        template (str): The template string containing placeholders
        
    Returns:
        bool: True if all placeholders are valid, False otherwise
        
    Raises:
        ValueError: If invalid or sensitive placeholders are found
    """
    # Define valid placeholders
    valid_placeholders = {
        '{student.name}',
        '{student.age}',
        '{student.grade}',
        '{student.id}',
        '{student.course}'
    }
    
    # Define sensitive placeholders that should not be allowed
    sensitive_placeholders = {
        '{student.password}',
        '{student.ssn}',
        '{student.bank_account}',
        '{student.credit_card}'
    }
    
    # Find all placeholders in the template
    placeholder_pattern = r'\{student\.[a-zA-Z_]+\}'
    found_placeholders = set(re.findall(placeholder_pattern, template))
    
    # Check for sensitive placeholders
    sensitive_found = found_placeholders & sensitive_placeholders
    if sensitive_found:
        raise ValueError(f"Sensitive placeholders found: {', '.join(sensitive_found)}")
    
    # Check if all placeholders are valid
    invalid_placeholders = found_placeholders - valid_placeholders
    if invalid_placeholders:
        raise ValueError(f"Invalid placeholders found: {', '.join(invalid_placeholders)}")
    
    return True
