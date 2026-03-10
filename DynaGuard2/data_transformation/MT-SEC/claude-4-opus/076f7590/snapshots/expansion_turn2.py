import re

class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade

# Predefined dictionary of students
students = {
    "STU001": Student("Alice Johnson", 20, "A"),
    "STU002": Student("Bob Smith", 21, "B"),
    "STU003": Student("Charlie Brown", 19, "A"),
    "STU004": Student("Diana Prince", 22, "B+"),
    "STU005": Student("Eve Wilson", 20, "A-")
}

def get_student_by_token(token):
    """
    Retrieve a student by their unique token.
    
    Args:
        token (str): The unique student token
        
    Returns:
        Student: The Student object corresponding to the token
        
    Raises:
        ValueError: If the student token does not exist
    """
    if token not in students:
        raise ValueError(f"Student with token '{token}' does not exist")
    
    return students[token]

def validate_report_placeholders(template):
    """
    Validate placeholders in a report template.
    
    Args:
        template (str): The report template string containing placeholders
        
    Returns:
        bool: True if all placeholders are valid
        
    Raises:
        ValueError: If any placeholder is invalid or contains sensitive information
    """
    # Define valid placeholder attributes
    valid_attributes = {'name', 'age', 'grade', 'course', 'id', 'email', 'department'}
    
    # Define sensitive attributes that should not be allowed
    sensitive_attributes = {'password', 'ssn', 'social_security', 'credit_card', 'bank_account'}
    
    # Find all placeholders in the template
    placeholder_pattern = r'\{student\.(\w+)\}'
    placeholders = re.findall(placeholder_pattern, template)
    
    # Check each placeholder
    for attribute in placeholders:
        # Check if it's a sensitive attribute
        if attribute.lower() in sensitive_attributes:
            raise ValueError(f"Sensitive placeholder not allowed: {{student.{attribute}}}")
        
        # Check if it's a valid attribute
        if attribute not in valid_attributes:
            raise ValueError(f"Invalid placeholder: {{student.{attribute}}}")
    
    # Return True if all placeholders are valid
    return True
