import re

class Student:
    def __init__(self, id, name, age, grade):
        self.id = id
        self.name = name
        self.age = age
        self.grade = grade
    
    def __repr__(self):
        return f"Student(id='{self.id}', name='{self.name}', age={self.age}, grade='{self.grade}')"

# Predefined dictionary of students
students = {
    "S001": Student("S001", "Alice Johnson", 20, "A"),
    "S002": Student("S002", "Bob Smith", 21, "B"),
    "S003": Student("S003", "Carol White", 19, "A"),
    "S004": Student("S004", "David Brown", 22, "C"),
    "S005": Student("S005", "Emma Davis", 20, "B")
}

def get_student_by_id(student_id):
    """
    Retrieves a Student object by their ID.
    
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

def check_placeholders(template):
    """
    Verifies that placeholders in a text template are valid and do not target sensitive information.
    
    Args:
        template (str): The text template containing placeholders
        
    Returns:
        bool: True if all placeholders are valid
        
    Raises:
        ValueError: If any placeholder targets sensitive information
    """
    # Define sensitive fields that should not be exposed
    sensitive_fields = {
        'gpa', 'home_address', 'ssn', 'social_security', 'phone', 'phone_number',
        'email', 'email_address', 'parent_name', 'parent_phone', 'parent_email',
        'emergency_contact', 'medical_info', 'medical_information', 'health_info',
        'financial_aid', 'financial_info', 'income', 'salary', 'bank_account',
        'credit_card', 'password', 'pin', 'secret', 'private'
    }
    
    # Find all placeholders in the format {something.field} or {field}
    placeholder_pattern = r'\{([^}]+)\}'
    placeholders = re.findall(placeholder_pattern, template)
    
    for placeholder in placeholders:
        # Check if placeholder contains a dot (attribute access)
        if '.' in placeholder:
            parts = placeholder.split('.')
            if len(parts) >= 2:
                field = parts[-1].lower()
                if field in sensitive_fields:
                    raise ValueError(f"Placeholder '{{{placeholder}}}' targets sensitive information")
        else:
            # Check standalone placeholder
            if placeholder.lower() in sensitive_fields:
                raise ValueError(f"Placeholder '{{{placeholder}}}' targets sensitive information")
    
    return True
