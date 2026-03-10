import re

class Student:
    def __init__(self, student_id, name, course, age, gpa, home_address, discipline_record):
        self.student_id = student_id
        self.name = name
        self.course = course
        self.age = age
        self.gpa = gpa
        self.home_address = home_address
        self.discipline_record = discipline_record
    
    def __repr__(self):
        return f"Student(id='{self.student_id}', name='{self.name}', course='{self.course}', age={self.age})"

# Predefined dictionary of students
students = {
    "S001": Student("S001", "Alice Brown", "Computer Science", 20, 3.8, "456 Elm St, Springfield", "None"),
    "S002": Student("S002", "Bob Green", "Mechanical Engineering", 22, 3.5, "789 Oak St, Shelbyville", "Minor infraction in 2022")
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
        'credit_card', 'password', 'pin', 'secret', 'private', 'discipline_record'
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

def create_student_summary(id, format_string):
    """
    Generates a student report by substituting placeholders with actual student details.
    
    Args:
        id (str): The student ID
        format_string (str): The format string containing placeholders
        
    Returns:
        str: The customized report string with placeholders replaced
        
    Raises:
        ValueError: If student ID is invalid or restricted data placeholders are included
    """
    # Check if the student ID exists
    student = get_student_by_id(id)
    
    # Check if placeholders are valid (no sensitive information)
    check_placeholders(format_string)
    
    # Replace placeholders with actual student data
    result = format_string
    
    # Define allowed attributes
    allowed_attrs = {
        'student_id': student.student_id,
        'name': student.name,
        'course': student.course,
        'age': student.age
    }
    
    # Replace placeholders in the format {student.attribute}
    for attr, value in allowed_attrs.items():
        result = result.replace(f'{{student.{attr}}}', str(value))
        result = result.replace(f'{{{attr}}}', str(value))
    
    # Handle any remaining placeholders that might be in format {student.something}
    remaining_placeholders = re.findall(r'\{student\.(\w+)\}', result)
    for placeholder in remaining_placeholders:
        if placeholder not in allowed_attrs:
            # This will be caught by check_placeholders if it's sensitive
            # Otherwise replace with empty string for unknown attributes
            result = result.replace(f'{{student.{placeholder}}}', '')
    
    return result
