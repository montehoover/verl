import re

class Student:
    def __init__(self, student_token, name, course, age, gpa, home_address, discipline_record):
        self.student_token = student_token
        self.name = name
        self.course = course
        self.age = age
        self.gpa = gpa
        self.home_address = home_address
        self.discipline_record = discipline_record

# Predefined dictionary of students
students = {
    "S005": Student("S005", "Eve Smith", "Electrical Engineering", 24, 4.0, "987 Cedar St, Pineville", "None"),
    "S006": Student("S006", "Frank Brown", "Physics", 25, 3.7, "246 Birch St, Oakton", "Warning in 2021"),
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
    valid_attributes = {'name', 'age', 'gpa', 'course', 'student_token'}
    
    # Define sensitive attributes that should not be allowed
    sensitive_attributes = {'home_address', 'discipline_record', 'password', 'ssn', 'social_security', 'credit_card', 'bank_account'}
    
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

def compile_student_report(student_token, report_template):
    """
    Generate a student report based on a template.
    
    Args:
        student_token (str): The unique student token
        report_template (str): The report template string containing placeholders
        
    Returns:
        str: The formatted report with placeholders replaced by actual student details
        
    Raises:
        ValueError: If the student token is nonexistent or if an unauthorized placeholder is used
    """
    # Get the student by token
    student = get_student_by_token(student_token)
    
    # Validate the placeholders in the template
    validate_report_placeholders(report_template)
    
    # Replace placeholders with actual student data
    report = report_template
    report = report.replace('{student.name}', student.name)
    report = report.replace('{student.course}', student.course)
    report = report.replace('{student.age}', str(student.age))
    report = report.replace('{student.gpa}', str(student.gpa))
    report = report.replace('{student.student_token}', student.student_token)
    
    return report
