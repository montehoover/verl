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

students = {
    "S001": Student("S001", "Alice Brown", "Computer Science", 20, 3.8, "456 Elm St, Springfield", "None"),
    "S002": Student("S002", "Bob Green", "Mechanical Engineering", 22, 3.5, "789 Oak St, Shelbyville", "Minor infraction in 2022")
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
        '{student.gpa}',
        '{student.student_id}',
        '{student.course}'
    }
    
    # Define sensitive placeholders that should not be allowed
    sensitive_placeholders = {
        '{student.home_address}',
        '{student.discipline_record}',
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

def build_student_profile(unique_id, report_template):
    """
    Generate a student report using a customizable template.
    
    Args:
        unique_id (str): The ID of the student
        report_template (str): Template string with placeholders
        
    Returns:
        str: The formatted report with placeholders replaced
        
    Raises:
        ValueError: If student ID doesn't exist or sensitive placeholders are used
    """
    # Get the student
    student = get_student_by_id(unique_id)
    
    # Validate the template
    validate_report_template(report_template)
    
    # Replace placeholders with actual values
    formatted_report = report_template
    formatted_report = formatted_report.replace('{student.name}', student.name)
    formatted_report = formatted_report.replace('{student.course}', student.course)
    formatted_report = formatted_report.replace('{student.age}', str(student.age))
    formatted_report = formatted_report.replace('{student.gpa}', str(student.gpa))
    formatted_report = formatted_report.replace('{student.student_id}', student.student_id)
    
    return formatted_report
