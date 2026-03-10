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
        '{student.gpa}',
        '{student.student_id}',
        '{student.course}'
    }
    
    # Define sensitive placeholders that should raise an error
    sensitive_placeholders = {
        '{student.home_address}',
        '{student.discipline_record}',
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

def generate_student_report(student_id, template):
    """
    Generate a student report based on a template.
    
    Args:
        student_id: The ID of the student
        template: The template string with placeholders
        
    Returns:
        str: The generated report with placeholders replaced
        
    Raises:
        ValueError: If student ID doesn't exist or template contains sensitive info
    """
    # Get the student
    student = get_student_by_id(student_id)
    
    # Validate the template
    validate_report_template(template)
    
    # Replace placeholders with actual data
    report = template
    report = report.replace('{student.name}', student.name)
    report = report.replace('{student.student_id}', student.student_id)
    report = report.replace('{student.course}', student.course)
    report = report.replace('{student.age}', str(student.age))
    report = report.replace('{student.gpa}', str(student.gpa))
    
    return report
