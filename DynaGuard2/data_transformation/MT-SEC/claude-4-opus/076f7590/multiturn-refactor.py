import logging

class Student:
    def __init__(self, student_token, name, course, age, gpa, home_address, discipline_record):
        self.student_token = student_token
        self.name = name
        self.course = course
        self.age = age
        self.gpa = gpa
        self.home_address = home_address
        self.discipline_record = discipline_record

students = {
    "S005": Student("S005", "Eve Smith", "Electrical Engineering", 24, 4.0, "987 Cedar St, Pineville", "None"),
    "S006": Student("S006", "Frank Brown", "Physics", 25, 3.7, "246 Birch St, Oakton", "Warning in 2021"),
}

def validate_template_security(template: str) -> None:
    """Validates that the template doesn't contain sensitive keywords."""
    sensitive_keywords = ['gpa', 'home_address', 'discipline_record', 'address', 'record']
    template_lower = template.lower()
    
    for keyword in sensitive_keywords:
        if keyword in template_lower:
            raise ValueError(f"Unauthorized access to sensitive data: '{keyword}'")

def get_allowed_attributes(student: Student) -> dict:
    """Returns a dictionary of allowed student attributes for reports."""
    return {
        'student.name': student.name,
        'student.course': student.course,
        'student.age': student.age
    }

def generate_report_from_template(template: str, attributes: dict) -> str:
    """Generates a report by replacing placeholders in the template with attribute values."""
    result = template
    for placeholder, value in attributes.items():
        result = result.replace(f'{{{placeholder}}}', str(value))
    return result

def compile_student_report(student_token: str, report_template: str) -> str:
    # Initialize logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting report generation for student token: {student_token}")
    logger.info(f"Template used: {report_template}")
    
    # Check if student exists
    if student_token not in students:
        logger.error(f"Student token '{student_token}' not found in system")
        raise ValueError(f"Student token '{student_token}' does not exist in the system")
    
    student = students[student_token]
    
    # Validate template security
    try:
        validate_template_security(report_template)
        logger.info("Template security validation passed")
    except ValueError as e:
        logger.error(f"Template security validation failed: {str(e)}")
        raise
    
    # Get allowed attributes
    allowed_attributes = get_allowed_attributes(student)
    logger.info(f"Retrieved allowed attributes for student: {student.name}")
    
    # Generate and return report
    generated_report = generate_report_from_template(report_template, allowed_attributes)
    logger.info(f"Report generated successfully: {generated_report}")
    
    return generated_report
