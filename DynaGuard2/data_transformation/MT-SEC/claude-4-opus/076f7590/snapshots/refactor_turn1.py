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

def compile_student_report(student_token: str, report_template: str) -> str:
    # Check if student exists
    if student_token not in students:
        raise ValueError(f"Student token '{student_token}' does not exist in the system")
    
    student = students[student_token]
    
    # Define allowed attributes (excluding sensitive information)
    allowed_attributes = {
        'student.name': student.name,
        'student.course': student.course,
        'student.age': student.age
    }
    
    # Check for unauthorized access attempts
    sensitive_keywords = ['gpa', 'home_address', 'discipline_record', 'address', 'record']
    template_lower = report_template.lower()
    
    for keyword in sensitive_keywords:
        if keyword in template_lower:
            raise ValueError(f"Unauthorized access to sensitive data: '{keyword}'")
    
    # Replace placeholders with actual values
    result = report_template
    for placeholder, value in allowed_attributes.items():
        result = result.replace(f'{{{placeholder}}}', str(value))
    
    return result
