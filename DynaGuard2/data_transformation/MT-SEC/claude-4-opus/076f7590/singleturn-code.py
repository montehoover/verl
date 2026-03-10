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
    """
    Create a customizable report for a student based on the provided template.
    
    Args:
        student_token: A unique string representing the student's identifier
        report_template: A format string used to create the report by embedding allowed student attributes
        
    Returns:
        The generated report as a formatted string based on the supplied template
        
    Raises:
        ValueError: If an unauthorized access to sensitive data is attempted or if the 
                   specified student_token does not exist in the system
    """
    # Check if student exists
    if student_token not in students:
        raise ValueError(f"Student token '{student_token}' does not exist in the system")
    
    student = students[student_token]
    
    # Define allowed and sensitive attributes
    allowed_attributes = {'name', 'course', 'age', 'student_token'}
    sensitive_attributes = {'gpa', 'home_address', 'discipline_record'}
    
    # Check for unauthorized access attempts in the template
    import re
    # Find all placeholders in the template
    placeholders = re.findall(r'\{student\.(\w+)\}', report_template)
    
    for attr in placeholders:
        if attr in sensitive_attributes:
            raise ValueError(f"Unauthorized access to sensitive data: {attr}")
        if attr not in allowed_attributes and attr not in sensitive_attributes:
            # If it's not a known attribute, we should still process it but it will fail naturally
            pass
    
    # Create a safe namespace for formatting
    safe_student = {
        'name': student.name,
        'course': student.course,
        'age': student.age,
        'student_token': student.student_token
    }
    
    # Format the template
    try:
        # Create a namespace object that only exposes allowed attributes
        class SafeStudent:
            def __init__(self, data):
                self._data = data
                
            def __getattr__(self, name):
                if name in self._data:
                    return self._data[name]
                raise AttributeError(f"'SafeStudent' object has no attribute '{name}'")
        
        safe_obj = SafeStudent(safe_student)
        result = report_template.format(student=safe_obj)
        return result
    except (KeyError, AttributeError) as e:
        # This handles cases where the template references non-existent attributes
        raise ValueError(f"Invalid template: {str(e)}")
