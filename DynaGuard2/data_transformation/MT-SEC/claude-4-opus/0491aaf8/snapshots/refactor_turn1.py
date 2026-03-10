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

def create_student_summary(id: str, format_string: str) -> str:
    """
    Generate a student report using a customizable format string.
    
    Args:
        id: A unique string representing the student ID
        format_string: A format string which contains placeholders for generating the report
        
    Returns:
        A formatted string representing the student's report
        
    Raises:
        ValueError: If attempting to access sensitive attributes or if student ID doesn't exist
    """
    # Check if student exists
    if id not in students:
        raise ValueError(f"Student with ID '{id}' does not exist")
    
    # Get the student object
    student = students[id]
    
    # Check for sensitive attributes in the format string
    sensitive_attributes = ['gpa', 'home_address', 'discipline_record']
    for attr in sensitive_attributes:
        if f'{{student.{attr}}}' in format_string or f'{{student.{attr}' in format_string:
            raise ValueError(f"Access to sensitive attribute '{attr}' is not allowed")
    
    # Create a safe student object with only allowed attributes
    class SafeStudent:
        def __init__(self, student):
            self.name = student.name
            self.course = student.course
            self.age = student.age
            self.student_id = student.student_id
    
    safe_student = SafeStudent(student)
    
    # Format and return the string
    try:
        return format_string.format(student=safe_student)
    except AttributeError as e:
        # If trying to access an attribute that doesn't exist on SafeStudent
        if 'gpa' in str(e) or 'home_address' in str(e) or 'discipline_record' in str(e):
            raise ValueError(f"Access to sensitive attribute is not allowed")
        raise
