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

def build_student_profile(unique_id: str, report_template: str) -> str:
    """
    Generate a student report using the provided template.
    
    Args:
        unique_id: A unique string representing the student ID
        report_template: A format string which contains placeholders for generating the report
        
    Returns:
        A formatted string representing the student's report
        
    Raises:
        ValueError: If trying to access sensitive attributes or if student ID doesn't exist
    """
    # Check if student exists
    if unique_id not in students:
        raise ValueError(f"Student with ID {unique_id} does not exist")
    
    # Get the student object
    student = students[unique_id]
    
    # Define sensitive attributes that should not be accessible
    sensitive_attributes = {'gpa', 'home_address', 'discipline_record'}
    
    # Check if template contains any sensitive attributes
    for attr in sensitive_attributes:
        if f'{{student.{attr}}}' in report_template or f'{{student.{attr}' in report_template:
            raise ValueError(f"Cannot access sensitive attribute: {attr}")
    
    # Create a safe namespace for formatting
    safe_student = {
        'student': {
            'student_id': student.student_id,
            'name': student.name,
            'course': student.course,
            'age': student.age
        }
    }
    
    # Format and return the report
    try:
        return report_template.format(**safe_student)
    except KeyError as e:
        # If trying to access an attribute that doesn't exist in safe_student
        raise ValueError(f"Invalid attribute access: {e}")
