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
    # Validate student existence
    if id not in students:
        raise ValueError(f"Student with ID '{id}' does not exist")
    
    # Retrieve student object
    target_student = students[id]
    
    # Define and check for sensitive attributes
    restricted_attributes = ['gpa', 'home_address', 'discipline_record']
    
    for attribute_name in restricted_attributes:
        if f'{{student.{attribute_name}}}' in format_string or f'{{student.{attribute_name}' in format_string:
            raise ValueError(f"Access to sensitive attribute '{attribute_name}' is not allowed")
    
    # Create a sanitized student object with only allowed attributes
    class SafeStudent:
        def __init__(self, original_student):
            self.name = original_student.name
            self.course = original_student.course
            self.age = original_student.age
            self.student_id = original_student.student_id
    
    sanitized_student = SafeStudent(target_student)
    
    # Format and return the report string
    try:
        return format_string.format(student=sanitized_student)
    except AttributeError as error:
        # Check if the error is due to accessing restricted attributes
        error_message = str(error)
        
        if any(attr in error_message for attr in ['gpa', 'home_address', 'discipline_record']):
            raise ValueError("Access to sensitive attribute is not allowed")
        
        raise
