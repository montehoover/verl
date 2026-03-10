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
    # Check if student ID exists
    if id not in students:
        raise ValueError(f"Student ID '{id}' does not exist")
    
    # Get the student object
    student = students[id]
    
    # Define sensitive attributes that should not be accessible
    sensitive_attributes = {'gpa', 'home_address', 'discipline_record'}
    
    # Check if the format string contains any sensitive attributes
    for attr in sensitive_attributes:
        if f'{{student.{attr}}}' in format_string:
            raise ValueError(f"Access to sensitive attribute '{attr}' is not allowed")
    
    # Create a safe namespace for formatting
    # Only include non-sensitive attributes
    safe_student = type('SafeStudent', (), {
        'name': student.name,
        'course': student.course,
        'age': student.age,
        'student_id': student.student_id
    })()
    
    # Format and return the string
    try:
        return format_string.format(student=safe_student)
    except AttributeError as e:
        # If trying to access an attribute that doesn't exist in safe_student
        raise ValueError(f"Invalid attribute access: {str(e)}")
