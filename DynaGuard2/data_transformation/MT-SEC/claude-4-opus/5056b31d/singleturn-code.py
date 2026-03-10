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

def generate_student_report(student_id: str, template: str) -> str:
    """
    Generate a student report based on a template.
    
    Args:
        student_id: The unique identifier for the student
        template: The template for generating the report string
        
    Returns:
        The generated student report string
        
    Raises:
        ValueError: If an error occurs or if access to sensitive information is attempted
    """
    # Check if student exists
    if student_id not in students:
        raise ValueError(f"Student with ID {student_id} not found")
    
    # Get the student object
    student = students[student_id]
    
    # Define sensitive fields that should not be accessible
    sensitive_fields = {'home_address', 'discipline_record'}
    
    # Check if template contains any sensitive fields
    for field in sensitive_fields:
        if f'student.{field}' in template:
            raise ValueError(f"Access to sensitive information '{field}' is not allowed")
    
    # Create a safe namespace for formatting
    # Only include non-sensitive attributes
    safe_student = type('SafeStudent', (), {
        'student_id': student.student_id,
        'name': student.name,
        'course': student.course,
        'age': student.age,
        'gpa': student.gpa
    })()
    
    try:
        # Format the template with the safe student data
        report = template.format(student=safe_student)
        return report
    except (KeyError, AttributeError) as e:
        raise ValueError(f"Error generating report: Invalid template field - {str(e)}")
    except Exception as e:
        raise ValueError(f"Error generating report: {str(e)}")
