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

def create_student_summary(id, format_string):
    if id not in students:
        raise ValueError(f"Student with ID {id} not found")
    
    student = students[id]
    
    # Check for sensitive fields in the format string
    sensitive_fields = ['gpa', 'home_address']
    for field in sensitive_fields:
        if f'{{student.{field}}}' in format_string.lower():
            raise ValueError(f"Access to {field} is not allowed")
    
    # Create a safe context for formatting
    safe_context = {
        'student': type('SafeStudent', (), {
            'name': student.name,
            'course': student.course,
            'age': student.age,
            'student_id': student.student_id,
            'discipline_record': student.discipline_record
        })()
    }
    
    return format_string.format(**safe_context)
