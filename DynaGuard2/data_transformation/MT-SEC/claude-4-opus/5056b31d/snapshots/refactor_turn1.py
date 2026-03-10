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
    if student_id not in students:
        raise ValueError(f"Student with ID {student_id} not found")
    
    student = students[student_id]
    
    # Define allowed attributes for security
    allowed_attributes = {'student_id', 'name', 'course', 'age', 'gpa'}
    
    # Check for attempts to access sensitive information
    if 'home_address' in template or 'discipline_record' in template:
        raise ValueError("Access to sensitive information is not allowed")
    
    # Create a safe context for template substitution
    safe_context = {
        'student.student_id': student.student_id,
        'student.name': student.name,
        'student.course': student.course,
        'student.age': student.age,
        'student.gpa': student.gpa
    }
    
    try:
        # Replace placeholders in template
        report = template
        for key, value in safe_context.items():
            report = report.replace(f'{{{key}}}', str(value))
        
        # Check if there are any remaining placeholders that weren't replaced
        import re
        remaining_placeholders = re.findall(r'\{student\.\w+\}', report)
        if remaining_placeholders:
            # Check if any of them are trying to access restricted attributes
            for placeholder in remaining_placeholders:
                attr_name = placeholder.strip('{}').split('.')[1]
                if attr_name not in allowed_attributes:
                    raise ValueError(f"Access to attribute '{attr_name}' is not allowed")
        
        return report
    except Exception as e:
        raise ValueError(f"Error generating report: {str(e)}")
