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

def validate_template_for_sensitive_data(report_template: str) -> None:
    """Check if the template contains any sensitive attributes."""
    sensitive_attrs = ['gpa', 'home_address', 'discipline_record']
    for attr in sensitive_attrs:
        if f'{{student.{attr}}}' in report_template or f'{{student.{attr.upper()}}}' in report_template or f'{{student.{attr.capitalize()}}}' in report_template:
            raise ValueError(f"Cannot access sensitive attribute: {attr}")

def create_safe_student_object(student: Student) -> object:
    """Create a safe student object with only allowed attributes."""
    return type('SafeStudent', (), {
        'name': student.name,
        'course': student.course,
        'age': student.age,
        'student_id': student.student_id
    })()

def format_report(safe_student: object, report_template: str) -> str:
    """Format the report template with the safe student object."""
    try:
        return report_template.format(student=safe_student)
    except AttributeError as e:
        attr_name = str(e).split("'")[-2] if "'" in str(e) else "unknown"
        raise ValueError(f"Cannot access sensitive attribute: {attr_name}")

def build_student_profile(unique_id: str, report_template: str) -> str:
    if unique_id not in students:
        raise ValueError(f"Student ID {unique_id} does not exist")
    
    student = students[unique_id]
    
    # Pipeline pattern: validate -> create safe object -> format
    validate_template_for_sensitive_data(report_template)
    safe_student = create_safe_student_object(student)
    return format_report(safe_student, report_template)
