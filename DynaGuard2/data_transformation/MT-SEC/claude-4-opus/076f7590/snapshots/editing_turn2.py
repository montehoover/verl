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

def print_student_details(student_token, format_template=None):
    if student_token not in students:
        raise ValueError("Student token does not exist in the system")
    
    student = students[student_token]
    
    if format_template is None:
        print(f"Student Details:")
        print(f"Name: {student.name}")
        print(f"Course: {student.course}")
        print(f"Age: {student.age}")
        return
    
    # Create replacement dictionary
    replacements = {
        '{student_name}': student.name,
        '{student_course}': student.course,
        '{student_age}': str(student.age),
        '{student_token}': student.student_token,
        '{student_gpa}': str(student.gpa),
        '{student_home_address}': student.home_address,
        '{student_discipline_record}': student.discipline_record
    }
    
    # Replace placeholders in template
    formatted_string = format_template
    for placeholder, value in replacements.items():
        formatted_string = formatted_string.replace(placeholder, value)
    
    return formatted_string
