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

def compile_student_report(student_token, report_template):
    if student_token not in students:
        raise ValueError("Student token does not exist in the system")
    
    student = students[student_token]
    
    # Replace placeholders with actual student data
    report = report_template
    report = report.replace('{student.name}', student.name)
    report = report.replace('{student.course}', student.course)
    report = report.replace('{student.age}', str(student.age))
    report = report.replace('{student.student_token}', student.student_token)
    report = report.replace('{student.gpa}', str(student.gpa))
    report = report.replace('{student.home_address}', student.home_address)
    report = report.replace('{student.discipline_record}', student.discipline_record)
    
    return report
