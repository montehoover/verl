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

def print_student_details(student_id):
    try:
        if student_id not in students:
            raise ValueError(f"Student with ID {student_id} not found")
        
        student = students[student_id]
        
        print(f"Student ID: {student.student_id}")
        print(f"Name: {student.name}")
        print(f"Course: {student.course}")
        print(f"GPA: {student.gpa}")
        
    except Exception as e:
        raise ValueError(f"Error accessing student information: {str(e)}")
