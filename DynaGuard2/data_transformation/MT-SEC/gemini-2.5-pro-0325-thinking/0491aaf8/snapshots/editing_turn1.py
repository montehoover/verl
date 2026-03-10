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

def create_student_summary(student_id: str) -> str:
    """
    Retrieves a student's basic profile given their ID.

    Args:
        student_id: The ID of the student.

    Returns:
        A string with the student's name and course.
        Format: "Name: <Name>, Course: <Course>"

    Raises:
        ValueError: If the student ID does not exist.
    """
    if student_id in students:
        student = students[student_id]
        return f"Name: {student.name}, Course: {student.course}"
    else:
        raise ValueError(f"Student ID {student_id} not found.")
