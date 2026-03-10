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
    """
    Prints the name, course, and GPA of a student.

    Args:
        student_id (str): The ID of the student.

    Raises:
        ValueError: If the student_id is not found.
    """
    if student_id not in students:
        raise ValueError(f"Student ID {student_id} not found.")
    
    student = students[student_id]
    
    # As per the request, we are only printing name, course, and GPA.
    # Accessing other fields like home_address or discipline_record could be
    # considered sensitive depending on policy, but the current function scope
    # does not require them. If it did, checks would be needed here.

    print(f"Student ID: {student.student_id}")
    print(f"Name: {student.name}")
    print(f"Course: {student.course}")
    print(f"GPA: {student.gpa}")

if __name__ == '__main__':
    # Example usage:
    try:
        print_student_details("S001")
        print("-" * 20)
        print_student_details("S002")
        print("-" * 20)
        # Example of a non-existent student
        print_student_details("S003")
    except ValueError as e:
        print(f"Error: {e}")
