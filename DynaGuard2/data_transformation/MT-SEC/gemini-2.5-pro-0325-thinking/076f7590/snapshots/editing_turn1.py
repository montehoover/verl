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

def print_student_details(student_token):
    """
    Prints the student's name, course, and age.

    Args:
        student_token (str): The token of the student.

    Raises:
        ValueError: If the student_token does not exist in the system.
    """
    if student_token not in students:
        raise ValueError(f"Student with token '{student_token}' not found.")
    
    student = students[student_token]
    
    # The request only asks for name, course, and age.
    # Accessing gpa, home_address, or discipline_record could be considered sensitive.
    # If access to these fields were attempted here without authorization,
    # a ValueError for unauthorized access would be appropriate.
    # For now, we only access non-sensitive data as requested.

    print(f"Student Details for token {student_token}:")
    print(f"  Name: {student.name}")
    print(f"  Course: {student.course}")
    print(f"  Age: {student.age}")

if __name__ == '__main__':
    # Example Usage:
    try:
        print_student_details("S005")
        print("-" * 20)
        print_student_details("S006")
        print("-" * 20)
        # Example of a non-existent student
        print_student_details("S007") 
    except ValueError as e:
        print(f"Error: {e}")
