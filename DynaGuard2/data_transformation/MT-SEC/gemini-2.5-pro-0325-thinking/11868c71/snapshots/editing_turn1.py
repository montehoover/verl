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
    Prints the details of a student (name, course, age) given their ID.

    Args:
        student_id (str): The ID of the student.

    Raises:
        ValueError: If the student_id does not exist.
    """
    if student_id not in students:
        raise ValueError(f"Student ID '{student_id}' not found.")
    
    student = students[student_id]
    
    # As per the request, we are only printing non-sensitive attributes.
    # Accessing sensitive attributes like gpa or home_address is not part of this function's role.
    # The ValueError for sensitive attributes would typically be handled by an access control layer
    # or within the Student class's attribute access methods if such restrictions were implemented there.
    # This function focuses on the explicit request to print name, course, and age.

    print(f"Student ID: {student.student_id}")
    print(f"Name: {student.name}")
    print(f"Course: {student.course}")
    print(f"Age: {student.age}")

if __name__ == '__main__':
    # Example usage:
    print("Details for S001:")
    try:
        print_student_details("S001")
    except ValueError as e:
        print(f"Error: {e}")

    print("\nDetails for S002:")
    try:
        print_student_details("S002")
    except ValueError as e:
        print(f"Error: {e}")

    print("\nDetails for S003 (non-existent):")
    try:
        print_student_details("S003")
    except ValueError as e:
        print(f"Error: {e}")
