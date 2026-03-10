class Student:
    def __init__(self, id, name, age, grade):
        self.id = id
        self.name = name
        self.age = age
        self.grade = grade

# Predefined dictionary of students
students = {
    1001: Student(1001, "Alice Johnson", 20, "A"),
    1002: Student(1002, "Bob Smith", 19, "B"),
    1003: Student(1003, "Charlie Brown", 21, "A"),
    1004: Student(1004, "Diana Prince", 20, "C"),
    1005: Student(1005, "Ethan Hunt", 22, "B")
}

def get_student_by_id(student_id):
    """
    Retrieve a student by their ID.
    
    Args:
        student_id: The ID of the student to retrieve
        
    Returns:
        Student object corresponding to the given ID
        
    Raises:
        ValueError: If the student ID does not exist
    """
    if student_id not in students:
        raise ValueError(f"Student with ID {student_id} does not exist")
    
    return students[student_id]
