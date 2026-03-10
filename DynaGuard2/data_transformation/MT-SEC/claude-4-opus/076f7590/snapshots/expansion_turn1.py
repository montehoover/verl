class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade

# Predefined dictionary of students
students = {
    "STU001": Student("Alice Johnson", 20, "A"),
    "STU002": Student("Bob Smith", 21, "B"),
    "STU003": Student("Charlie Brown", 19, "A"),
    "STU004": Student("Diana Prince", 22, "B+"),
    "STU005": Student("Eve Wilson", 20, "A-")
}

def get_student_by_token(token):
    """
    Retrieve a student by their unique token.
    
    Args:
        token (str): The unique student token
        
    Returns:
        Student: The Student object corresponding to the token
        
    Raises:
        ValueError: If the student token does not exist
    """
    if token not in students:
        raise ValueError(f"Student with token '{token}' does not exist")
    
    return students[token]
