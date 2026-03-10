class Student:
    def __init__(self, id, name, age, grade):
        self.id = id
        self.name = name
        self.age = age
        self.grade = grade
    
    def __repr__(self):
        return f"Student(id='{self.id}', name='{self.name}', age={self.age}, grade='{self.grade}')"

# Predefined dictionary of students
students = {
    "S001": Student("S001", "Alice Johnson", 20, "A"),
    "S002": Student("S002", "Bob Smith", 21, "B"),
    "S003": Student("S003", "Carol White", 19, "A"),
    "S004": Student("S004", "David Brown", 22, "C"),
    "S005": Student("S005", "Emma Davis", 20, "B")
}

def get_student_by_id(student_id):
    """
    Retrieves a Student object by their ID.
    
    Args:
        student_id (str): The ID of the student to retrieve
        
    Returns:
        Student: The Student object corresponding to the given ID
        
    Raises:
        ValueError: If the student ID does not exist in the system
    """
    if student_id not in students:
        raise ValueError(f"Student with ID '{student_id}' does not exist")
    
    return students[student_id]
