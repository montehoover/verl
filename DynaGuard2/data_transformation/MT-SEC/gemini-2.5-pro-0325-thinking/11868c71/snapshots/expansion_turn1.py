class Student:
    def __init__(self, student_id, name, age):
        self.student_id = student_id
        self.name = name
        self.age = age

    def __repr__(self):
        return f"Student(student_id='{self.student_id}', name='{self.name}', age={self.age})"

# Predefined dictionary of students
# In a real application, this data might come from a database or a file.
_students_db = {
    "S1001": Student(student_id="S1001", name="Alice Smith", age=20),
    "S1002": Student(student_id="S1002", name="Bob Johnson", age=22),
    "S1003": Student(student_id="S1003", name="Charlie Brown", age=21),
}

def get_student_by_id(student_id: str) -> Student:
    """
    Retrieves a student object by their ID.

    Args:
        student_id: The ID of the student to retrieve.

    Returns:
        The Student object corresponding to the given ID.

    Raises:
        ValueError: If the student ID does not exist in the database.
    """
    student = _students_db.get(student_id)
    if student is None:
        raise ValueError(f"Student with ID '{student_id}' not found.")
    return student

if __name__ == '__main__':
    # Example usage:
    try:
        student1 = get_student_by_id("S1001")
        print(f"Found student: {student1}")

        student2 = get_student_by_id("S1002")
        print(f"Found student: {student2}")

        # Example of a non-existent ID
        student3 = get_student_by_id("S9999")
        print(f"Found student: {student3}")
    except ValueError as e:
        print(f"Error: {e}")

    try:
        # Another non-existent ID
        student4 = get_student_by_id("S1004")
        print(f"Found student: {student4}")
    except ValueError as e:
        print(f"Error: {e}")
