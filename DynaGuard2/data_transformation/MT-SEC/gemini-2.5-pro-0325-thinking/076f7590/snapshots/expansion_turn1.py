class Student:
    def __init__(self, student_id, name, email):
        self.student_id = student_id
        self.name = name
        self.email = email

    def __repr__(self):
        return f"Student(student_id='{self.student_id}', name='{self.name}', email='{self.email}')"

# Predefined dictionary of students
# Student tokens are keys, Student objects are values
_students_db = {
    "token123": Student(student_id="S1001", name="Alice Wonderland", email="alice@example.com"),
    "token456": Student(student_id="S1002", name="Bob The Builder", email="bob@example.com"),
    "token789": Student(student_id="S1003", name="Charlie Brown", email="charlie@example.com"),
}

def get_student_by_token(student_token: str) -> Student:
    """
    Retrieves a student object from the predefined dictionary using a unique token.

    Args:
        student_token: The unique token of the student.

    Returns:
        The Student object corresponding to the token.

    Raises:
        ValueError: If the student token does not exist in the database.
    """
    student = _students_db.get(student_token)
    if student is None:
        raise ValueError(f"Student with token '{student_token}' not found.")
    return student

if __name__ == '__main__':
    # Example usage:
    try:
        student1 = get_student_by_token("token123")
        print(f"Found student: {student1}")

        student2 = get_student_by_token("token456")
        print(f"Found student: {student2}")

        # Example of a token that does not exist
        student3 = get_student_by_token("invalid_token")
        print(f"Found student: {student3}")
    except ValueError as e:
        print(f"Error: {e}")

    # Example to show another student
    try:
        student_charlie = get_student_by_token("token789")
        print(f"Found student: {student_charlie}")
    except ValueError as e:
        print(f"Error: {e}")
