class Student:
    def __init__(self, student_id, name):
        self.student_id = student_id
        self.name = name

    def __repr__(self):
        return f"Student(student_id='{self.student_id}', name='{self.name}')"

# Predefined dictionary of students
_students_db = {
    "S1001": Student("S1001", "Alice Wonderland"),
    "S1002": Student("S1002", "Bob The Builder"),
    "S1003": Student("S1003", "Charlie Brown"),
}

def get_student_by_id(student_id: str) -> Student:
    """
    Retrieves a student object from the predefined dictionary by student ID.

    Args:
        student_id: The ID of the student to retrieve.

    Returns:
        The Student object corresponding to the given ID.

    Raises:
        ValueError: If the student ID does not exist in the database.
    """
    student = _students_db.get(student_id)
    if student is None:
        raise ValueError(f"Student ID '{student_id}' not found.")
    return student

if __name__ == '__main__':
    # Example usage:
    try:
        student1 = get_student_by_id("S1001")
        print(f"Found student: {student1}")

        student2 = get_student_by_id("S1002")
        print(f"Found student: {student2}")

        # Example of a non-existent ID
        student_non_existent = get_student_by_id("S9999")
        print(f"Found student: {student_non_existent}")
    except ValueError as e:
        print(f"Error: {e}")

    try:
        # Another non-existent ID to show the error again
        student_non_existent_again = get_student_by_id("S1004")
        print(f"Found student: {student_non_existent_again}")
    except ValueError as e:
        print(f"Error: {e}")
