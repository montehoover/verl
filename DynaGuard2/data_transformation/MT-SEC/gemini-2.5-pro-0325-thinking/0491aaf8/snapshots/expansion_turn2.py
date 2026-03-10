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

# Define sensitive placeholders
_SENSITIVE_PLACEHOLDERS = {
    "student.gpa",
    "student.home_address",
    "student.financial_aid_details", # Example of another sensitive field
    "student.medical_conditions"
}

def check_placeholders(text_template: str) -> bool:
    """
    Verifies that placeholders in a text template are valid and not targeting sensitive information.

    Args:
        text_template: The text template string containing placeholders like {placeholder_name}.

    Returns:
        True if all placeholders are valid.

    Raises:
        ValueError: If any placeholder targets sensitive information.
    """
    import re
    placeholders = re.findall(r"\{(.*?)\}", text_template)
    for ph in placeholders:
        if ph in _SENSITIVE_PLACEHOLDERS:
            raise ValueError(f"Sensitive placeholder '{ph}' found in template.")
        # Add more validation logic for placeholder format if needed
        # For example, check if the placeholder refers to valid student attributes
        if ph.startswith("student."):
            # A simple check: ensure it's not an arbitrary attribute.
            # This could be expanded to check against Student class attributes.
            allowed_student_attributes = {"student_id", "name"} # From Student class
            attribute_name = ph.split(".", 1)[1]
            if attribute_name not in allowed_student_attributes:
                # This check is for general validity, not just sensitivity.
                # Depending on requirements, this could be a separate check or combined.
                # For now, let's consider unknown student attributes as potentially problematic.
                # Or, we can simply allow any non-sensitive student attribute.
                # For this implementation, we will focus on the explicitly sensitive ones.
                pass # Allow other student attributes if not in _SENSITIVE_PLACEHOLDERS

    return True

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

    print("\n--- Placeholder Checks ---")
    # Example usage for check_placeholders:
    valid_template_1 = "Report for {student.name} (ID: {student.student_id})."
    valid_template_2 = "Student: {student.name}."
    sensitive_template_gpa = "Student {student.name} has GPA: {student.gpa}."
    sensitive_template_address = "Address for {student.name}: {student.home_address}."
    unknown_placeholder_template = "Details: {course.name} for {student.name}." # Assuming course.name is fine

    templates_to_test = [
        ("Valid Template 1", valid_template_1),
        ("Valid Template 2", valid_template_2),
        ("Sensitive GPA Template", sensitive_template_gpa),
        ("Sensitive Address Template", sensitive_template_address),
        ("Unknown Placeholder Template", unknown_placeholder_template),
    ]

    for name, template in templates_to_test:
        try:
            check_placeholders(template)
            print(f"'{name}': OK")
        except ValueError as e:
            print(f"'{name}': Error - {e}")
