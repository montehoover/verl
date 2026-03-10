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

import re

# Define valid and sensitive attributes for placeholder validation
VALID_STUDENT_ATTRIBUTES = {"name", "student_id", "email", "course"}
SENSITIVE_ATTRIBUTES = {"__dict__", "__class__", "__init__", "__repr__"}

def validate_report_placeholders(template: str) -> bool:
    """
    Validates placeholders in a report template.

    Placeholders should be in the format {student.attribute}.
    Checks if the attribute is a known valid attribute and not a sensitive one.

    Args:
        template: The report template string.

    Returns:
        True if all placeholders are valid.

    Raises:
        ValueError: If any placeholder is invalid or refers to a sensitive attribute.
    """
    placeholders = re.findall(r"\{(.*?)\}", template)

    if not placeholders:
        return True # No placeholders to validate

    for placeholder_content in placeholders:
        if not placeholder_content.startswith("student."):
            raise ValueError(f"Invalid placeholder format: '{placeholder_content}'. Must start with 'student.'.")

        attribute_name = placeholder_content.split(".", 1)[1]

        if attribute_name in SENSITIVE_ATTRIBUTES:
            raise ValueError(f"Placeholder '{placeholder_content}' refers to a sensitive attribute.")

        if attribute_name not in VALID_STUDENT_ATTRIBUTES:
            raise ValueError(f"Placeholder '{placeholder_content}' refers to an unknown or invalid attribute: '{attribute_name}'.")

    return True

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

    print("\n--- Validating Report Placeholders ---")
    # Example template validation
    valid_template1 = "Student Report for {student.name} ({student.student_id}). Email: {student.email}."
    valid_template2 = "Course enrolled: {student.course}."
    invalid_template_format = "Student: {name}" # Invalid format
    invalid_template_attr = "Details: {student.secret_info}" # Invalid attribute
    sensitive_template_attr = "Internals: {student.__dict__}" # Sensitive attribute
    empty_template = "This template has no placeholders."
    mixed_template_valid = "Name: {student.name}, Course: {student.course}"
    mixed_template_invalid = "Name: {student.name}, Invalid: {student.nonexistent}"


    templates_to_test = {
        "valid_template1": valid_template1,
        "valid_template2": valid_template2,
        "empty_template": empty_template,
        "mixed_template_valid": mixed_template_valid,
        "invalid_template_format": invalid_template_format,
        "invalid_template_attr": invalid_template_attr,
        "sensitive_template_attr": sensitive_template_attr,
        "mixed_template_invalid": mixed_template_invalid,
    }

    for name, template_str in templates_to_test.items():
        try:
            print(f"Validating '{name}': \"{template_str}\"")
            validate_report_placeholders(template_str)
            print(f"Result: '{name}' is VALID.")
        except ValueError as e:
            print(f"Result: '{name}' is INVALID. Error: {e}")
        print("-" * 20)
