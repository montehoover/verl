class Student:
    def __init__(self, student_id, name, age):
        self.student_id = student_id
        self.name = name
        self.age = age

    def __repr__(self):
        return f"Student(student_id='{self.student_id}', name='{self.name}', age={self.age})"

# Predefined dictionary of students
# In a real application, this data might come from a database or an external file.
_students_db = {
    "S1001": Student(student_id="S1001", name="Alice Wonderland", age=20),
    "S1002": Student(student_id="S1002", name="Bob The Builder", age=22),
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

import re

# Define allowed attributes for student placeholders
# 'course' is included as per the example in the prompt.
# In a real system, this list would be carefully managed.
ALLOWED_STUDENT_ATTRIBUTES = {"name", "age", "student_id", "course"}
SENSITIVE_ATTRIBUTE_PREFIXES = ("_", "__") # Attributes starting with these are forbidden

def validate_report_template(template_string: str) -> bool:
    """
    Validates a report template string for valid student placeholders.

    Placeholders should be in the format {student.attribute_name}.
    Valid attributes are defined in ALLOWED_STUDENT_ATTRIBUTES.
    Attributes starting with SENSITIVE_ATTRIBUTE_PREFIXES are forbidden.

    Args:
        template_string: The template string to validate.

    Returns:
        True if all placeholders are valid.

    Raises:
        ValueError: If an invalid or sensitive placeholder is found.
    """
    placeholders = re.findall(r"\{(.*?)\}", template_string)

    if not placeholders and template_string: # Template has content but no placeholders
        return True
    if not template_string: # Empty template is valid
        return True

    for ph_content in placeholders:
        if not ph_content.startswith("student."):
            raise ValueError(f"Invalid placeholder format: '{ph_content}'. Must start with 'student.'.")

        attribute_name = ph_content.split(".", 1)[1]

        if not attribute_name:
            raise ValueError(f"Empty attribute name in placeholder: '{ph_content}'.")

        if attribute_name.startswith(SENSITIVE_ATTRIBUTE_PREFIXES):
            raise ValueError(f"Sensitive or private attribute access attempted in placeholder: '{ph_content}'.")

        if attribute_name not in ALLOWED_STUDENT_ATTRIBUTES:
            raise ValueError(f"Invalid student attribute in placeholder: '{attribute_name}' in '{ph_content}'. Allowed attributes are: {ALLOWED_STUDENT_ATTRIBUTES}.")

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
        # Another non-existent ID
        student_non_existent_2 = get_student_by_id("S1004")
        print(f"Found student: {student_non_existent_2}")
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Testing validate_report_template ---")
    valid_templates = [
        "Student Name: {student.name}, Age: {student.age}",
        "Report for {student.student_id}.",
        "Course enrolled: {student.course}.",
        "No placeholders here.",
        ""
    ]
    invalid_templates = [
        ("Student Name: {student.name}, Secret: {student._secret_info}", "Sensitive or private attribute access attempted"),
        ("Details: {student.__repr__}", "Sensitive or private attribute access attempted"),
        ("Student GPA: {student.gpa}", "Invalid student attribute in placeholder: 'gpa'"),
        ("User: {user.name}", "Invalid placeholder format: 'user.name'"),
        ("Student: {student.}", "Empty attribute name in placeholder"),
        ("Student: {student.name} and {student.address}", "Invalid student attribute in placeholder: 'address'")
    ]

    for i, template in enumerate(valid_templates):
        try:
            validate_report_template(template)
            print(f"Valid template {i+1} ('{template}') PASSED validation.")
        except ValueError as e:
            print(f"Valid template {i+1} ('{template}') FAILED validation: {e}")

    for i, (template, error_msg_part) in enumerate(invalid_templates):
        try:
            validate_report_template(template)
            print(f"Invalid template {i+1} ('{template}') UNEXPECTEDLY PASSED validation.")
        except ValueError as e:
            if error_msg_part in str(e):
                print(f"Invalid template {i+1} ('{template}') CORRECTLY FAILED validation: {e}")
            else:
                print(f"Invalid template {i+1} ('{template}') FAILED with UNEXPECTED error: {e}. Expected part: '{error_msg_part}'")
