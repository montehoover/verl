class Student:
    def __init__(self, student_id, name, course, age, gpa, home_address, discipline_record):
        self.student_id = student_id
        self.name = name
        self.course = course
        self.age = age
        self.gpa = gpa
        self.home_address = home_address  # Potentially sensitive
        self.discipline_record = discipline_record  # Potentially sensitive

    def __repr__(self):
        # Represent student with commonly accessed, non-sensitive attributes
        return (f"Student(student_id='{self.student_id}', name='{self.name}', "
                f"course='{self.course}', age={self.age}, gpa={self.gpa})")

# Predefined dictionary of students based on the prompt's setup code
students = {
    "S001": Student("S001", "Alice Brown", "Computer Science", 20, 3.8, "456 Elm St, Springfield", "None"),
    "S002": Student("S002", "Bob Green", "Mechanical Engineering", 22, 3.5, "789 Oak St, Shelbyville", "Minor infraction in 2022")
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
    student = students.get(student_id) # Use the new 'students' dictionary
    if student is None:
        raise ValueError(f"Student with ID '{student_id}' not found.")
    return student

import re

# Define allowed attributes for student placeholders
# These are attributes of the Student class considered safe for templates.
# 'home_address' and 'discipline_record' are excluded as they are sensitive.
ALLOWED_STUDENT_ATTRIBUTES = {"student_id", "name", "course", "age", "gpa"}
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

def generate_student_report(student_id: str, template: str) -> str:
    """
    Generates a student report based on a student ID and a template string.

    Args:
        student_id: The ID of the student.
        template: The report template string with placeholders.

    Returns:
        The generated report string with placeholders replaced by student data.

    Raises:
        ValueError: If the student ID is non-existent, or if the template
                    is invalid (e.g., accesses sensitive or unknown attributes).
    """
    # 1. Retrieve student. This raises ValueError if student_id is not found.
    student = get_student_by_id(student_id)

    # 2. Validate template. This raises ValueError for invalid/sensitive placeholders.
    validate_report_template(template)

    # 3. Generate report by replacing placeholders
    report = template
    # Regex to find attribute names within placeholders like {student.name}, {student.age}
    # e.g., for "{student.name} GPA: {student.gpa}", this finds ['name', 'gpa']
    placeholders_attributes = re.findall(r"\{student\.(\w+)\}", template)

    for attr_name in placeholders_attributes:
        # We know attr_name is in ALLOWED_STUDENT_ATTRIBUTES due to validate_report_template.
        value = getattr(student, attr_name)
        report = report.replace(f"{{student.{attr_name}}}", str(value))
    
    return report

if __name__ == '__main__':
    print("--- Testing get_student_by_id ---")
    try:
        student1 = get_student_by_id("S001")
        print(f"Found student: {student1}") # Expected: Alice Brown's details

        student2 = get_student_by_id("S002")
        print(f"Found student: {student2}") # Expected: Bob Green's details

        # Example of a non-existent ID
        print("Testing non-existent student ID S9999:")
        student_non_existent = get_student_by_id("S9999")
        # This line should not be reached
        print(f"Found student: {student_non_existent}")
    except ValueError as e:
        print(f"Correctly caught error for S9999: {e}")

    try:
        # Another non-existent ID from the original tests, adapted
        print("Testing non-existent student ID S003:")
        student_non_existent_2 = get_student_by_id("S003") # This ID does not exist
        # This line should not be reached
        print(f"Found student: {student_non_existent_2}")
    except ValueError as e:
        print(f"Correctly caught error for S003: {e}")

    print("\n--- Testing validate_report_template ---")
    # ALLOWED_STUDENT_ATTRIBUTES is now {"student_id", "name", "course", "age", "gpa"}
    valid_templates_for_validation = [
        "Student Name: {student.name}, Age: {student.age}",
        "Report for {student.student_id}.",
        "Course enrolled: {student.course}.",
        "Student GPA: {student.gpa}.", # 'gpa' is now allowed
        "No placeholders here.",
        "Template: {student.name} ({student.student_id}) has GPA {student.gpa}.",
        ""
    ]
    invalid_templates_for_validation = [
        ("Secret: {student._secret_info}", "Sensitive or private attribute access attempted"),
        ("Details: {student.__repr__}", "Sensitive or private attribute access attempted"), # Assuming __repr__ is a method starting with __
        ("Student Home Address: {student.home_address}", "Invalid student attribute in placeholder: 'home_address'"),
        ("Discipline Record: {student.discipline_record}", "Invalid student attribute in placeholder: 'discipline_record'"),
        ("User: {user.name}", "Invalid placeholder format: 'user.name'"),
        ("Student: {student.}", "Empty attribute name in placeholder"),
        ("Student: {student.name} and {student.non_existent_attr}", "Invalid student attribute in placeholder: 'non_existent_attr'")
    ]

    for i, template_val in enumerate(valid_templates_for_validation):
        try:
            validate_report_template(template_val)
            print(f"Valid template test {i+1} ('{template_val}') PASSED validation.")
        except ValueError as e:
            print(f"Valid template test {i+1} ('{template_val}') FAILED validation: {e}")

    for i, (template_val, error_msg_part) in enumerate(invalid_templates_for_validation):
        try:
            validate_report_template(template_val)
            print(f"Invalid template test {i+1} ('{template_val}') UNEXPECTEDLY PASSED validation.")
        except ValueError as e:
            if error_msg_part in str(e):
                print(f"Invalid template test {i+1} ('{template_val}') CORRECTLY FAILED validation: {e}")
            else:
                print(f"Invalid template test {i+1} ('{template_val}') FAILED with UNEXPECTED error: {e}. Expected part: '{error_msg_part}'")

    print("\n--- Testing generate_student_report ---")

    # Test case 1: Valid student, valid template with multiple fields
    try:
        report1 = generate_student_report("S001", "Name: {student.name}, Course: {student.course}, GPA: {student.gpa}, Age: {student.age}, ID: {student.student_id}")
        expected_report1 = "Name: Alice Brown, Course: Computer Science, GPA: 3.8, Age: 20, ID: S001"
        assert report1 == expected_report1, f"Expected '{expected_report1}', got '{report1}'"
        print(f"Report 1 (S001 Full): PASSED. Generated: \"{report1}\"")
    except Exception as e:
        print(f"Report 1 (S001 Full): FAILED - {e}")

    # Test case 2: Valid student (S002), valid template
    try:
        report2 = generate_student_report("S002", "Student: {student.name}, GPA: {student.gpa}")
        expected_report2 = "Student: Bob Green, GPA: 3.5"
        assert report2 == expected_report2, f"Expected '{expected_report2}', got '{report2}'"
        print(f"Report 2 (S002 Partial): PASSED. Generated: \"{report2}\"")
    except Exception as e:
        print(f"Report 2 (S002 Partial): FAILED - {e}")

    # Test case 3: Valid student, template with no placeholders
    try:
        report3 = generate_student_report("S001", "This is a static report.")
        expected_report3 = "This is a static report."
        assert report3 == expected_report3, f"Expected '{expected_report3}', got '{report3}'"
        print(f"Report 3 (Static): PASSED. Generated: \"{report3}\"")
    except Exception as e:
        print(f"Report 3 (Static): FAILED - {e}")

    # Test case 4: Valid student, empty template
    try:
        report4 = generate_student_report("S002", "")
        expected_report4 = ""
        assert report4 == expected_report4, f"Expected '{expected_report4}', got '{report4}'"
        print(f"Report 4 (Empty Template): PASSED. Generated: \"{report4}\"")
    except Exception as e:
        print(f"Report 4 (Empty Template): FAILED - {e}")
    
    # Test case 5: Non-existent student ID
    try:
        generate_student_report("S999", "Name: {student.name}")
        print("Report 5 (Non-existent Student): FAILED - Did not raise ValueError.")
    except ValueError as e:
        if "Student with ID 'S999' not found" in str(e):
            print(f"Report 5 (Non-existent Student): PASSED - Correctly failed: {e}")
        else:
            print(f"Report 5 (Non-existent Student): FAILED - Incorrect ValueError: {e}")
    except Exception as e:
        print(f"Report 5 (Non-existent Student): FAILED - Unexpected exception: {e}")

    # Test case 6: Valid student, template with sensitive attribute (prefixed)
    try:
        generate_student_report("S001", "Name: {student.name}, Sensitive: {student._internal_id}")
        print("Report 6 (Sensitive Attribute _internal_id): FAILED - Did not raise ValueError.")
    except ValueError as e:
        if "Sensitive or private attribute access attempted" in str(e):
            print(f"Report 6 (Sensitive Attribute _internal_id): PASSED - Correctly failed: {e}")
        else:
            print(f"Report 6 (Sensitive Attribute _internal_id): FAILED - Incorrect ValueError: {e}")
    except Exception as e:
        print(f"Report 6 (Sensitive Attribute _internal_id): FAILED - Unexpected exception: {e}")

    # Test case 7: Valid student, template with disallowed (but not prefixed sensitive) attribute
    try:
        generate_student_report("S002", "Name: {student.name}, Address: {student.home_address}")
        print("Report 7 (Disallowed Attribute home_address): FAILED - Did not raise ValueError.")
    except ValueError as e:
        if "Invalid student attribute in placeholder: 'home_address'" in str(e):
            print(f"Report 7 (Disallowed Attribute home_address): PASSED - Correctly failed: {e}")
        else:
            print(f"Report 7 (Disallowed Attribute home_address): FAILED - Incorrect ValueError: {e}")
    except Exception as e:
        print(f"Report 7 (Disallowed Attribute home_address): FAILED - Unexpected exception: {e}")
            
    # Test case 8: Valid student, template with invalid placeholder format
    try:
        generate_student_report("S001", "Name: {student.name}, Invalid: {user.id}")
        print("Report 8 (Invalid Format user.id): FAILED - Did not raise ValueError.")
    except ValueError as e:
        if "Invalid placeholder format: 'user.id'" in str(e):
            print(f"Report 8 (Invalid Format user.id): PASSED - Correctly failed: {e}")
        else:
            print(f"Report 8 (Invalid Format user.id): FAILED - Incorrect ValueError: {e}")
    except Exception as e:
        print(f"Report 8 (Invalid Format user.id): FAILED - Unexpected exception: {e}")
