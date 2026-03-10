class Student:
    def __init__(self, student_token, name, course, age, gpa, home_address, discipline_record):
        self.student_token = student_token
        self.name = name
        self.course = course
        self.age = age
        self.gpa = gpa
        self.home_address = home_address
        self.discipline_record = discipline_record

    # Adding a __repr__ for easier debugging, similar to the original.
    def __repr__(self):
        return (f"Student(student_token='{self.student_token}', name='{self.name}', "
                f"course='{self.course}', age={self.age}, gpa={self.gpa}, "
                f"home_address='{self.home_address}', discipline_record='{self.discipline_record}')")

students = {
    "S005": Student("S005", "Eve Smith", "Electrical Engineering", 24, 4.0, "987 Cedar St, Pineville", "None"),
    "S006": Student("S006", "Frank Brown", "Physics", 25, 3.7, "246 Birch St, Oakton", "Warning in 2021"),
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
    student = students.get(student_token)
    if student is None:
        raise ValueError(f"Student with token '{student_token}' not found.")
    return student

import re

# Define valid and sensitive attributes for placeholder validation
VALID_STUDENT_ATTRIBUTES = {"student_token", "name", "course", "age", "gpa"} # Updated attributes
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

def compile_student_report(student_token: str, report_template: str) -> str:
    """
    Generates a formatted report string by replacing placeholders with student details.

    Args:
        student_token: The unique token of the student.
        report_template: The report template string with placeholders.

    Returns:
        The formatted report string.

    Raises:
        ValueError: If the student token is non-existent, or if the template
                    contains invalid or unauthorized placeholders.
    """
    # 1. Validate placeholders first. This will raise ValueError for bad placeholders.
    validate_report_placeholders(report_template)

    # 2. Retrieve student. This will raise ValueError if token is non-existent.
    student = get_student_by_token(student_token)

    # 3. Compile the report by replacing placeholders
    def replace_match(match):
        attribute_name = match.group(1) # The part after "student."
        # getattr will fetch the student's attribute.
        # validate_report_placeholders ensures attribute_name is in VALID_STUDENT_ATTRIBUTES
        # and not in SENSITIVE_ATTRIBUTES.
        value = getattr(student, attribute_name)
        return str(value)

    compiled_report = re.sub(r"\{student\.(.*?)\}", replace_match, report_template)
    return compiled_report

if __name__ == '__main__':
    # Example usage for get_student_by_token (updated for new students)
    print("--- Testing get_student_by_token ---")
    try:
        student_eve = get_student_by_token("S005")
        print(f"Found student: {student_eve}")

        student_frank = get_student_by_token("S006")
        print(f"Found student: {student_frank}")

        # Example of a token that does not exist
        student_invalid = get_student_by_token("invalid_token")
        print(f"Found student: {student_invalid}")
    except ValueError as e:
        print(f"Error: {e}")
    print("-" * 20)

    print("\n--- Validating Report Placeholders ---")
    # Example template validation (updated for new attributes)
    valid_template1 = "Student Report for {student.name} ({student.student_token}). Age: {student.age}." # student_id -> student_token, email -> age
    valid_template2 = "Course enrolled: {student.course}. GPA: {student.gpa}." # Added GPA
    invalid_template_format = "Student: {name}" # Invalid format (no "student." prefix)
    invalid_template_attr = "Details: {student.home_address}" # Invalid attribute (home_address is not in VALID_STUDENT_ATTRIBUTES)
    sensitive_template_attr = "Internals: {student.__dict__}" # Sensitive attribute
    empty_template = "This template has no placeholders."
    mixed_template_valid = "Name: {student.name}, Course: {student.course}, GPA: {student.gpa}" # Added GPA
    mixed_template_invalid = "Name: {student.name}, Invalid: {student.discipline_record}" # discipline_record is not in VALID_STUDENT_ATTRIBUTES


    templates_to_test = {
        "valid_template1": valid_template1,
        "valid_template2": valid_template2,
        "empty_template": empty_template,
        "mixed_template_valid": mixed_template_valid,
        "invalid_template_format": invalid_template_format,
        "invalid_template_attr": invalid_template_attr, # Tests home_address
        "sensitive_template_attr": sensitive_template_attr,
        "mixed_template_invalid": mixed_template_invalid, # Tests discipline_record
    }

    for name, template_str in templates_to_test.items():
        try:
            print(f"Validating '{name}': \"{template_str}\"")
            validate_report_placeholders(template_str)
            print(f"Result: '{name}' is VALID.")
        except ValueError as e:
            print(f"Result: '{name}' is INVALID. Error: {e}")
        print("-" * 20)

    print("\n--- Testing compile_student_report ---")
    report_template_valid = "Report for {student.name} ({student.student_token}): Course - {student.course}, Age - {student.age}, GPA - {student.gpa}."
    report_template_invalid_attr = "Report for {student.name}: Address - {student.home_address}." # Uses unauthorized placeholder
    report_template_sensitive_attr = "Report for {student.name}: Internals - {student.__dict__}." # Uses sensitive placeholder

    # Test case 1: Valid report
    try:
        print(f"Compiling report for S005 with template: \"{report_template_valid}\"")
        compiled_report = compile_student_report("S005", report_template_valid)
        print(f"Compiled Report (S005): {compiled_report}")
    except ValueError as e:
        print(f"Error compiling report (S005): {e}")
    print("-" * 20)

    # Test case 2: Valid report for another student
    try:
        print(f"Compiling report for S006 with template: \"{report_template_valid}\"")
        compiled_report = compile_student_report("S006", report_template_valid)
        print(f"Compiled Report (S006): {compiled_report}")
    except ValueError as e:
        print(f"Error compiling report (S006): {e}")
    print("-" * 20)

    # Test case 3: Invalid student token
    try:
        print(f"Compiling report for S999 (invalid token) with template: \"{report_template_valid}\"")
        compiled_report = compile_student_report("S999", report_template_valid)
        print(f"Compiled Report (S999): {compiled_report}")
    except ValueError as e:
        print(f"Error compiling report (S999): {e}")
    print("-" * 20)

    # Test case 4: Template with unauthorized placeholder
    try:
        print(f"Compiling report for S005 with template: \"{report_template_invalid_attr}\"")
        compiled_report = compile_student_report("S005", report_template_invalid_attr)
        print(f"Compiled Report (S005, invalid attr): {compiled_report}")
    except ValueError as e:
        print(f"Error compiling report (S005, invalid attr): {e}")
    print("-" * 20)

    # Test case 5: Template with sensitive placeholder
    try:
        print(f"Compiling report for S005 with template: \"{report_template_sensitive_attr}\"")
        compiled_report = compile_student_report("S005", report_template_sensitive_attr)
        print(f"Compiled Report (S005, sensitive attr): {compiled_report}")
    except ValueError as e:
        print(f"Error compiling report (S005, sensitive attr): {e}")
    print("-" * 20)
