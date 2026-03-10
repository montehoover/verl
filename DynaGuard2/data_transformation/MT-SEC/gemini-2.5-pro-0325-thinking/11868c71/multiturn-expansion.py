class Student:
    def __init__(self, student_id, name, course, age, gpa, home_address, discipline_record):
        self.student_id = student_id
        self.name = name
        self.course = course
        self.age = age
        self.gpa = gpa
        self.home_address = home_address
        self.discipline_record = discipline_record

    def __repr__(self):
        return (f"Student(student_id='{self.student_id}', name='{self.name}', "
                f"course='{self.course}', age={self.age}, gpa={self.gpa})")

# Predefined dictionary of students
# In a real application, this data might come from a database or a file.
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
    student = students.get(student_id) # Updated to use 'students' dictionary
    if student is None:
        raise ValueError(f"Student with ID '{student_id}' not found.")
    return student

# Allowed attributes for the Student object in templates
VALID_STUDENT_ATTRIBUTES = {"student_id", "name", "course", "age", "gpa"} # Updated attributes

def validate_report_template(template_string: str) -> bool:
    """
    Validates a report template string for valid placeholders.

    Placeholders should be in the format {student.attribute}, where attribute
    is one of the allowed student attributes (e.g., name, age, student_id).

    Args:
        template_string: The report template string to validate.

    Returns:
        True if all placeholders are valid.

    Raises:
        ValueError: If any placeholder is invalid (e.g., wrong format,
                    refers to a non-student object, or uses a non-allowed
                    or sensitive attribute).
    """
    import re
    # Regex to find anything that looks like a placeholder: {content}
    placeholders = re.findall(r'\{([^}]+)\}', template_string)

    if not placeholders and '{' in template_string and '}' in template_string:
        # Handles cases like "{}" or "{invalid content without dot}"
        # This specific check might be too broad, but the goal is to catch malformed placeholders.
        # A more precise regex for the initial findall might be better,
        # but the current loop handles specific format checks.
        pass # Let the loop below handle more specific errors if needed.

    for placeholder_content in placeholders:
        parts = placeholder_content.split('.')
        if len(parts) != 2:
            raise ValueError(f"Invalid placeholder format: '{placeholder_content}'. Expected 'object.attribute'.")

        object_name, attribute_name = parts
        if object_name != "student":
            raise ValueError(f"Invalid object in placeholder: '{object_name}'. Only 'student' is allowed.")

        if attribute_name not in VALID_STUDENT_ATTRIBUTES:
            raise ValueError(f"Invalid or sensitive attribute in placeholder: '{attribute_name}'. Allowed attributes are: {', '.join(VALID_STUDENT_ATTRIBUTES)}.")

    return True

def build_student_profile(unique_id: str, report_template: str) -> str:
    """
    Generates a student report by replacing placeholders in a template with student details.

    Args:
        unique_id: The ID of the student.
        report_template: The template string with placeholders like {student.name}.

    Returns:
        The formatted report string.

    Raises:
        ValueError: If the student ID does not exist, or if the template
                    contains invalid or sensitive placeholders.
    """
    student = get_student_by_id(unique_id)  # Raises ValueError if ID not found

    # This will raise ValueError if template is invalid (e.g., uses sensitive placeholders
    # like {student.home_address} or non-existent attributes like {student.hobby},
    # or malformed placeholders).
    validate_report_template(report_template)

    current_report = report_template
    
    # Replace valid placeholders with student data
    # validate_report_template ensures that any {student.attribute} placeholder
    # has 'attribute' in VALID_STUDENT_ATTRIBUTES.
    for attr_name in VALID_STUDENT_ATTRIBUTES:
        placeholder = f"{{student.{attr_name}}}"
        # getattr will fetch the actual data from the student object
        value = getattr(student, attr_name)
        current_report = current_report.replace(placeholder, str(value))
            
    return current_report

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

    print("\n--- Template Validation Examples ---")
    valid_template1 = "Student Report for {student.name}, ID: {student.student_id}."
    valid_template2 = "Age: {student.age}."
    invalid_template_format = "Student: {student_name}" # Missing dot
    invalid_template_object = "Details: {user.name}"
    invalid_template_attribute = "Course: {student.course}"
    invalid_template_sensitive = "Details: {student.password}" # Assuming password is sensitive and not in VALID_STUDENT_ATTRIBUTES
    empty_template = "This is a template with no placeholders."
    template_with_curly_braces_not_placeholder = "This is a set: {1, 2, 3}"


    templates_to_test = {
        "Valid Template 1": valid_template1,
        "Valid Template 2": valid_template2,
        "Empty Template": empty_template,
        "Template with non-placeholder braces": template_with_curly_braces_not_placeholder,
        "Invalid Format (missing dot)": invalid_template_format,
        "Invalid Object": invalid_template_object,
        "Invalid Attribute (course)": invalid_template_attribute,
        "Invalid Attribute (sensitive)": invalid_template_sensitive,
    }

    for name, template in templates_to_test.items():
        try:
            print(f"Validating '{name}': \"{template}\"")
            is_valid = validate_report_template(template)
            print(f"Result: {'Valid' if is_valid else 'Invalid (but should have raised error)'}")
        except ValueError as e:
            print(f"Result: Invalid - {e}")
        print("-" * 20)

    print("\n--- Build Student Profile Examples ---")
    template1 = "Student Name: {student.name}, Age: {student.age}, Course: {student.course}, GPA: {student.gpa}"
    template_sensitive = "Student Name: {student.name}, Address: {student.home_address}"
    template_invalid_attr = "Student Name: {student.name}, Hobby: {student.hobby}"

    # Test cases for build_student_profile
    profile_tests = [
        ("Valid Profile S001", "S001", template1),
        ("Valid Profile S002", "S002", template1),
        ("Non-existent Student ID", "S999", template1),
        ("Template with Sensitive Placeholder", "S001", template_sensitive),
        ("Template with Invalid Attribute", "S001", template_invalid_attr),
    ]

    for name, student_id, template_str in profile_tests:
        try:
            print(f"Generating report for '{name}': Student ID '{student_id}'")
            report = build_student_profile(student_id, template_str)
            print(f"Generated Report:\n{report}")
        except ValueError as e:
            print(f"Error generating report: {e}")
        print("-" * 20)
