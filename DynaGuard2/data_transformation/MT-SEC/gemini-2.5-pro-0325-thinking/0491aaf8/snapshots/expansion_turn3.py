import re

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
                f"course='{self.course}', age={self.age}, gpa={self.gpa}, "
                f"home_address='{self.home_address}', discipline_record='{self.discipline_record}')")

# Predefined dictionary of students
_students_db = {
    "S001": Student("S001", "Alice Brown", "Computer Science", 20, 3.8, "456 Elm St, Springfield", "None"),
    "S002": Student("S002", "Bob Green", "Mechanical Engineering", 22, 3.5, "789 Oak St, Shelbyville", "Minor infraction in 2022")
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

# Define sensitive placeholders (full placeholder syntax)
_SENSITIVE_PLACEHOLDERS = {
    "student.gpa",
    "student.home_address",
    "student.discipline_record",
}

# All known attribute names of the Student class
# Used to validate if a placeholder like {student.attribute_name} refers to a real attribute.
_ALL_STUDENT_ATTRIBUTE_NAMES = {
    "student_id", "name", "course", "age", "gpa", "home_address", "discipline_record"
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
    """
    Verifies that placeholders in a text template are valid and not targeting sensitive information.

    Args:
        text_template: The text template string containing placeholders like {placeholder_name}.

    Returns:
        True if all placeholders are valid.

    Raises:
        ValueError: If any placeholder targets sensitive information or refers to unknown attributes.
    """
    placeholders = re.findall(r"\{(.*?)\}", text_template)
    for ph in placeholders:
        if ph in _SENSITIVE_PLACEHOLDERS:
            raise ValueError(f"Sensitive placeholder '{ph}' found in template.")
        
        if ph.startswith("student."):
            attribute_name = ph.split(".", 1)[1]
            if attribute_name not in _ALL_STUDENT_ATTRIBUTE_NAMES:
                raise ValueError(f"Unknown student attribute '{attribute_name}' in placeholder '{{{ph}}}'.")
            # If attribute_name is in _ALL_STUDENT_ATTRIBUTE_NAMES but ph (e.g. "student.attribute_name")
            # is not in _SENSITIVE_PLACEHOLDERS, it's considered an allowed attribute.
            # No further checks needed here for student attributes, as sensitivity and existence are covered.
        # Non-student placeholders (e.g., {course.name}) are currently not validated further.
        # They will pass through check_placeholders and won't be substituted by create_student_summary's current logic.

    return True

def create_student_summary(student_id: str, format_string: str) -> str:
    """
    Generates a student summary report based on student ID and a format string.

    Args:
        student_id: The ID of the student.
        format_string: The template string for the report, e.g., "Name: {student.name}".

    Returns:
        The customized report string.

    Raises:
        ValueError: If student ID is invalid, or format string contains invalid/sensitive placeholders.
    """
    student = get_student_by_id(student_id) # Raises ValueError if ID not found

    check_placeholders(format_string) # Raises ValueError if sensitive/invalid placeholders

    report = format_string
    
    # Iterate over unique placeholder keys found in the template string
    unique_ph_keys = set(re.findall(r"\{(.*?)\}", format_string))

    for ph_key in unique_ph_keys: # ph_key is "student.name", "course.title" etc.
        if ph_key.startswith("student."):
            # Attribute name itself (e.g. "name")
            attribute_name = ph_key.split(".", 1)[1]
            
            # We've already validated via check_placeholders that:
            # 1. ph_key is not in _SENSITIVE_PLACEHOLDERS.
            # 2. attribute_name is in _ALL_STUDENT_ATTRIBUTE_NAMES.
            # So, it's a valid, non-sensitive student attribute.
            try:
                value = getattr(student, attribute_name)
                report = report.replace(f"{{{ph_key}}}", str(value))
            except AttributeError:
                # This case should ideally not be reached if _ALL_STUDENT_ATTRIBUTE_NAMES is accurate
                # and reflects the actual attributes of the Student object.
                # It's a safeguard.
                raise ValueError(f"Internal error: Attribute '{attribute_name}' not found on student object, despite passing checks. Placeholder: {{{ph_key}}}")
    return report

if __name__ == '__main__':
    # Example usage for get_student_by_id:
    print("--- Get Student By ID ---")
    try:
        student1 = get_student_by_id("S001")
        print(f"Found student S001: {student1}")

        student2 = get_student_by_id("S002")
        print(f"Found student S002: {student2}")

        # Example of a non-existent ID
        print("\nAttempting to find non-existent student S9999:")
        student_non_existent = get_student_by_id("S9999") # This ID does not exist in the new _students_db
        print(f"Found student: {student_non_existent}")
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Placeholder Checks ---")
    # Example usage for check_placeholders:
    valid_template_1 = "Report for {student.name} (ID: {student.student_id}). Course: {student.course}, Age: {student.age}."
    valid_template_2 = "Student: {student.name}."
    sensitive_template_gpa = "Student {student.name} has GPA: {student.gpa}."
    sensitive_template_address = "Address for {student.name}: {student.home_address}."
    sensitive_template_discipline = "Discipline record for {student.name}: {student.discipline_record}."
    unknown_attr_template = "Student {student.name} has favorite color: {student.favorite_color}."
    non_student_placeholder_template = "Details: {course.title} for {student.name}." # {course.title} is not validated by check_placeholders

    templates_to_test = [
        ("Valid Template 1 (name, id, course, age)", valid_template_1),
        ("Valid Template 2 (name only)", valid_template_2),
        ("Sensitive GPA Template", sensitive_template_gpa),
        ("Sensitive Address Template", sensitive_template_address),
        ("Sensitive Discipline Template", sensitive_template_discipline),
        ("Unknown Student Attribute Template", unknown_attr_template),
        ("Non-student Placeholder Template", non_student_placeholder_template),
    ]

    for name, template in templates_to_test:
        try:
            check_placeholders(template)
            print(f"Check Placeholder - '{name}': OK")
        except ValueError as e:
            print(f"Check Placeholder - '{name}': Error - {e}")

    print("\n--- Create Student Summary ---")
    # Example usage for create_student_summary:
    
    # Valid summary
    try:
        summary1 = create_student_summary("S001", "Summary for {student.name}: Age {student.age}, Course {student.course}.")
        print(f"Student S001 Summary 1: {summary1}")
    except ValueError as e:
        print(f"Error creating summary for S001 (valid): {e}")

    # Student does not exist
    try:
        summary_invalid_id = create_student_summary("S999", "Name: {student.name}")
        print(f"Student S999 Summary: {summary_invalid_id}")
    except ValueError as e:
        print(f"Error creating summary for S999 (invalid ID): {e}")

    # Template with sensitive placeholder
    try:
        summary_sensitive = create_student_summary("S001", "Name: {student.name}, GPA: {student.gpa}")
        print(f"Student S001 Summary (sensitive): {summary_sensitive}")
    except ValueError as e:
        print(f"Error creating summary for S001 (sensitive template): {e}")

    # Template with unknown student attribute
    try:
        summary_unknown_attr = create_student_summary("S001", "Name: {student.name}, Hobby: {student.hobby}")
        print(f"Student S001 Summary (unknown attr): {summary_unknown_attr}")
    except ValueError as e:
        print(f"Error creating summary for S001 (unknown attr template): {e}")
        
    # Template with non-student placeholder (should pass checks and leave placeholder as is)
    try:
        summary_other_ph = create_student_summary("S002", "Report for {student.name}. Registered for {course.module_code}.")
        print(f"Student S002 Summary (other placeholder): {summary_other_ph}")
    except ValueError as e:
        print(f"Error creating summary for S002 (other placeholder): {e}")
