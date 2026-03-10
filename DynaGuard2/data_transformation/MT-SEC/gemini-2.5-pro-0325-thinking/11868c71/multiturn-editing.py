class Student:
    def __init__(self, student_id, name, course, age, gpa, home_address, discipline_record):
        self.student_id = student_id
        self.name = name
        self.course = course
        self.age = age
        self.gpa = gpa
        self.home_address = home_address
        self.discipline_record = discipline_record

students = {
    "S001": Student("S001", "Alice Brown", "Computer Science", 20, 3.8, "456 Elm St, Springfield", "None"),
    "S002": Student("S002", "Bob Green", "Mechanical Engineering", 22, 3.5, "789 Oak St, Shelbyville", "Minor infraction in 2022")
}

ALLOWED_PUBLIC_ATTRIBUTES = ['student_id', 'name', 'course', 'age']
SENSITIVE_ATTRIBUTES = ['gpa', 'home_address', 'discipline_record']

class StudentView:
    """
    A view for the Student object that exposes only allowed public attributes
    and handles access to sensitive attributes.
    """
    def __init__(self, student_obj):
        self._student_obj = student_obj
        # Populate only allowed public attributes
        for attr_name in ALLOWED_PUBLIC_ATTRIBUTES:
            if hasattr(self._student_obj, attr_name):
                setattr(self, attr_name, getattr(self._student_obj, attr_name))
            # If student_obj doesn't have an allowed attribute, it won't be set on StudentView.
            # Accessing it via format() will then lead to __getattr__ or AttributeError.

    def __getattr__(self, name):
        """
        Called when an attribute is not found in the usual places.
        Used here to intercept access to sensitive attributes or undefined attributes.
        """
        if name in SENSITIVE_ATTRIBUTES:
            raise ValueError(f"Access to sensitive attribute '{name}' is not allowed.")
        # For any other attribute not found (neither explicitly allowed and set, nor sensitive),
        # raise an AttributeError. This is standard behavior.
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}', or it is not an allowed public attribute.")

def build_student_profile(unique_id, report_template):
    """
    Generates a student profile string based on a template.

    Args:
        unique_id (str): The ID of the student.
        report_template (str): A string template with placeholders like
                               '{student.name}', '{student.course}'.

    Returns:
        str: The generated report string.

    Raises:
        ValueError: If unique_id does not exist, or if the template attempts
                    to access sensitive attributes.
        AttributeError: If the template attempts to access a non-sensitive attribute
                        that is not defined for the student.
    """
    if unique_id not in students:
        raise ValueError(f"Student ID '{unique_id}' not found.")
    
    student_obj = students[unique_id]
    student_view = StudentView(student_obj)
    
    try:
        # The format() method will interact with StudentView.
        # StudentView.__getattr__ will handle requests for attributes.
        return report_template.format(student=student_view)
    except ValueError:  # Re-raise ValueError from StudentView (sensitive attribute access)
        raise
    # AttributeError for non-sensitive, non-allowed/missing attributes will propagate from format()

if __name__ == '__main__':
    # Example usage:
    valid_template = "Name: {student.name}, Course: {student.course}, Age: {student.age}, ID: {student.student_id}"
    sensitive_template = "Name: {student.name}, GPA: {student.gpa}"
    missing_attr_template = "Name: {student.name}, Hobby: {student.hobby}"

    print("--- Profile for S001 (Valid Template) ---")
    try:
        profile_s001 = build_student_profile("S001", valid_template)
        print(profile_s001)
    except (ValueError, AttributeError) as e:
        print(f"Error: {e}")

    print("\n--- Profile for S002 (Valid Template) ---")
    try:
        profile_s002 = build_student_profile("S002", valid_template)
        print(profile_s002)
    except (ValueError, AttributeError) as e:
        print(f"Error: {e}")

    print("\n--- Profile for S001 (Sensitive Template) ---")
    try:
        profile_s001_sensitive = build_student_profile("S001", sensitive_template)
        print(profile_s001_sensitive)
    except (ValueError, AttributeError) as e:
        print(f"Error: {e}")

    print("\n--- Profile for S001 (Missing Attribute Template) ---")
    try:
        profile_s001_missing = build_student_profile("S001", missing_attr_template)
        print(profile_s001_missing)
    except (ValueError, AttributeError) as e:
        print(f"Error: {e}")
        
    print("\n--- Profile for S003 (Non-existent Student) ---")
    try:
        profile_s003 = build_student_profile("S003", valid_template)
        print(profile_s003)
    except ValueError as e:
        print(f"Error: {e}")
