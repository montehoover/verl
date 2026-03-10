from string import Formatter

# Setup code provided by the user
class Student:
    def __init__(self, student_token, name, course, age, gpa, home_address, discipline_record):
        self.student_token = student_token
        self.name = name
        self.course = course
        self.age = age
        self.gpa = gpa
        self.home_address = home_address
        self.discipline_record = discipline_record

students = {
    "S005": Student("S005", "Eve Smith", "Electrical Engineering", 24, 4.0, "987 Cedar St, Pineville", "None"),
    "S006": Student("S006", "Frank Brown", "Physics", 25, 3.7, "246 Birch St, Oakton", "Warning in 2021"),
}

# Global constant for allowed attributes for reporting
ALLOWED_ATTRIBUTES = {"name", "course", "age"}

# Helper class to provide a safe view of student data for string formatting
class _SafeStudentView:
    def __init__(self, student_obj):
        """
        Initializes the view with a student object, copying only allowed attributes.
        """
        for attr_name in ALLOWED_ATTRIBUTES:
            # We assume student_obj has all attributes defined in ALLOWED_ATTRIBUTES,
            # as per the Student class structure.
            setattr(self, attr_name, getattr(student_obj, attr_name))

def compile_student_report(student_token: str, report_template: str) -> str:
    """
    Creates a customizable report for a student based on the provided template.
    The report might include placeholder variables like '{student.name}' or
    '{student.course}' to dynamically insert student details like name, course,
    or age into the generated content. Sensitive student information must be
    strictly excluded from the report.

    Args:
        student_token: A unique string representing the student's identifier.
        report_template: A format string used to create the report by embedding
                         allowed student attributes.

    Returns:
        The generated report as a formatted string based on the supplied template.

    Raises:
        ValueError: If an unauthorized access to sensitive data is attempted,
                    if the specified student_token does not exist in the system,
                    or if the template is malformed or contains invalid placeholders.
    """
    if student_token not in students:
        raise ValueError(f"Student token '{student_token}' does not exist in the system.")

    student = students[student_token]

    # Validate the template for any unauthorized attribute access attempts
    formatter = Formatter()
    for _, field_name, _, _ in formatter.parse(report_template):
        if field_name:
            # Split placeholder like "student.name" into "student" and "name"
            parts = field_name.split('.', 1)

            if len(parts) == 2 and parts[0] == "student":
                attribute_name = parts[1]
                # Disallow access to nested attributes like "student.address.street"
                if '.' in attribute_name:
                    raise ValueError(
                        f"Unauthorized access: Nested attributes like '{attribute_name}' "
                        "are not allowed in placeholders."
                    )
                # Check if the requested attribute is in the allowed list
                if attribute_name not in ALLOWED_ATTRIBUTES:
                    raise ValueError(
                        f"Unauthorized access to sensitive or disallowed student data: "
                        f"Attempted to access '{attribute_name}'."
                    )
            elif field_name == "student": # Handles direct placeholder like {student}
                raise ValueError(
                    "Direct placeholder '{student}' is not allowed. "
                    "Please use '{student.attribute}' format (e.g., '{student.name}')."
                )
            # else:
                # This placeholder is not 'student' or 'student.attribute' (e.g., '{other_data}').
                # These will be handled by str.format(). If 'other_data' is not provided
                # as a keyword argument to .format(), a KeyError will occur.
                # This function is primarily responsible for validating 'student' data access.
                pass

    # Create a safe view of the student object that only exposes allowed attributes
    safe_student_view = _SafeStudentView(student)

    try:
        # Format the report. The .format() method will use safe_student_view for any
        # {student.attribute} placeholders.
        report = report_template.format(student=safe_student_view)
    except AttributeError as e:
        # This catch block is a safeguard. The validation loop above should prevent
        # attempts to access disallowed attributes on _SafeStudentView.
        # If an AttributeError occurs here, it implies an attempt to access an attribute
        # not present on _SafeStudentView (e.g., a sensitive one like 'gpa').
        raise ValueError(
            f"Error formatting report: An attribute was accessed that is not allowed "
            f"or does not exist on the safe student view. Original error: {e}"
        )
    except KeyError as e:
        # This occurs if the template contains a placeholder (e.g., "{foo}")
        # for which no corresponding data was provided in .format() (and foo is not "student").
        raise ValueError(
            f"Error formatting report: The template placeholder {e} is missing required data."
        )
    except Exception as e:
        # Catch any other unexpected errors during string formatting.
        raise ValueError(f"An unexpected error occurred during report formatting: {e}")

    return report
