import logging
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


def _parse_template_and_validate_attributes(report_template: str, allowed_attrs: set[str]):
    """
    Parses the report template and validates that all requested student attributes
    are allowed and correctly formatted.

    Args:
        report_template: The format string for the report.
        allowed_attrs: A set of attribute names that are permitted for student data.

    Raises:
        ValueError: If the template attempts to access unauthorized attributes,
                    uses disallowed formats (e.g., nested attributes, direct {student} placeholder).
    """
    formatter = Formatter()
    for _, field_name, _, _ in formatter.parse(report_template):
        if field_name:
            parts = field_name.split('.', 1)

            if len(parts) == 2 and parts[0] == "student":
                attribute_name = parts[1]
                if '.' in attribute_name:
                    raise ValueError(
                        f"Unauthorized access: Nested attributes like '{attribute_name}' "
                        "are not allowed in placeholders."
                    )
                if attribute_name not in allowed_attrs:
                    raise ValueError(
                        f"Unauthorized access to sensitive or disallowed student data: "
                        f"Attempted to access '{attribute_name}'."
                    )
            elif field_name == "student":
                raise ValueError(
                    "Direct placeholder '{student}' is not allowed. "
                    "Please use '{student.attribute}' format (e.g., '{student.name}')."
                )
            # Other placeholders (not starting with 'student.') are allowed but will
            # require corresponding data in the .format() call, which is handled
            # by _generate_report_from_template.


def _generate_report_from_template(report_template: str, student_data_view: _SafeStudentView) -> str:
    """
    Generates the report string by formatting the template with the provided student data view.

    Args:
        report_template: The format string for the report.
        student_data_view: An instance of _SafeStudentView providing allowed student attributes.
                           Accessed via a 'student' key in the template (e.g., {student.name}).

    Returns:
        The formatted report string.

    Raises:
        ValueError: If an error occurs during formatting (e.g., attribute not found on
                    student_data_view, or other template placeholders are missing data).
    """
    try:
        # Format the report. The .format() method will use student_data_view for any
        # {student.attribute} placeholders.
        report = report_template.format(student=student_data_view)
    except AttributeError as e:
        # This catch block is a safeguard. Validation in _parse_template_and_validate_attributes
        # and the nature of _SafeStudentView should prevent attempts to access disallowed attributes.
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
    # Initialize basic logging configuration if not already configured
    # In a real application, logging is typically configured once at startup.
    # For this specific request, initializing within the function.
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logger = logging.getLogger(__name__)

    logger.info(f"Attempting to generate report for student token: '{student_token}'")
    logger.info(f"Using template: \"{report_template}\"")

    try:
        if student_token not in students:
            logger.error(f"Student token '{student_token}' does not exist.")
            raise ValueError(f"Student token '{student_token}' does not exist in the system.")

        student = students[student_token]

        # Validate the template for allowed attributes and format
        _parse_template_and_validate_attributes(report_template, ALLOWED_ATTRIBUTES)
        logger.info("Template validation successful.")

        # Create a safe view of the student object that only exposes allowed attributes
        safe_student_view = _SafeStudentView(student)
        logger.info("Safe student view created.")

        # Generate the report using the validated template and safe data view
        report = _generate_report_from_template(report_template, safe_student_view)
        logger.info(f"Report generated successfully for student token: '{student_token}'.")
        logger.debug(f"Generated report content: \"{report}\"") # Log full report at DEBUG

        return report
    except ValueError as ve:
        logger.error(f"ValueError during report generation for token '{student_token}': {ve}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during report generation for token '{student_token}': {e}", exc_info=True)
        raise
