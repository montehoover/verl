import re
import logging

# Configure basic logging
# In a real application, this might be configured globally.
logger = logging.getLogger(__name__)
# Example: Set to INFO level to see the logs.
# logging.basicConfig(level=logging.INFO)


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

SENSITIVE_ATTRIBUTES_FOR_TEMPLATE = ["gpa", "home_address"]


def _validate_template_for_sensitive_data(report_template: str, sensitive_attributes_list: list[str]) -> None:
    """
    Validates the report template against a list of sensitive attributes.

    Args:
        report_template: The template string to check.
        sensitive_attributes_list: A list of attribute names considered sensitive.

    Raises:
        ValueError: If the template attempts to access a sensitive attribute.
    """
    # Find all attribute names requested in the template (e.g., {student.attribute_name})
    accessed_attributes = re.findall(r"\{student\.(\w+)\}", report_template)

    for attr_name in accessed_attributes:
        if attr_name in sensitive_attributes_list:
            raise ValueError(
                f"Attempt to access sensitive attribute '{attr_name}' in report template."
            )


def _format_report(student: Student, report_template: str) -> str:
    """
    Formats the student report using the provided student data and template.

    Args:
        student: The Student object containing the data.
        report_template: The template string.

    Returns:
        The formatted report string.
    """
    # The str.format() method can access attributes of objects passed as keyword arguments.
    # e.g., "Name: {student.name}".format(student=student_object)
    # If the template tries to access an attribute that doesn't exist on the Student object
    # (and is not caught as sensitive), str.format() will raise an AttributeError.
    return report_template.format(student=student)


def build_student_profile(unique_id: str, report_template: str) -> str:
    """
    Generates a student report based on a template, ensuring no sensitive data is exposed.

    Information such as the student’s name, course, and age will be presented
    in a customizable report format. The provided template can include
    placeholders like '{student.name}' or '{student.course}'.

    Args:
        unique_id: A unique string representing the student ID.
        report_template: A format string which contains placeholders for
                         generating the report.

    Returns:
        A formatted string representing the student's report, generated
        using the provided template.

    Raises:
        ValueError: If the provided unique_id does not exist, or if there's
                    an attempt to access sensitive attributes (e.g., GPA,
                    home_address) via the template.
    """
    logger.info(
        f"Attempting to build student profile for ID: '{unique_id}' with template: '{report_template}'"
    )
    try:
        if unique_id not in students:
            logger.error(f"Student ID '{unique_id}' not found.")
            raise ValueError(f"Student ID '{unique_id}' not found.")

        student = students[unique_id]

        # Validate the template for sensitive attributes
        _validate_template_for_sensitive_data(report_template, SENSITIVE_ATTRIBUTES_FOR_TEMPLATE)
        logger.debug(f"Template validation successful for ID: '{unique_id}'.")

        # Format the report
        formatted_report = _format_report(student, report_template)
        logger.info(
            f"Successfully generated report for ID: '{unique_id}'. Report: '{formatted_report}'"
        )
        
        return formatted_report
    except ValueError as e:
        logger.error(f"Error building student profile for ID '{unique_id}': {e}")
        raise # Re-raise the caught ValueError
    except Exception as e:
        logger.exception(f"An unexpected error occurred while building profile for ID '{unique_id}'.")
        raise # Re-raise any other unexpected exception
