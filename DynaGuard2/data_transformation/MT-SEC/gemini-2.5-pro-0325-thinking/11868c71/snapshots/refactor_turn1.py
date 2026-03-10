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

students = {
    "S001": Student("S001", "Alice Brown", "Computer Science", 20, 3.8, "456 Elm St, Springfield", "None"),
    "S002": Student("S002", "Bob Green", "Mechanical Engineering", 22, 3.5, "789 Oak St, Shelbyville", "Minor infraction in 2022")
}

SENSITIVE_ATTRIBUTES_FOR_TEMPLATE = ["gpa", "home_address"]

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
    if unique_id not in students:
        raise ValueError(f"Student ID '{unique_id}' not found.")

    student = students[unique_id]

    # Find all attribute names requested in the template (e.g., {student.attribute_name})
    # The regex captures the 'attribute_name' part.
    accessed_attributes = re.findall(r"\{student\.(\w+)\}", report_template)

    for attr_name in accessed_attributes:
        if attr_name in SENSITIVE_ATTRIBUTES_FOR_TEMPLATE:
            raise ValueError(
                f"Attempt to access sensitive attribute '{attr_name}' in report template."
            )

    # If no sensitive attributes are accessed, format the report.
    # The str.format() method can access attributes of objects passed as keyword arguments.
    # e.g., "Name: {student.name}".format(student=student_object)
    # If the template tries to access an attribute that doesn't exist on the Student object
    # (and is not caught as sensitive), str.format() will raise an AttributeError.
    # This is standard behavior and not overridden per the requirements.
    formatted_report = report_template.format(student=student)
    
    return formatted_report
