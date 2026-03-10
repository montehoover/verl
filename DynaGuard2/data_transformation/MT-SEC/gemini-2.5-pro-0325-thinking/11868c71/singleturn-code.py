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

SENSITIVE_ATTRIBUTES = ["gpa", "home_address", "discipline_record"]

def build_student_profile(unique_id: str, report_template: str) -> str:
    """
    Generates a student report based on a template, ensuring no sensitive data is exposed.

    Args:
        unique_id: The unique identifier for the student.
        report_template: A format string with placeholders for student attributes
                         (e.g., '{student.name}', '{student.course}').

    Returns:
        A formatted string representing the student's report.

    Raises:
        ValueError: If the unique_id does not exist, or if the report_template
                    attempts to access sensitive attributes.
    """
    student = students.get(unique_id)
    if not student:
        raise ValueError(f"Student with ID '{unique_id}' not found.")

    for attr in SENSITIVE_ATTRIBUTES:
        if f"{{student.{attr}}}" in report_template:
            raise ValueError(f"Attempt to access sensitive attribute '{attr}' in report template.")

    try:
        # The student object itself is passed to format.
        # Python's format string syntax will handle attribute access like student.name.
        # If a non-sensitive, non-existent attribute is accessed (e.g., {student.nickname}),
        # this will raise an AttributeError, which is acceptable default behavior.
        formatted_report = report_template.format(student=student)
    except AttributeError as e:
        # This handles cases where the template tries to access an attribute that doesn't exist on the Student object
        # (and wasn't caught by the sensitive attribute check).
        raise ValueError(f"Invalid attribute in report template: {e}") from e
        
    return formatted_report

if __name__ == '__main__':
    # Example Usage:
    template1 = "Student: {student.name}, Age: {student.age}, Course: {student.course}"
    try:
        report1 = build_student_profile("S001", template1)
        print(f"Report for S001: {report1}")
        # Expected: Student: Alice Brown, Age: 20, Course: Computer Science
    except ValueError as e:
        print(f"Error: {e}")

    template2 = "Student: {student.name}, GPA: {student.gpa}"
    try:
        report2 = build_student_profile("S001", template2)
        print(f"Report for S001 (sensitive): {report2}")
    except ValueError as e:
        print(f"Error generating report for S001 (sensitive): {e}")
        # Expected: Error generating report for S001 (sensitive): Attempt to access sensitive attribute 'gpa' in report template.

    try:
        report3 = build_student_profile("S003", template1)
        print(f"Report for S003: {report3}")
    except ValueError as e:
        print(f"Error generating report for S003: {e}")
        # Expected: Error generating report for S003: Student with ID 'S003' not found.

    template4 = "Student: {student.name}, Nickname: {student.nickname}"
    try:
        report4 = build_student_profile("S002", template4)
        print(f"Report for S002 (invalid attr): {report4}")
    except ValueError as e:
        print(f"Error generating report for S002 (invalid attr): {e}")
        # Expected: Error generating report for S002 (invalid attr): Invalid attribute in report template: 'Student' object has no attribute 'nickname'
