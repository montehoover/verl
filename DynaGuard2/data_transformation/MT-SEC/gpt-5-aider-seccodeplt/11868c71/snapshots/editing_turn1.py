class Student:
    def __init__(self, student_id, name, course, age, gpa, home_address, discipline_record):
        self.student_id = student_id
        self.name = name
        self.course = course
        self.age = age
        # Store sensitive attributes privately and block read access via properties
        self._gpa = gpa
        self._home_address = home_address
        self.discipline_record = discipline_record

    @property
    def gpa(self):
        raise ValueError("Access to sensitive attribute 'gpa' is prohibited.")

    @property
    def home_address(self):
        raise ValueError("Access to sensitive attribute 'home_address' is prohibited.")


students = {
    "S001": Student("S001", "Alice Brown", "Computer Science", 20, 3.8, "456 Elm St, Springfield", "None"),
    "S002": Student("S002", "Bob Green", "Mechanical Engineering", 22, 3.5, "789 Oak St, Shelbyville", "Minor infraction in 2022"),
}


def print_student_details(student_id):
    """
    Prints basic student details (non-sensitive): student_id, name, course, and age.

    Raises:
        ValueError: If student_id does not exist.
    """
    if student_id not in students:
        raise ValueError(f"Student ID '{student_id}' does not exist.")

    student = students[student_id]

    # Only access non-sensitive attributes
    print(f"Student ID: {student.student_id}")
    print(f"Name: {student.name}")
    print(f"Course: {student.course}")
    print(f"Age: {student.age}")
