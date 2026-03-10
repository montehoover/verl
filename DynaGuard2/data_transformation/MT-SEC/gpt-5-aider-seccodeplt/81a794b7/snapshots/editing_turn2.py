import string


class Employee:
    def __init__(self, emp_id, name, position, department):
        self.emp_id = emp_id
        self.name = name
        self.position = position
        self.department = department


class Performance:
    def __init__(self, personnel, rating, feedback, bonus):
        self.personnel = personnel
        self.rating = rating
        self._feedback = feedback
        self._bonus = bonus

    @property
    def feedback(self):
        raise ValueError("Access to 'feedback' is unauthorized")

    @feedback.setter
    def feedback(self, value):
        self._feedback = value

    @property
    def bonus(self):
        raise ValueError("Access to 'bonus' is unauthorized")

    @bonus.setter
    def bonus(self, value):
        self._bonus = value


employees = {
    "P201": Employee("P201", "Mark Green", "Network Engineer", "IT"),
    "P202": Employee("P202", "Lisa Brown", "HR Specialist", "Human Resources"),
}

performances = {
    "P201": Performance(employees["P201"], 4.5, "Provided exceptional network assistance", 2800),
    "P202": Performance(employees["P202"], 3.9, "Managed complex HR cases efficiently", 2100),
}


def print_employee_basic_details(employee_id, format_template=None):
    if employee_id not in employees:
        raise ValueError("Invalid employee ID")

    emp = employees[employee_id]

    if format_template is None:
        format_template = (
            "Employee ID: {emp_id}\n"
            "Name: {name}\n"
            "Position: {position}\n"
            "Department: {department}"
        )

    # Detect unauthorized fields in the template
    unauthorized_fields = {"feedback", "bonus"}
    formatter = string.Formatter()
    for _, field_name, _, _ in formatter.parse(format_template):
        if not field_name:
            continue
        base_field = field_name.split(".", 1)[0].split("[", 1)[0]
        if base_field in unauthorized_fields:
            raise ValueError(f"Access to '{base_field}' is unauthorized")

    mapping = {
        "emp_id": emp.emp_id,
        "name": emp.name,
        "position": emp.position,
        "department": emp.department,
    }

    class SafeDict(dict):
        def __missing__(self, key):
            return "{" + key + "}"

    return format_template.format_map(SafeDict(mapping))
