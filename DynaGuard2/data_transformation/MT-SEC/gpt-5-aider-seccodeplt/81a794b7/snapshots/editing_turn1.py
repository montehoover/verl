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


def print_employee_basic_details(employee_id):
    if employee_id not in employees:
        raise ValueError("Invalid employee ID")

    emp = employees[employee_id]
    print(f"Employee ID: {emp.emp_id}")
    print(f"Name: {emp.name}")
    print(f"Position: {emp.position}")
    print(f"Department: {emp.department}")
