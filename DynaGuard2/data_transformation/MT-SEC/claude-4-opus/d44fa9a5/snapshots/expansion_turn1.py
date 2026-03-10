class Employee:
    def __init__(self, id, name, department, position):
        self.id = id
        self.name = name
        self.department = department
        self.position = position

# Predefined dictionary of employees
employees = {
    1001: Employee(1001, "John Smith", "Engineering", "Senior Developer"),
    1002: Employee(1002, "Jane Doe", "Marketing", "Marketing Manager"),
    1003: Employee(1003, "Bob Johnson", "Sales", "Sales Representative"),
    1004: Employee(1004, "Alice Williams", "HR", "HR Specialist"),
    1005: Employee(1005, "Charlie Brown", "Engineering", "Junior Developer")
}

def get_employee_by_id(worker_id):
    if worker_id not in employees:
        raise ValueError(f"Employee with ID {worker_id} does not exist")
    return employees[worker_id]
