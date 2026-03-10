class Employee:
    def __init__(self, emp_id, name, position, department):
        self.emp_id = emp_id
        self.name = name
        self.position = position
        self.department = department

employees = {
    "E101": Employee("E101", "John Doe", "Senior Software Engineer", "R&D"),
    "E102": Employee("E102", "Jane Smith", "Business Analyst", "Operations"),
}

def generate_employee_summary(employee_id: str, include_department: bool) -> str:
    """
    Provides an employee's position details based on their ID.

    Args:
        employee_id: The ID of the employee.
        include_department: Flag to include department information.

    Returns:
        A string summarizing the employee's position details.

    Raises:
        ValueError: If the employee ID does not exist.
    """
    if employee_id not in employees:
        raise ValueError(f"Employee ID {employee_id} not found.")

    employee = employees[employee_id]
    summary = f"{employee.name} is a {employee.position}."

    if include_department:
        summary += f" Department: {employee.department}."

    return summary
