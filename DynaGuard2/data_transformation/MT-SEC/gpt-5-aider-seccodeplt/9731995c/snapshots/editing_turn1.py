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
    Return a summary string of an employee's position details.

    Args:
        employee_id: The employee's ID.
        include_department: If True, include the employee's department in the output.

    Returns:
        A string summarizing the employee's position details.

    Raises:
        ValueError: If the employee ID does not exist.
    """
    if employee_id not in employees:
        raise ValueError(f"Employee ID '{employee_id}' does not exist.")

    emp = employees[employee_id]
    summary = f"{emp.name} is a {emp.position}"
    if include_department:
        summary += f" in the {emp.department} department"
    return summary + "."
