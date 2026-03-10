class Employee:
    def __init__(self, emp_id, name, position, department, performance_rating):
        self.emp_id = emp_id
        self.name = name
        self.position = position
        self.department = department
        self.performance_rating = performance_rating

employees = {
    "E101": Employee("E101", "John Doe", "Senior Software Engineer", "R&D", "Excellent"),
    "E102": Employee("E102", "Jane Smith", "Business Analyst", "Operations", "Good"),
}

def generate_employee_summary(employee_id: str, format_type: str) -> str:
    """
    Provides an employee's position and/or performance details based on their ID and format type.

    Args:
        employee_id: The ID of the employee.
        format_type: Specifies the level of detail: "basic" (position only)
                     or "comprehensive" (position and performance rating).

    Returns:
        A string summarizing the employee's details based on the format_type.

    Raises:
        ValueError: If the employee ID does not exist, or if format_type is invalid,
                    or if attempting to access sensitive information without authorization (not yet implemented).
    """
    if employee_id not in employees:
        raise ValueError(f"Employee ID {employee_id} not found.")

    employee = employees[employee_id]

    if format_type == "basic":
        return f"{employee.name} is a {employee.position}."
    elif format_type == "comprehensive":
        # In a real system, you might check authorization here before revealing performance_rating
        return f"{employee.name} is a {employee.position}. Performance Rating: {employee.performance_rating}."
    else:
        raise ValueError(f"Invalid format_type: '{format_type}'. Must be 'basic' or 'comprehensive'.")
