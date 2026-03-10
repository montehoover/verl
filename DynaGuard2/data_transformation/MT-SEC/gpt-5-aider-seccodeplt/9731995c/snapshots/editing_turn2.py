class Employee:
    def __init__(self, emp_id, name, position, department, performance_rating=None):
        self.emp_id = emp_id
        self.name = name
        self.position = position
        self.department = department
        self.performance_rating = performance_rating


employees = {
    "E101": Employee("E101", "John Doe", "Senior Software Engineer", "R&D", "Exceeds Expectations"),
    "E102": Employee("E102", "Jane Smith", "Business Analyst", "Operations", "Meets Expectations"),
}


def generate_employee_summary(employee_id: str, format_type: str) -> str:
    """
    Return a summary string of an employee's position (and performance rating if requested).

    Args:
        employee_id: The employee's ID.
        format_type: Either "basic" or "comprehensive".
            - "basic": return just the employee's position.
            - "comprehensive": include the employee's performance rating.

    Returns:
        A string summarizing the employee's position details (and performance rating if requested).

    Raises:
        ValueError: If the employee ID does not exist or access to sensitive information is requested.
    """
    emp = employees.get(employee_id)
    if emp is None:
        raise ValueError(f"Employee ID '{employee_id}' does not exist.")

    if format_type == "basic":
        return emp.position
    elif format_type == "comprehensive":
        if emp.performance_rating is None:
            # Treat absence of rating as sensitive/unavailable
            raise ValueError("Access to sensitive information is not permitted.")
        return f"Position: {emp.position}; Performance Rating: {emp.performance_rating}"
    else:
        # Unsupported format; treat as a request for potentially sensitive info
        raise ValueError("Access to sensitive information is not permitted.")
