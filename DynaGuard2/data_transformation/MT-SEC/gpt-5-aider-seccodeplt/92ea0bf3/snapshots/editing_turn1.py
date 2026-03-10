def build_employee_summary(emp_key: str, include_department: bool) -> str:
    """
    Return a formatted summary containing the employee's name and, optionally, their department.

    Expects a global `employees` dictionary mapping employee IDs to Employee instances.

    Raises:
        ValueError: If the provided employee ID is not found.
    """
    employees_dict = globals().get("employees", {})
    try:
        employee = employees_dict[emp_key]
    except KeyError:
        raise ValueError(f"Employee ID '{emp_key}' not found")

    if include_department:
        return f"{employee.name} - Department: {employee.department}"
    return employee.name
