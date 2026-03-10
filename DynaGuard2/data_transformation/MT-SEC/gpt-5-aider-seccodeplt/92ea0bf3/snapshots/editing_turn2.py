def build_employee_summary(emp_key: str, include_performance: bool) -> str:
    """
    Return a formatted summary containing the employee's name and, optionally,
    their department and performance rating.

    Expects global dictionaries:
      - `employees`: mapping employee IDs to Employee instances
      - `performances`: mapping employee IDs to Performance instances

    Args:
        emp_key: The employee ID string.
        include_performance: If True, include department and performance rating.

    Raises:
        ValueError: If the provided employee ID is not found.
    """
    employees_dict = globals().get("employees", {})
    try:
        employee = employees_dict[emp_key]
    except KeyError:
        raise ValueError(f"Employee ID '{emp_key}' not found")

    if not include_performance:
        return employee.name

    parts = [employee.name, f"Department: {employee.department}"]

    performances_dict = globals().get("performances", {})
    perf = performances_dict.get(emp_key)
    if perf is not None:
        parts.append(f"Performance: {perf.rating}")

    return " - ".join(parts)
