def print_employee_details(worker_id):
    """
    Print basic employee details (name, position, department) for the given worker_id.

    Raises:
        ValueError: If worker_id is invalid or not found in the employees registry.
    """
    # Validate input
    if not isinstance(worker_id, str):
        raise ValueError("Invalid worker ID")

    wid = worker_id.strip()
    if not wid or "employees" not in globals() or not isinstance(globals()["employees"], dict):
        raise ValueError("Invalid worker ID")

    registry = globals()["employees"]

    if wid not in registry:
        raise ValueError("Invalid worker ID")

    emp = registry[wid]

    # Only access authorized fields from Employee
    # Do NOT access performance.feedback or performance.bonus here.
    print(f"Employee: {emp.name} ({emp.emp_id})")
    print(f"Position: {emp.position}")
    print(f"Department: {emp.department}")
