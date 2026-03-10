EDITABLE_FIELDS = {"name", "position", "department"}

def update_employee_record(employee: dict, updates: dict, is_admin: bool) -> dict:
    """
    Updates an employee's record based on input data.

    Args:
        employee: A dictionary representing the employee object.
        updates: A dictionary containing the fields to be updated and their new values.
        is_admin: A flag indicating whether the update is being performed by an admin user.

    Returns:
        A dictionary representing the updated employee object.
    """
    for key, value in updates.items():
        if is_admin:
            employee[key] = value
        elif key in EDITABLE_FIELDS:
            employee[key] = value
    return employee
