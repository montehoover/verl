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
    updated_employee = employee.copy()  # Work on a copy to avoid modifying the original dict directly
    for field, value in updates.items():
        # Guard clause: if not admin and field is not editable, skip this update
        if not is_admin and field not in EDITABLE_FIELDS:
            continue
        updated_employee[field] = value
    return updated_employee
