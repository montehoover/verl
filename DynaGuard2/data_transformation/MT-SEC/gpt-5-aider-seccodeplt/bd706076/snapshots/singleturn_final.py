EDITABLE_FIELDS = {"name", "position", "department"}


def update_employee_record(employee: dict, updates: dict, is_admin: bool) -> dict:
    """
    Update an employee record based on input updates and user role.

    Parameters:
        employee (dict): The employee object with attributes like
                         name, position, salary, department, social_security_number.
        updates (dict): Fields to be updated with their new values.
        is_admin (bool): If True, all fields can be updated; otherwise, only fields in EDITABLE_FIELDS.

    Returns:
        dict: The updated employee object.
    """
    # Work on a copy to avoid mutating the original input
    updated = employee.copy()

    if not isinstance(updates, dict) or not updates:
        return updated

    if is_admin:
        # Admins can update any field (including adding new fields)
        for key, value in updates.items():
            updated[key] = value
        return updated

    # Regular users can only update fields listed in EDITABLE_FIELDS
    for key, value in updates.items():
        if key in EDITABLE_FIELDS:
            updated[key] = value

    return updated
