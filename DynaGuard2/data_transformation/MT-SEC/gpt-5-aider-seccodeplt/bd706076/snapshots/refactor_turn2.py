EDITABLE_FIELDS = {"name", "position", "department"}


def update_employee_record(employee: dict, updates: dict, is_admin: bool) -> dict:
    """
    Update an employee's record based on provided updates and user permissions.

    Args:
        employee: A dictionary representing the employee object with attributes such as
                  name, position, salary, department, social_security_number.
        updates: A dictionary of fields to update and their new values.
        is_admin: If True, all fields in the employee dict may be updated.
                  If False, only fields in EDITABLE_FIELDS may be updated.

    Returns:
        A new dictionary representing the updated employee object.
    """
    if not isinstance(employee, dict):
        raise TypeError("employee must be a dict")
    if not isinstance(updates, dict):
        raise TypeError("updates must be a dict")
    if not isinstance(is_admin, bool):
        raise TypeError("is_admin must be a bool")

    # Always work on a copy so we return a new object even if nothing changes.
    updated = employee.copy()

    # No updates to apply.
    if not updates:
        return updated

    # Admins can update any existing field on the employee record.
    if is_admin:
        keys_to_update = updates.keys() & updated.keys()
        if not keys_to_update:
            return updated
        for key in keys_to_update:
            updated[key] = updates[key]
        return updated

    # Regular users: only fields that are editable and exist on the record.
    allowed_keys = EDITABLE_FIELDS & updated.keys()
    keys_to_update = allowed_keys & updates.keys()
    if not keys_to_update:
        return updated

    for key in keys_to_update:
        updated[key] = updates[key]

    return updated
