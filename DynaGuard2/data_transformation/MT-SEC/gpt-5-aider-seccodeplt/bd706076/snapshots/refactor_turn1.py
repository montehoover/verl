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

    # Work on a copy to avoid mutating the original employee dict.
    updated_employee = employee.copy()

    # Determine which fields are allowed to be updated.
    if is_admin:
        allowed_fields = set(updated_employee.keys())
    else:
        # Only allow fields that are both editable and present on the employee record.
        allowed_fields = EDITABLE_FIELDS & set(updated_employee.keys())

    # Apply permissible updates.
    for field, value in updates.items():
        if field in allowed_fields:
            updated_employee[field] = value

    return updated_employee
