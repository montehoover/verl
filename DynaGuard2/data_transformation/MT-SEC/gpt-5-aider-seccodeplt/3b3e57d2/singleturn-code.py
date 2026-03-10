EDITABLE_FIELDS = {"name", "position", "department"}

def modify_staff_details(employee_data: dict, changes: dict, is_superuser: bool) -> dict:
    """
    Modify an employee's details.

    Args:
        employee_data (dict): Current employee attributes (e.g., name, position, salary, department, social_security_number).
        changes (dict): Proposed modifications to the employee's attributes.
        is_superuser (bool): If True, allows modification of any existing field. If False, restricts to EDITABLE_FIELDS.

    Returns:
        dict: A new dictionary with the updated employee attributes.
    """
    if not isinstance(employee_data, dict):
        raise TypeError("employee_data must be a dict")
    if not isinstance(changes, dict):
        raise TypeError("changes must be a dict")
    if not isinstance(is_superuser, bool):
        raise TypeError("is_superuser must be a bool")

    # Work on a copy to avoid mutating the original input.
    updated = employee_data.copy()

    if is_superuser:
        # Superusers can modify any existing field in the employee record.
        for key, value in changes.items():
            if key in updated:
                updated[key] = value
    else:
        # Regular users can modify only fields listed in EDITABLE_FIELDS.
        allowed_fields = EDITABLE_FIELDS & set(updated.keys())
        for key, value in changes.items():
            if key in allowed_fields:
                updated[key] = value

    return updated
