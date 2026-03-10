EDITABLE_FIELDS = {"name", "position", "department"}

def edit_personnel_info(employee_record: dict, adjustments: dict, admin_flag: bool) -> dict:
    """
    Update an employee record with the given adjustments, respecting edit permissions.

    - Admins (admin_flag=True) can modify any existing field.
    - Non-admins (admin_flag=False) can only modify fields in EDITABLE_FIELDS.
    - Fields not present in the initial record will not be added.

    Args:
        employee_record: The original employee dictionary (e.g., includes name, position, salary, department, social_security_number).
        adjustments: A dictionary of proposed changes.
        admin_flag: Whether the editor has admin privileges.

    Returns:
        A new dictionary representing the updated employee record with permissible changes applied.
    """
    if not isinstance(employee_record, dict):
        raise TypeError("employee_record must be a dict")
    if not isinstance(adjustments, dict):
        raise TypeError("adjustments must be a dict")
    if not isinstance(admin_flag, bool):
        raise TypeError("admin_flag must be a bool")

    updated = dict(employee_record)  # shallow copy; do not mutate original

    if admin_flag:
        keys_to_update = adjustments.keys()
    else:
        keys_to_update = set(adjustments.keys()).intersection(EDITABLE_FIELDS)

    for key in keys_to_update:
        if key in updated:
            updated[key] = adjustments[key]

    return updated
