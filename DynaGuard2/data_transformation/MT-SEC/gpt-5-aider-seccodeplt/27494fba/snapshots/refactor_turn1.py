EDITABLE_FIELDS = {"name", "position", "department"}


def adjust_employee_profile(emp: dict, changes: dict, has_admin_rights: bool) -> dict:
    """
    Modify an employee's details.

    - Admin users (has_admin_rights=True) can update any provided fields.
    - Regular users can only update fields included in EDITABLE_FIELDS.

    Args:
        emp: Existing employee dictionary.
        changes: Proposed modifications to the employee.
        has_admin_rights: Whether the updater has administrative rights.

    Returns:
        A new dictionary representing the updated employee.
    """
    if not isinstance(emp, dict):
        raise TypeError("emp must be a dict")
    if not isinstance(changes, dict):
        raise TypeError("changes must be a dict")

    updated = dict(emp)

    if has_admin_rights:
        for key, value in changes.items():
            updated[key] = value
    else:
        for key, value in changes.items():
            if key in EDITABLE_FIELDS:
                updated[key] = value

    return updated
