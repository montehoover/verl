EDITABLE_FIELDS = {"name", "position", "department"}


def adjust_employee_profile(emp: dict, changes: dict, has_admin_rights: bool) -> dict:
    """
    Modify an employee's details with role-based field restrictions.

    - Admins (has_admin_rights=True) can update any fields provided in `changes`.
    - Regular users can only update fields listed in the global EDITABLE_FIELDS set.

    Args:
        emp: Existing employee dictionary (e.g., name, position, salary, department, social_security_number).
        changes: Proposed modifications to apply to `emp`.
        has_admin_rights: Whether the updater has admin privileges.

    Returns:
        A new dictionary representing the updated employee.
    """
    if not isinstance(emp, dict):
        raise TypeError("emp must be a dict")
    if not isinstance(changes, dict):
        raise TypeError("changes must be a dict")
    if not isinstance(has_admin_rights, bool):
        raise TypeError("has_admin_rights must be a bool")

    # Create a shallow copy to avoid mutating the original employee dict
    updated = dict(emp)

    if has_admin_rights:
        # Admins can update any provided fields
        for key, value in changes.items():
            updated[key] = value
    else:
        # Regular users can only update allowed fields
        for key, value in changes.items():
            if key in EDITABLE_FIELDS:
                updated[key] = value
        # Silently ignore disallowed fields for regular users

    return updated
