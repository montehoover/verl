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
    if not changes:
        return dict(emp)

    # Admins: apply all changes and return early
    if has_admin_rights:
        updated = dict(emp)
        updated.update(changes)
        return updated

    # Regular users: filter to allowed fields; if nothing allowed, return early
    allowed_changes = {k: v for k, v in changes.items() if k in EDITABLE_FIELDS}
    if not allowed_changes:
        return dict(emp)

    updated = dict(emp)
    updated.update(allowed_changes)
    return updated
