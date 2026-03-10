EDITABLE_FIELDS = {"name", "position", "department"}

def modify_staff_info(staff: dict, changes: dict, admin_status: bool) -> dict:
    """
    Modify an employee's information with respect to admin permissions.

    Args:
        staff (dict): Current employee information.
        changes (dict): Proposed updates as key-value pairs.
        admin_status (bool): True if the requester is an admin; otherwise False.

    Returns:
        dict: A new dictionary with valid updates applied.
    """
    if not isinstance(staff, dict):
        raise TypeError("staff must be a dict")
    if not isinstance(changes, dict):
        raise TypeError("changes must be a dict")

    # Filter changes for non-admin users to only the allowed fields.
    if admin_status:
        allowed_changes = changes
    else:
        allowed_changes = {k: v for k, v in changes.items() if k in EDITABLE_FIELDS}

    # Return a new dict reflecting valid changes.
    updated_staff = staff.copy()
    if allowed_changes:
        updated_staff.update(allowed_changes)

    return updated_staff
