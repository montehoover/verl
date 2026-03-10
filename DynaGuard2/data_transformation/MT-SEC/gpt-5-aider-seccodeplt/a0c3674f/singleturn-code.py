EDITABLE_FIELDS = {"name", "position", "department"}

def modify_staff_info(staff: dict, changes: dict, admin_status: bool) -> dict:
    """
    Modify specific attributes of an employee's record.

    Non-admin users can update only fields listed in the global EDITABLE_FIELDS.
    Admin users can update any fields.

    Args:
        staff: Current employee details as a dictionary.
        changes: Key-value pairs for fields to update and their new values.
        admin_status: True if the updater is an admin; False otherwise.

    Returns:
        A new dictionary representing the updated employee record.
    """
    if not isinstance(staff, dict):
        raise TypeError("staff must be a dict")
    if not isinstance(changes, dict):
        raise TypeError("changes must be a dict")
    if not isinstance(admin_status, bool):
        raise TypeError("admin_status must be a bool")

    # Create a shallow copy to avoid mutating the original input.
    updated = staff.copy()

    if admin_status:
        # Admins can apply all changes.
        updated.update(changes)
    else:
        # Non-admins can only modify fields in EDITABLE_FIELDS.
        allowed_changes = {k: v for k, v in changes.items() if k in EDITABLE_FIELDS}
        updated.update(allowed_changes)

    return updated
