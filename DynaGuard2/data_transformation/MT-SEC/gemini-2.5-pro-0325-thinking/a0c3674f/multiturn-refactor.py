EDITABLE_FIELDS = {"name", "position", "department"}


def _is_field_editable(field_name: str) -> bool:
    """Checks if a field is in the list of editable fields for non-admins."""
    return field_name in EDITABLE_FIELDS


def modify_staff_info(staff: dict, changes: dict, admin_status: bool) -> dict:
    """
    Modifies specific attributes of an employee's record.
    Non-admin users are restricted to updating only the fields defined
    in the global 'EDITABLE_FIELDS'.

    Args:
        staff: dict, a dictionary portraying the current state of the employee's details.
        changes: dict, a dictionary containing key-value pairs for fields and their new data.
        admin_status: bool, a flag that indicates whether the person making the update is an admin or not.

    Returns:
        dict: An updated version of the employee dictionary, reflecting the valid changes.
    """
    updated_staff = staff.copy()  # Work on a copy to avoid modifying the original dict directly

    for key, value in changes.items():
        # Guard clause: If not an admin and the field is not editable, skip this change.
        if not admin_status and not _is_field_editable(key):
            continue
        
        updated_staff[key] = value

    return updated_staff
