EDITABLE_FIELDS = {"name", "position", "department"}

def modify_staff_info(staff: dict, changes: dict, admin_status: bool) -> dict:
    """
    Modifies specific attributes of an employee's record.
    Non-admin users are restricted to updating only the fields defined in EDITABLE_FIELDS.

    Args:
        staff: dict, a dictionary portraying the current state of the employee's details.
        changes: dict, a dictionary containing key-value pairs for fields and their new data.
        admin_status: bool, a flag that indicates whether the person making the update is an admin or not.

    Returns:
        dict: An updated version of the employee dictionary, reflecting the valid changes.
    """
    updated_staff = staff.copy()  # Work on a copy to avoid modifying the original dict directly

    for key, value in changes.items():
        if admin_status:
            updated_staff[key] = value
        elif key in EDITABLE_FIELDS:
            updated_staff[key] = value
    
    return updated_staff
