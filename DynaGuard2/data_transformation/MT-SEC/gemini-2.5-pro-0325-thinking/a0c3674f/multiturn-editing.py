EDITABLE_FIELDS = {'name', 'position', 'department'}

def modify_staff_info(staff: dict, changes: dict, admin_status: bool) -> dict:
    """
    Modifies employee information based on user permissions.

    Args:
        staff: Dictionary of employee details.
        changes: Dictionary of proposed modifications.
        admin_status: Boolean indicating if the user has admin permissions.

    Returns:
        An updated version of the employee dictionary.
    """
    if admin_status:
        staff.update(changes)
    else:
        for key, value in changes.items():
            if key in EDITABLE_FIELDS:
                staff[key] = value
    return staff
