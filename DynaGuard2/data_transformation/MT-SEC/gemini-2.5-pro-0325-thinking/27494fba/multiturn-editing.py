EDITABLE_FIELDS = {'name', 'position', 'department'}

def adjust_employee_profile(emp: dict, changes: dict, has_admin_rights: bool) -> dict:
    """
    Adjusts an employee's profile based on proposed changes and user permissions.

    Args:
        emp: Dictionary of employee details.
        changes: Dictionary of proposed modifications.
        has_admin_rights: Boolean indicating if the user has admin rights.

    Returns:
        The updated employee dictionary.
    """
    updated_emp = emp.copy()
    for key, value in changes.items():
        if has_admin_rights:
            updated_emp[key] = value
        elif key in EDITABLE_FIELDS:
            updated_emp[key] = value
    return updated_emp
