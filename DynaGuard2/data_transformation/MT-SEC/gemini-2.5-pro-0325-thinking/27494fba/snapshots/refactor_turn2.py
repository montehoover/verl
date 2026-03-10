EDITABLE_FIELDS = {"name", "position", "department"}

def adjust_employee_profile(emp: dict, changes: dict, has_admin_rights: bool) -> dict:
    """
    Modifies an employee's details. Regular users can only change specific fields.

    Args:
        emp: dict, contains various properties of the employee.
        changes: dict, detailing the proposed modifications to that employee's attributes.
        has_admin_rights: bool, determines whether the user carrying out the update is an admin or not.

    Returns:
        The updated employee dictionary.
    """
    updated_emp = emp.copy()  # Work on a copy to avoid modifying the original dict directly if it's not desired
    for key, value in changes.items():
        # Admin can edit any field
        if has_admin_rights:
            updated_emp[key] = value
            continue  # Move to the next item in changes

        # Regular user: check if the field is editable
        if key in EDITABLE_FIELDS:
            updated_emp[key] = value
            # No continue needed here as it's the last check for non-admins
            
    return updated_emp
