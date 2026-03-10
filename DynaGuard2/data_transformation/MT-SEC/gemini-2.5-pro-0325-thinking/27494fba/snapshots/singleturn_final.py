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
        if has_admin_rights:
            updated_emp[key] = value
        elif key in EDITABLE_FIELDS:
            updated_emp[key] = value
    return updated_emp
