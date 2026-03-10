EDITABLE_FIELDS = {"name", "position", "department"}

def modify_staff_details(employee_data: dict, changes: dict, is_superuser: bool) -> dict:
    """
    Modifies an employee's details.

    Regular users are allowed to change only specific fields predefined
    in the global 'EDITABLE_FIELDS' set.

    Args:
        employee_data: dict, which contains various properties of the employee.
        changes: dict, detailing the proposed modifications to that employee's attributes.
        is_superuser: bool, determines whether the user carrying out the update is an admin or not.

    Returns:
        The updated employee dictionary.
    """
    updated_employee_data = employee_data.copy()

    for key, value in changes.items():
        if is_superuser:
            updated_employee_data[key] = value
        elif key in EDITABLE_FIELDS:
            updated_employee_data[key] = value
        # If not a superuser and key is not in EDITABLE_FIELDS, the change is ignored.

    return updated_employee_data
