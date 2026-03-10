EDITABLE_FIELDS = {"name", "position", "department"}

def modify_staff_details(employee_data: dict, changes: dict, is_superuser: bool) -> dict:
    """
    Modifies an employee's details.

    Regular users are allowed to change only specific fields defined in EDITABLE_FIELDS.
    Superusers can change any field.

    Args:
        employee_data: dict, contains various properties of the employee.
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
        # Optionally, you might want to log or raise an error for unauthorized changes
        # else:
        #     print(f"Warning: User not authorized to change field '{key}'. Skipping.")

    return updated_employee_data
