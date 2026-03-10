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
        # Guard clause: If not a superuser and the field is not editable, skip this change.
        if not is_superuser and key not in EDITABLE_FIELDS:
            # Optionally, you might want to log or raise an error for unauthorized changes
            # print(f"Warning: User not authorized to change field '{key}'. Skipping.")
            continue

        # If the guard clause didn't trigger, it's safe to update the field.
        updated_employee_data[key] = value

    return updated_employee_data
