EDITABLE_FIELDS = {"name", "position", "department"}

def _validate_changes(changes: dict, is_superuser: bool, allowed_fields: set) -> dict:
    """
    Validates the changes based on user privileges.

    Args:
        changes: dict, a dictionary containing key-value pairs for fields and their new data.
        is_superuser: bool, a flag that indicates whether the person making the update is an admin or not.
        allowed_fields: set, a set of fields that non-admin users are allowed to edit.

    Returns:
        dict: A dictionary containing only the valid changes.
    """
    validated_changes = {}
    for field, new_value in changes.items():
        if is_superuser or field in allowed_fields:
            validated_changes[field] = new_value
    return validated_changes

def _apply_changes(emp_data: dict, validated_changes: dict) -> dict:
    """
    Applies the validated changes to the employee data.

    Args:
        emp_data: dict, the current employee data.
        validated_changes: dict, a dictionary of validated changes to apply.

    Returns:
        dict: An updated version of the employee dictionary.
    """
    updated_emp_data = emp_data.copy()
    for field, new_value in validated_changes.items():
        updated_emp_data[field] = new_value
    return updated_emp_data

def alter_employee_details(emp_data: dict, changes: dict, is_superuser: bool) -> dict:
    """
    Allows modifications to specific attributes of an employee's record.
    Non-admin users are restricted to updating only the fields defined
    in the global 'EDITABLE_FIELDS'.

    Args:
        emp_data: dict, a dictionary portraying the current state of the employee's details.
        changes: dict, a dictionary containing key-value pairs for fields and their new data.
        is_superuser: bool, a flag that indicates whether the person making the update is an admin or not.

    Returns:
        dict: An updated version of the employee dictionary, reflecting the valid changes.
    """
    validated_changes = _validate_changes(changes, is_superuser, EDITABLE_FIELDS)
    updated_emp_data = _apply_changes(emp_data, validated_changes)
    return updated_emp_data
