EDITABLE_FIELDS = {'name', 'position', 'department'}


def merge_dicts(original: dict, new_data: dict) -> dict:
    """
    Updates a dictionary with new values.

    Args:
        original: The original dictionary.
        new_data: A dictionary with keys and values to update in the original.

    Returns:
        The updated dictionary.
    """
    updated_dict = original.copy()
    updated_dict.update(new_data)
    return updated_dict


def validate_required_fields(data: dict, required_keys: set) -> bool:
    """
    Validates that required keys are present in a dictionary.

    Args:
        data: The dictionary to validate.
        required_keys: A set of keys that must be present in the data.

    Returns:
        True if all required keys are present, False otherwise.
    """
    return required_keys.issubset(data.keys())


def update_employee_record(employee: dict, updates: dict, is_admin: bool) -> dict:
    """
    Updates an employee's record with specific permissions.

    Args:
        employee: A dictionary of employee details.
        updates: A dictionary of fields to update.
        is_admin: A boolean indicating user permissions.

    Returns:
        The updated employee dictionary.
    """
    updated_employee = employee.copy()
    for key, value in updates.items():
        if is_admin or key in EDITABLE_FIELDS:
            updated_employee[key] = value
    return updated_employee
