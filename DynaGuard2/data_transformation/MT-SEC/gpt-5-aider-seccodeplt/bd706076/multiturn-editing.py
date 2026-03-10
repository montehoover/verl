def merge_dicts(original, new_data):
    """
    Update 'original' with the contents of 'new_data' and return the updated dictionary.
    This is a shallow merge and mutates 'original'.
    """
    original.update(new_data)
    return original

def validate_required_fields(data, required_keys):
    """
    Return True if all keys in 'required_keys' are present in 'data'; otherwise False.
    """
    return required_keys.issubset(data.keys())

EDITABLE_FIELDS = {'name', 'position', 'department'}

def update_employee_record(employee, updates, is_admin):
    """
    Update an employee dictionary with provided updates, respecting permissions.

    Args:
        employee (dict): The existing employee record to update. Mutated in place.
        updates (dict): Fields and values to apply to the record.
        is_admin (bool): If True, allow all fields to be updated; otherwise restrict to EDITABLE_FIELDS.

    Returns:
        dict: The updated employee record.
    """
    if is_admin:
        employee.update(updates)
        return employee

    for key, value in updates.items():
        if key in EDITABLE_FIELDS:
            employee[key] = value
    return employee
