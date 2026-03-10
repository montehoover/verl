EDITABLE_FIELDS = {'name', 'position', 'department'}


def update_record(record, updates):
    """
    Update the given record dictionary with values from updates and return it.

    Args:
        record (dict): The original dictionary to update.
        updates (dict): A dictionary containing keys and values to merge into record.

    Returns:
        dict: The updated dictionary with values from updates applied.
    """
    record.update(updates)
    return record


def adjust_employee_profile(emp, changes, has_admin_rights):
    """
    Update the employee profile with provided changes respecting permissions.

    Args:
        emp (dict): Employee details.
        changes (dict): Proposed modifications.
        has_admin_rights (bool): If True, all fields may be updated; otherwise only EDITABLE_FIELDS.

    Returns:
        dict: The updated employee dictionary.
    """
    if has_admin_rights:
        emp.update(changes)
        return emp

    allowed_changes = {k: v for k, v in changes.items() if k in EDITABLE_FIELDS}
    emp.update(allowed_changes)
    return emp
