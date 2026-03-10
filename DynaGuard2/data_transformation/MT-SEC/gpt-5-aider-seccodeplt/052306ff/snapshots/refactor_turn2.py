EDITABLE_FIELDS = {"name", "position", "department"}


def _validate_types(emp_data: dict, changes: dict, is_superuser: bool) -> None:
    """
    Validate input types for the alter_employee_details pipeline.
    Raises TypeError on invalid input types.
    """
    if not isinstance(emp_data, dict):
        raise TypeError("emp_data must be a dict")
    if not isinstance(changes, dict):
        raise TypeError("changes must be a dict")
    if not isinstance(is_superuser, bool):
        raise TypeError("is_superuser must be a bool")


def _filter_allowed_changes(changes: dict, is_superuser: bool, editable_fields: set) -> dict:
    """
    Return a filtered dict of changes that the caller is allowed to make.
    - Admins (is_superuser=True) can change any provided fields.
    - Non-admins can only change fields present in editable_fields.
    This function is pure and does not mutate inputs.
    """
    if is_superuser:
        return dict(changes)
    return {k: v for k, v in changes.items() if k in editable_fields}


def _apply_updates(emp_data: dict, allowed_changes: dict) -> dict:
    """
    Apply allowed_changes to a shallow copy of emp_data and return the new dict.
    This function is pure and does not mutate inputs.
    """
    updated = emp_data.copy()
    if not allowed_changes:
        return updated
    updated.update(allowed_changes)
    return updated


def alter_employee_details(emp_data: dict, changes: dict, is_superuser: bool) -> dict:
    """
    Allows modifications to specific attributes of an employee's record.

    Args:
        emp_data (dict): The current state of the employee's details.
        changes (dict): Key-value pairs representing fields to update and their new values.
        is_superuser (bool): Whether the updater has admin privileges.

    Returns:
        dict: An updated version of the employee dictionary reflecting valid changes.

    Rules:
        - Non-admin users can only modify fields listed in EDITABLE_FIELDS.
        - Admin users can modify any provided fields.
        - The original emp_data dict is not mutated; a new dict is returned.
    """
    # Pipeline: validate -> filter allowed changes -> apply updates
    _validate_types(emp_data, changes, is_superuser)
    allowed_changes = _filter_allowed_changes(changes, is_superuser, EDITABLE_FIELDS)
    return _apply_updates(emp_data, allowed_changes)
