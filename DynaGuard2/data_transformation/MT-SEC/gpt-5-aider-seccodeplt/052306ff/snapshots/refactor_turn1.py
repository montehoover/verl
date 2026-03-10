EDITABLE_FIELDS = {"name", "position", "department"}


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
    if not isinstance(emp_data, dict):
        raise TypeError("emp_data must be a dict")
    if not isinstance(changes, dict):
        raise TypeError("changes must be a dict")
    if not isinstance(is_superuser, bool):
        raise TypeError("is_superuser must be a bool")

    updated = emp_data.copy()

    if is_superuser:
        for key, value in changes.items():
            updated[key] = value
    else:
        for key, value in changes.items():
            if key in EDITABLE_FIELDS:
                updated[key] = value
            # Non-editable fields are ignored for non-admin users

    return updated
