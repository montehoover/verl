EDITABLE_FIELDS = {"name", "position", "department"}

def modify_employee_profile(worker: dict, changes: dict, is_privileged: bool) -> dict:
    """
    Modifies certain fields within an employee's profile.

    Args:
        worker: dict, represents the current employee profile.
        changes: dict, a dictionary specifying the fields and new values.
        is_privileged: bool, indicates if the operation is by an admin.

    Returns:
        dict: The revised employee profile dictionary.
    """
    for field, value in changes.items():
        if is_privileged or field in EDITABLE_FIELDS:
            worker[field] = value
    return worker
