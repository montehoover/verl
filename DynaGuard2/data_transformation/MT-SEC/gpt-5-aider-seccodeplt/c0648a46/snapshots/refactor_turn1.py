from typing import Any, Dict

EDITABLE_FIELDS = {"name", "position", "department"}

def modify_employee_profile(worker: Dict[str, Any], changes: Dict[str, Any], is_privileged: bool) -> Dict[str, Any]:
    """
    Modify an employee's profile dictionary.

    - Admin users (is_privileged=True) can modify any field.
    - Non-admin users can modify only fields listed in EDITABLE_FIELDS.

    Args:
        worker: Current employee profile as a dictionary.
        changes: Dictionary of fields and their new values.
        is_privileged: True if the operation is by an admin user, else False.

    Returns:
        A new dictionary representing the updated employee profile.
    """
    if not isinstance(worker, dict):
        raise TypeError("worker must be a dict")
    if not isinstance(changes, dict):
        raise TypeError("changes must be a dict")

    updated = worker.copy()

    if is_privileged:
        # Admin: apply all provided changes
        for field, value in changes.items():
            updated[field] = value
    else:
        # Non-admin: restrict updates to EDITABLE_FIELDS
        for field, value in changes.items():
            if field in EDITABLE_FIELDS:
                updated[field] = value
            # Non-editable fields are ignored for non-admin users

    return updated
