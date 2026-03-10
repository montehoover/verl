from typing import Dict, Any

# Fields that non-admin users are allowed to edit
EDITABLE_FIELDS = {"name", "position", "department"}


def alter_employee_details(emp_data: Dict[str, Any], changes: Dict[str, Any], is_superuser: bool) -> Dict[str, Any]:
    """
    Modify an employee's details based on the provided changes and privileges.

    Args:
        emp_data: dict portraying the current state of the employee's details.
        changes: dict containing key-value pairs for fields and their new data.
        is_superuser: bool flag indicating whether the updater is an admin.

    Returns:
        A new dict with the employee details updated according to the rules:
        - If is_superuser is True, all provided changes are applied.
        - If is_superuser is False, only fields listed in EDITABLE_FIELDS are applied.

    Notes:
        The function does not modify the original emp_data dictionary.
    """
    if not isinstance(emp_data, dict):
        raise TypeError("emp_data must be a dict")
    if not isinstance(changes, dict):
        raise TypeError("changes must be a dict")
    if not isinstance(is_superuser, bool):
        raise TypeError("is_superuser must be a bool")

    # Create a shallow copy to avoid mutating the original input
    updated = dict(emp_data)

    if is_superuser:
        # Admins can update any field present in changes (including adding new fields)
        for key, value in changes.items():
            updated[key] = value
    else:
        # Non-admins can update only whitelisted fields
        for key, value in changes.items():
            if key in EDITABLE_FIELDS:
                updated[key] = value
            # Silently ignore attempts to modify non-editable fields

    return updated
