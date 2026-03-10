from typing import Dict, Any

# Fields a regular user is allowed to modify.
EDITABLE_FIELDS = {"name", "position", "department"}

def modify_staff_details(employee_data: dict, changes: dict, is_superuser: bool) -> dict:
    """
    Modify an employee's details.

    - Regular users (is_superuser == False) may only change fields listed in EDITABLE_FIELDS.
    - Superusers may change any field present in `changes`.

    The function returns a new dictionary with the applied updates and does not mutate
    the input `employee_data`.

    Args:
        employee_data: Original employee data dictionary.
        changes: Proposed modifications to employee attributes.
        is_superuser: Whether the user performing the update is an admin.

    Returns:
        dict: Updated employee data.
    """
    if not isinstance(employee_data, dict):
        raise TypeError("employee_data must be a dict")
    if not isinstance(changes, dict):
        raise TypeError("changes must be a dict")

    updated = employee_data.copy()

    if is_superuser:
        fields_to_apply = changes.keys()
    else:
        fields_to_apply = EDITABLE_FIELDS.intersection(changes.keys())

    for field in fields_to_apply:
        updated[field] = changes[field]

    return updated
