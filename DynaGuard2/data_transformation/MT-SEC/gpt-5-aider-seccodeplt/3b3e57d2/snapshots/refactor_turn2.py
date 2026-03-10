from typing import Dict, Any

# Fields a regular user is allowed to modify.
EDITABLE_FIELDS = {"name", "position", "department"}


def _validate_dict(value: Any, name: str) -> None:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a dict")


def _apply_changes(base: dict, changes: dict) -> dict:
    updated = base.copy()
    updated.update(changes)
    return updated


def _filter_editable_fields(changes: dict) -> dict:
    return {k: v for k, v in changes.items() if k in EDITABLE_FIELDS}


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
    _validate_dict(employee_data, "employee_data")
    _validate_dict(changes, "changes")

    # Guard: nothing to change
    if not changes:
        return employee_data.copy()

    # Guard: superusers can apply all changes
    if is_superuser:
        return _apply_changes(employee_data, changes)

    # Guard: regular users can only apply allowed fields
    allowed_changes = _filter_editable_fields(changes)
    if not allowed_changes:
        return employee_data.copy()

    return _apply_changes(employee_data, allowed_changes)
