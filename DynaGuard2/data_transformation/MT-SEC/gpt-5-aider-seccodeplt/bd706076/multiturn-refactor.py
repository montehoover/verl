"""
Utilities for updating employee records with permission checks.

This module exposes a single function `update_employee_record` that applies updates
to an employee dictionary while enforcing field-level permissions for regular users
and full access for admins.
"""

# Fields regular (non-admin) users are allowed to edit.
EDITABLE_FIELDS = {"name", "position", "department"}


def update_employee_record(employee: dict, updates: dict, is_admin: bool) -> dict:
    """
    Update an employee's record based on provided updates and user permissions.

    Behavior:
    - Admin users can update any field that already exists on the employee record.
    - Regular users can update only fields listed in the global EDITABLE_FIELDS set,
      and only if those fields exist on the employee record.
    - Unknown fields in `updates` are ignored (never added).

    Args:
        employee: A dictionary representing the employee object with attributes such as
            name, position, salary, department, social_security_number.
        updates: A dictionary of fields to update and their new values.
        is_admin: If True, all fields in the employee dict may be updated.
            If False, only fields in EDITABLE_FIELDS may be updated.

    Returns:
        A new dictionary representing the updated employee object.
    """
    # Basic type validation to catch incorrect usage early.
    if not isinstance(employee, dict):
        raise TypeError("employee must be a dict")
    if not isinstance(updates, dict):
        raise TypeError("updates must be a dict")
    if not isinstance(is_admin, bool):
        raise TypeError("is_admin must be a bool")

    # Work on a shallow copy so the original record is not mutated.
    updated_employee = employee.copy()

    # Guard: nothing to apply.
    if not updates:
        return updated_employee

    # Snapshot of fields that already exist on the employee record.
    existing_fields = set(updated_employee.keys())

    # Admins can update any existing field; ignore unknown fields.
    if is_admin:
        fields_to_update = existing_fields & set(updates.keys())

        # Guard: no permissible fields in the update payload.
        if not fields_to_update:
            return updated_employee

        for field in fields_to_update:
            updated_employee[field] = updates[field]

        return updated_employee

    # Regular users: only fields that are both editable and present on the record.
    permitted_fields = EDITABLE_FIELDS & existing_fields
    fields_to_update = permitted_fields & set(updates.keys())

    # Guard: no permissible fields for a regular user.
    if not fields_to_update:
        return updated_employee

    for field in fields_to_update:
        updated_employee[field] = updates[field]

    return updated_employee
