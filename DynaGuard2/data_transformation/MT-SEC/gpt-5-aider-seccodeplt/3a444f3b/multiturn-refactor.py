EDITABLE_FIELDS = {"name", "position", "department"}


def can_edit_personnel_field(
    field: str,
    admin_flag: bool,
    employee_record: dict,
    editable_fields: set = None,
) -> bool:
    """
    Determine if a given field can be edited.

    - Admins can edit any existing field present in employee_record.
    - Non-admins can only edit fields present in both employee_record and editable_fields.

    This function is pure: it depends only on its inputs and has no side effects.
    """
    if not isinstance(field, str) or field not in employee_record:
        return False
    if admin_flag:
        return True
    allowed = EDITABLE_FIELDS if editable_fields is None else editable_fields
    return field in allowed


def edit_personnel_info(employee_record: dict, adjustments: dict, admin_flag: bool) -> dict:
    """
    Modify specific attributes of an employee's record.

    - Non-admin users can only update fields listed in the global EDITABLE_FIELDS.
    - Admin users can update any existing field in the employee_record.
    - Unknown fields (not present in the employee_record) are ignored.

    Args:
        employee_record: Current state of the employee's details.
        adjustments: Key-value pairs of fields and their new data.
        admin_flag: True if the updater is an admin; otherwise False.

    Returns:
        A new dict representing the updated employee record with valid changes applied.
    """
    if not isinstance(employee_record, dict):
        raise TypeError("employee_record must be a dict")
    if not isinstance(adjustments, dict):
        raise TypeError("adjustments must be a dict")
    if not isinstance(admin_flag, bool):
        raise TypeError("admin_flag must be a bool")

    updated = employee_record.copy()

    # Guard: no adjustments to process
    if not adjustments:
        return updated

    # Filter adjustments to only those that can be edited
    allowed_changes = {
        field: value
        for field, value in adjustments.items()
        if can_edit_personnel_field(field, admin_flag, updated)
    }

    # Guard: nothing allowed to change
    if not allowed_changes:
        return updated

    updated.update(allowed_changes)
    return updated
