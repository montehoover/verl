EDITABLE_FIELDS = {"name", "position", "department"}


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

    if admin_flag:
        # Admins can update any existing field
        for field, value in adjustments.items():
            if field in updated:
                updated[field] = value
        return updated

    # Non-admins: only allow updates to EDITABLE_FIELDS and existing keys
    for field, value in adjustments.items():
        if field in EDITABLE_FIELDS and field in updated:
            updated[field] = value

    return updated
