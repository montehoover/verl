EDITABLE_FIELDS = {"name", "position", "department"}


def edit_personnel_info(employee_record: dict, adjustments: dict, admin_flag: bool) -> dict:
    """
    Modify an employee's information according to role-based field permissions.

    Args:
        employee_record (dict): Current state of the employee's details.
        adjustments (dict): Key-value pairs for fields to update and their new values.
        admin_flag (bool): True if the requester is an admin; False otherwise.

    Returns:
        dict: An updated copy of the employee record reflecting valid changes.

    Behavior:
        - Admin users can update any field (existing or new).
        - Non-admin users can only update fields in EDITABLE_FIELDS.
        - The original employee_record is not mutated; a copy is returned.
        - Fields in adjustments that are not permitted for non-admins are ignored.
    """
    if not isinstance(employee_record, dict):
        raise TypeError("employee_record must be a dict")
    if not isinstance(adjustments, dict):
        raise TypeError("adjustments must be a dict")
    if not isinstance(admin_flag, bool):
        raise TypeError("admin_flag must be a bool")

    updated_record = dict(employee_record)  # shallow copy

    if admin_flag:
        # Admin can update any provided fields.
        for field, value in adjustments.items():
            updated_record[field] = value
        return updated_record

    # Non-admin: restrict updates to EDITABLE_FIELDS
    for field, value in adjustments.items():
        if field in EDITABLE_FIELDS:
            updated_record[field] = value

    return updated_record
