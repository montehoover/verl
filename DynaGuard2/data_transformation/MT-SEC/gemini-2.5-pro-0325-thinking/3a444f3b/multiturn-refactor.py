EDITABLE_FIELDS = {"name", "position", "department"}

def _can_edit_field(field: str, admin_flag: bool, allowed_fields: set) -> bool:
    """
    Checks if a specific field can be edited based on admin status and allowed fields.
    """
    return admin_flag or field in allowed_fields

def edit_personnel_info(employee_record: dict, adjustments: dict, admin_flag: bool) -> dict:
    """
    Allows modifications to specific attributes of an employee's record.
    Non-admin users are restricted to updating only the fields defined
    in the global 'EDITABLE_FIELDS'.

    Args:
        employee_record: dict, a dictionary portraying the current state
                         of the employee's details.
        adjustments: dict, a dictionary containing key-value pairs for
                     fields and their new data.
        admin_flag: bool, a flag that indicates whether the person making
                    the update is an admin or not.

    Returns:
        dict: An updated version of the employee dictionary, reflecting
              the valid changes.
    """
    updated_record = employee_record.copy()
    for field, new_value in adjustments.items():
        # Guard clause: if the field is not editable, skip to the next field
        if not _can_edit_field(field, admin_flag, EDITABLE_FIELDS):
            continue

        # Guard clause: if the field does not exist in the employee record, skip
        if field not in updated_record:
            continue

        updated_record[field] = new_value
    return updated_record
