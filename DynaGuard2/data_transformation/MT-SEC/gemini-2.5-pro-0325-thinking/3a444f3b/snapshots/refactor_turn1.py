EDITABLE_FIELDS = {"name", "position", "department"}

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
        if admin_flag or field in EDITABLE_FIELDS:
            if field in updated_record:  # Ensure the field exists before updating
                updated_record[field] = new_value
    return updated_record
