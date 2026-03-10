EDITABLE_FIELDS = {"name", "position", "department"}

def edit_personnel_info(employee_record: dict, adjustments: dict, admin_flag: bool) -> dict:
    """
    Allows modifications to specific attributes of an employee's record.
    Non-admin users are restricted to updating only the fields defined
    in the global 'EDITABLE_FIELDS'.

    Args:
        employee_record: A dictionary portraying the current state of the employee's details.
        adjustments: A dictionary containing key-value pairs for fields and their new data.
        admin_flag: A flag that indicates whether the person making the update is an admin or not.

    Returns:
        An updated version of the employee dictionary, reflecting the valid changes.
    """
    updated_record = employee_record.copy()  # Work on a copy to avoid modifying the original dict directly

    for key, value in adjustments.items():
        if admin_flag:
            updated_record[key] = value
        elif key in EDITABLE_FIELDS:
            updated_record[key] = value
    
    return updated_record
