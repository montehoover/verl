EDITABLE_FIELDS = {"name", "position", "department"}

def edit_personnel_info(employee_record, adjustments, admin_flag):
    """
    Allows modifications to specific attributes of an employee's record.
    
    Args:
        employee_record: dict, a dictionary portraying the current state of the employee's details.
        adjustments: dict, a dictionary containing key-value pairs for fields and their new data.
        admin_flag: bool, a flag that indicates whether the person making the update is an admin or not.
    
    Returns:
        dict: An updated version of the employee dictionary, reflecting the valid changes.
    """
    # Create a copy of the employee record to avoid modifying the original
    updated_record = employee_record.copy()
    
    # Iterate through the adjustments
    for field, new_value in adjustments.items():
        # Check if the field can be updated
        if admin_flag or field in EDITABLE_FIELDS:
            # Only update if the field exists in the original record
            if field in updated_record:
                updated_record[field] = new_value
    
    return updated_record
