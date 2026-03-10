EDITABLE_FIELDS = {"name", "position", "department"}

def modify_staff_info(staff, changes, admin_status):
    """
    Modify specific attributes of an employee's record.
    
    Args:
        staff (dict): A dictionary portraying the current state of the employee's details.
        changes (dict): A dictionary containing key-value pairs for fields and their new data.
        admin_status (bool): A flag that indicates whether the person making the update is an admin or not.
    
    Returns:
        dict: An updated version of the employee dictionary, reflecting the valid changes.
    """
    # Create a copy of the staff dictionary to avoid modifying the original
    updated_staff = staff.copy()
    
    # Iterate through the requested changes
    for field, new_value in changes.items():
        # If admin, allow any field to be updated
        if admin_status:
            updated_staff[field] = new_value
        # If not admin, only allow updates to fields in EDITABLE_FIELDS
        elif field in EDITABLE_FIELDS:
            updated_staff[field] = new_value
        # Non-admin users cannot update fields not in EDITABLE_FIELDS
        # These changes are simply ignored
    
    return updated_staff
