EDITABLE_FIELDS = {"name", "position", "department"}

def modify_employee_profile(worker, changes, is_privileged):
    """
    Modifies certain fields within an employee's profile.
    
    Args:
        worker (dict): The current employee profile dictionary
        changes (dict): Dictionary specifying fields and new values to update
        is_privileged (bool): Whether the user is an admin (True) or not (False)
    
    Returns:
        dict: The revised employee profile dictionary after updates
    """
    # Create a copy of the worker profile to avoid modifying the original
    updated_worker = worker.copy()
    
    # Apply changes based on user privileges
    for field, new_value in changes.items():
        if is_privileged:
            # Admin users can modify any field
            updated_worker[field] = new_value
        else:
            # Non-admin users can only modify fields in EDITABLE_FIELDS
            if field in EDITABLE_FIELDS:
                updated_worker[field] = new_value
    
    return updated_worker
