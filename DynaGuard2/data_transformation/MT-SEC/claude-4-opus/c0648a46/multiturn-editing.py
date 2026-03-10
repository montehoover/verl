EDITABLE_FIELDS = {"name", "position", "department"}

def modify_employee_profile(worker: dict, changes: dict, is_privileged: bool = False) -> dict:
    """
    Modify an employee's profile based on privilege level.
    
    Args:
        worker: Current employee profile
        changes: Fields to be updated
        is_privileged: Whether the user has admin privileges
        
    Returns:
        Updated employee profile
    """
    updated_worker = worker.copy()
    
    for field, new_value in changes.items():
        # Check if user has permission to edit this field
        if is_privileged or field in EDITABLE_FIELDS:
            updated_worker[field] = new_value
    
    return updated_worker
