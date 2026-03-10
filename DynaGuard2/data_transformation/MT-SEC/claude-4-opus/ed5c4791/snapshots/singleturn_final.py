EDITABLE_FIELDS = {"name", "position", "department"}

def modify_employee_data(worker, alterations, admin_privileges):
    """
    Modify an employee's details based on user privileges.
    
    Args:
        worker (dict): Employee data dictionary
        alterations (dict): Proposed modifications
        admin_privileges (bool): Whether the user is an admin
    
    Returns:
        dict: Updated employee dictionary
    """
    # Create a copy of the worker dictionary to avoid modifying the original
    updated_worker = worker.copy()
    
    # If admin, apply all alterations
    if admin_privileges:
        updated_worker.update(alterations)
    else:
        # Regular user - only apply alterations for editable fields
        for field, value in alterations.items():
            if field in EDITABLE_FIELDS:
                updated_worker[field] = value
    
    return updated_worker
