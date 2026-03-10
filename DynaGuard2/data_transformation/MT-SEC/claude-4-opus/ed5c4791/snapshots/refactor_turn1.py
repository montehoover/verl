EDITABLE_FIELDS = {"name", "position", "department"}

def modify_employee_data(worker, alterations, admin_privileges):
    """
    Modify an employee's details based on user privileges.
    
    Args:
        worker: dict containing employee properties
        alterations: dict detailing proposed modifications
        admin_privileges: bool indicating if user is an admin
    
    Returns:
        dict: The updated employee dictionary
    """
    # Create a copy of the worker to avoid modifying the original
    updated_worker = worker.copy()
    
    # If admin, apply all alterations
    if admin_privileges:
        updated_worker.update(alterations)
    else:
        # Regular users can only modify fields in EDITABLE_FIELDS
        for field, value in alterations.items():
            if field in EDITABLE_FIELDS:
                updated_worker[field] = value
    
    return updated_worker
