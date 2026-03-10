EDITABLE_FIELDS = {"name", "position", "department"}

def update_employee_fields(employee, updates):
    """
    Apply updates to employee fields.
    
    Args:
        employee: dict containing employee properties
        updates: dict containing field updates
    
    Returns:
        dict: Updated employee dictionary
    """
    updated_employee = employee.copy()
    updated_employee.update(updates)
    return updated_employee

def filter_allowed_fields(updates, allowed_fields):
    """
    Filter updates to only include allowed fields.
    
    Args:
        updates: dict containing proposed updates
        allowed_fields: set of field names that are allowed
    
    Returns:
        dict: Filtered updates containing only allowed fields
    """
    return {field: value for field, value in updates.items() if field in allowed_fields}

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
    # Guard clause: if no alterations, return unchanged worker
    if not alterations:
        return worker.copy()
    
    # Admin users can update all fields
    if admin_privileges:
        return update_employee_fields(worker, alterations)
    
    # Regular users can only update editable fields
    allowed_updates = filter_allowed_fields(alterations, EDITABLE_FIELDS)
    return update_employee_fields(worker, allowed_updates)
