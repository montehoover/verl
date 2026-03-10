EDITABLE_FIELDS = {"name", "position", "department"}


def modify_staff_details(employee_data, changes, is_superuser):
    """
    Modify an employee's details based on user permissions.
    
    Args:
        employee_data (dict): Employee information to be modified
        changes (dict): Proposed modifications to employee attributes
        is_superuser (bool): Whether the user has admin privileges
        
    Returns:
        dict: Updated employee dictionary
    """
    # Create a copy to preserve the original data
    updated_employee = employee_data.copy()
    
    # Superusers can modify all fields
    if is_superuser:
        updated_employee.update(changes)
        return updated_employee
    
    # Regular users can only modify allowed fields
    for field, value in changes.items():
        if field not in EDITABLE_FIELDS:
            continue
            
        updated_employee[field] = value
    
    return updated_employee
