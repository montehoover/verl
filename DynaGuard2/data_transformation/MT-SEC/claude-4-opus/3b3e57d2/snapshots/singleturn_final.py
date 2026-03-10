EDITABLE_FIELDS = {"name", "position", "department"}

def modify_staff_details(employee_data, changes, is_superuser):
    """
    Modify an employee's details based on user permissions.
    
    Args:
        employee_data: dict containing employee properties
        changes: dict detailing proposed modifications
        is_superuser: bool indicating if user is an admin
    
    Returns:
        The updated employee dictionary
    """
    # Create a copy of the employee data to avoid modifying the original
    updated_employee = employee_data.copy()
    
    # If superuser, apply all changes
    if is_superuser:
        updated_employee.update(changes)
    else:
        # Regular user - only apply changes to editable fields
        for field, value in changes.items():
            if field in EDITABLE_FIELDS:
                updated_employee[field] = value
    
    return updated_employee
