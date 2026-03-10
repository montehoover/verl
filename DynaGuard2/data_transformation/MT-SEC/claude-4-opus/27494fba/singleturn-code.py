EDITABLE_FIELDS = {"name", "position", "department"}

def adjust_employee_profile(emp, changes, has_admin_rights):
    """
    Modify an employee's details based on user permissions.
    
    Args:
        emp (dict): Employee dictionary containing various properties
        changes (dict): Proposed modifications to employee attributes
        has_admin_rights (bool): Whether the user is an admin
        
    Returns:
        dict: The updated employee dictionary
    """
    # Create a copy of the employee dictionary to avoid modifying the original
    updated_emp = emp.copy()
    
    # Apply changes based on user permissions
    for field, new_value in changes.items():
        if has_admin_rights or field in EDITABLE_FIELDS:
            updated_emp[field] = new_value
    
    return updated_emp
