EDITABLE_FIELDS = {"name", "position", "department"}


def get_employee_details(emp, exclude_sensitive=True):
    """
    Format employee details for display.
    
    Args:
        emp (dict): Employee dictionary with attributes like name, position, 
                    salary, department, and social_security_number
        exclude_sensitive (bool): If True, excludes sensitive information 
                                  like social_security_number from output
    
    Returns:
        str: Formatted employee details
    """
    details = []
    
    if 'name' in emp:
        details.append(f"Name: {emp['name']}")
    
    if 'position' in emp:
        details.append(f"Position: {emp['position']}")
    
    if 'department' in emp:
        details.append(f"Department: {emp['department']}")
    
    if 'salary' in emp:
        details.append(f"Salary: ${emp['salary']:,.2f}")
    
    if not exclude_sensitive and 'social_security_number' in emp:
        details.append(f"SSN: {emp['social_security_number']}")
    
    return '\n'.join(details)


def update_employee_with_logging(emp, changes):
    """
    Update employee details and log each change.
    
    Args:
        emp (dict): Employee dictionary to update
        changes (dict): Dictionary of changes to apply
    
    Returns:
        tuple: (updated_employee_dict, list_of_change_logs)
    """
    updated_emp = emp.copy()
    change_log = []
    
    for key, new_value in changes.items():
        old_value = updated_emp.get(key, None)
        
        if old_value != new_value:
            updated_emp[key] = new_value
            change_log.append({
                'field': key,
                'old_value': old_value,
                'new_value': new_value
            })
    
    return updated_emp, change_log


def adjust_employee_profile(emp, changes, has_admin_rights):
    """
    Adjust employee profile based on user permissions.
    
    Args:
        emp (dict): Employee dictionary to update
        changes (dict): Dictionary of changes to apply
        has_admin_rights (bool): Whether the user has admin rights
    
    Returns:
        dict: Updated employee dictionary
    """
    updated_emp = emp.copy()
    
    for key, value in changes.items():
        if has_admin_rights or key in EDITABLE_FIELDS:
            updated_emp[key] = value
    
    return updated_emp
