EDITABLE_FIELDS = {"name", "position", "department"}


def display_employee_profile(worker):
    """
    Display a formatted employee profile.
    
    Args:
        worker (dict): Dictionary containing employee information with keys:
                      - name (str): Employee's name
                      - position (str): Employee's job position
                      - salary (float/int): Employee's salary
                      - department (str): Employee's department
                      - social_security_number (str): Employee's SSN
    
    Returns:
        str: Formatted string displaying employee profile
    """
    profile = f"""Employee Profile
===================
Name: {worker['name']}
Position: {worker['position']}
Department: {worker['department']}
Salary: ${worker['salary']:,.2f}
SSN: {worker['social_security_number']}"""
    
    return profile


def track_and_update_employee(worker, modifications):
    """
    Update employee information and log all changes.
    
    Args:
        worker (dict): Original employee dictionary
        modifications (dict): Dictionary containing fields to update
    
    Returns:
        dict: Updated employee dictionary
    """
    import copy
    from datetime import datetime
    
    # Create a copy of the original employee to preserve it
    updated_worker = copy.deepcopy(worker)
    
    # Track changes
    change_log = []
    
    for field, new_value in modifications.items():
        if field in updated_worker:
            old_value = updated_worker[field]
            if old_value != new_value:
                change_log.append({
                    'field': field,
                    'old_value': old_value,
                    'new_value': new_value,
                    'timestamp': datetime.now().isoformat()
                })
                updated_worker[field] = new_value
    
    # Log changes
    if change_log:
        print("Employee Update Log")
        print("==================")
        for change in change_log:
            print(f"Field: {change['field']}")
            print(f"  Old Value: {change['old_value']}")
            print(f"  New Value: {change['new_value']}")
            print(f"  Timestamp: {change['timestamp']}")
            print()
    else:
        print("No changes were made to the employee profile.")
    
    return updated_worker


def modify_employee_profile(worker, changes, is_privileged):
    """
    Modify employee profile with permission-based restrictions.
    
    Args:
        worker (dict): Employee dictionary to modify
        changes (dict): Dictionary containing fields to update
        is_privileged (bool): True if user has admin privileges
    
    Returns:
        dict: Updated employee dictionary
    """
    import copy
    
    # Create a copy of the original employee
    updated_worker = copy.deepcopy(worker)
    
    # Filter changes based on permissions
    allowed_changes = {}
    
    for field, new_value in changes.items():
        if is_privileged:
            # Admin users can edit any field
            allowed_changes[field] = new_value
        else:
            # Non-admin users can only edit fields in EDITABLE_FIELDS
            if field in EDITABLE_FIELDS:
                allowed_changes[field] = new_value
            else:
                print(f"Permission denied: Cannot modify field '{field}'")
    
    # Apply allowed changes
    for field, new_value in allowed_changes.items():
        if field in updated_worker:
            updated_worker[field] = new_value
    
    return updated_worker
