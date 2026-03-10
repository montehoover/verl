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
