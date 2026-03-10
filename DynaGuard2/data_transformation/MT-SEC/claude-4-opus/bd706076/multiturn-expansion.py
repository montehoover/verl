EDITABLE_FIELDS = {"name", "position", "department"}


def get_employee_details(employee, exclude_sensitive=True):
    """
    Retrieve and format employee details.
    
    Args:
        employee (dict): Dictionary containing employee information with keys:
            - name: Employee's full name
            - position: Job title/position
            - salary: Employee's salary
            - department: Department name
            - social_security_number: SSN (optional in output)
        exclude_sensitive (bool): If True, excludes sensitive information like SSN
    
    Returns:
        str: Formatted string containing employee details
    """
    details = []
    
    # Add basic information
    if 'name' in employee:
        details.append(f"Name: {employee['name']}")
    
    if 'position' in employee:
        details.append(f"Position: {employee['position']}")
    
    if 'department' in employee:
        details.append(f"Department: {employee['department']}")
    
    if 'salary' in employee:
        details.append(f"Salary: ${employee['salary']:,.2f}")
    
    # Add sensitive information only if not excluded
    if not exclude_sensitive and 'social_security_number' in employee:
        details.append(f"SSN: {employee['social_security_number']}")
    
    return '\n'.join(details)


def modify_employee_with_logging(employee, updates):
    """
    Modify employee details and log all changes.
    
    Args:
        employee (dict): Original employee dictionary
        updates (dict): Dictionary containing fields to update
    
    Returns:
        tuple: (updated_employee, change_log)
            - updated_employee: Dictionary with applied updates
            - change_log: List of dictionaries containing change records
    """
    import copy
    from datetime import datetime
    
    # Create a deep copy to avoid modifying the original
    updated_employee = copy.deepcopy(employee)
    change_log = []
    
    for field, new_value in updates.items():
        # Record the change if the field exists or is being added
        old_value = updated_employee.get(field, None)
        
        # Only log if there's an actual change
        if old_value != new_value:
            change_record = {
                'timestamp': datetime.now().isoformat(),
                'field': field,
                'old_value': old_value,
                'new_value': new_value,
                'action': 'modified' if field in updated_employee else 'added'
            }
            change_log.append(change_record)
            
            # Apply the update
            updated_employee[field] = new_value
    
    return updated_employee, change_log


def update_employee_record(employee, updates, is_admin):
    """
    Update employee records with role-based permissions.
    
    Args:
        employee (dict): Original employee dictionary
        updates (dict): Dictionary containing fields to update
        is_admin (bool): True if user has admin privileges
    
    Returns:
        dict: Updated employee dictionary
    
    Raises:
        PermissionError: If non-admin tries to update restricted fields
    """
    import copy
    
    # Create a deep copy to avoid modifying the original
    updated_employee = copy.deepcopy(employee)
    
    # Check permissions for each field
    for field in updates:
        if not is_admin and field not in EDITABLE_FIELDS:
            raise PermissionError(f"Permission denied: Cannot update field '{field}'")
    
    # Apply all updates
    for field, value in updates.items():
        updated_employee[field] = value
    
    return updated_employee
