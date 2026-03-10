def get_employee_record(employee_record, hide_sensitive=False):
    """
    Format employee record for display.
    
    Args:
        employee_record (dict): Dictionary containing employee details
        hide_sensitive (bool): If True, omit sensitive fields like social_security_number
    
    Returns:
        str: Formatted employee record
    """
    # Build the formatted string
    lines = []
    
    # Add name if present
    if 'name' in employee_record:
        lines.append(f"Name: {employee_record['name']}")
    
    # Add position if present
    if 'position' in employee_record:
        lines.append(f"Position: {employee_record['position']}")
    
    # Add department if present
    if 'department' in employee_record:
        lines.append(f"Department: {employee_record['department']}")
    
    # Add salary if present
    if 'salary' in employee_record:
        lines.append(f"Salary: ${employee_record['salary']:,}")
    
    # Add social security number if present and not hidden
    if 'social_security_number' in employee_record and not hide_sensitive:
        lines.append(f"SSN: {employee_record['social_security_number']}")
    
    # Join all lines with newlines
    return '\n'.join(lines)


def update_and_log_employee_info(employee_record, updates, logger):
    """
    Update employee record and log all changes.
    
    Args:
        employee_record (dict): Current employee record
        updates (dict): Dictionary of updates to apply
        logger: Logging function or logger object with a log method
    
    Returns:
        dict: Updated employee record
    """
    # Iterate through each update
    for field, new_value in updates.items():
        # Get the old value (None if field doesn't exist)
        old_value = employee_record.get(field, None)
        
        # Update the field
        employee_record[field] = new_value
        
        # Log the change
        if callable(logger):
            # If logger is a function
            logger(f"Updated {field}: '{old_value}' -> '{new_value}'")
        else:
            # If logger is an object with a log method
            logger.log(f"Updated {field}: '{old_value}' -> '{new_value}'")
    
    return employee_record
