EDITABLE_FIELDS = {"name", "position", "department"}

def update_employee_record(employee, updates, is_admin):
    """
    Update an employee's record with new values.
    
    Args:
        employee (dict): The original employee record containing fields like
                        name, position, salary, department, social_security_number
        updates (dict): Dictionary of field names and their new values
        is_admin (bool): Whether the user has admin privileges
        
    Returns:
        dict: A new dictionary with the updated employee information
        
    Note:
        - Regular users can only update fields in EDITABLE_FIELDS
        - Admins can update any existing field
        - Non-existent fields are ignored
    """
    # Create a new dictionary to preserve the original employee data
    updated_employee_record = employee.copy()
    
    # Process each requested update
    for field_name, new_value in updates.items():
        
        # Check if user has permission to update this field
        if not is_admin and field_name not in EDITABLE_FIELDS:
            continue
            
        # Verify the field exists in the employee record
        if field_name not in updated_employee_record:
            continue
            
        # Apply the update
        updated_employee_record[field_name] = new_value
    
    return updated_employee_record
