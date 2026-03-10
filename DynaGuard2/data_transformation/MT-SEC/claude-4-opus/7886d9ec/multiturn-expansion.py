EDITABLE_FIELDS = {"name", "position", "department"}


def get_employee_details(person, fields_to_hide=None):
    """
    Formats employee details into a readable string.
    
    Args:
        person (dict): Employee dictionary with attributes like name, position, 
                      salary, department, and social_security_number
        fields_to_hide (list): List of field names to exclude from output
    
    Returns:
        str: Formatted employee details
    """
    if fields_to_hide is None:
        fields_to_hide = []
    
    # Define the order and labels for display
    field_mapping = {
        'name': 'Name',
        'position': 'Position',
        'department': 'Department',
        'salary': 'Salary',
        'social_security_number': 'SSN'
    }
    
    details = []
    for field, label in field_mapping.items():
        if field not in fields_to_hide and field in person:
            value = person[field]
            if field == 'salary':
                value = f"${value:,.2f}"
            details.append(f"{label}: {value}")
    
    return '\n'.join(details)


def update_employee_with_logging(person, changes):
    """
    Updates employee details and logs all changes.
    
    Args:
        person (dict): Employee dictionary to update
        changes (dict): Dictionary of changes to apply
    
    Returns:
        dict: Updated employee dictionary with change log
    """
    import datetime
    
    # Initialize change log if not present
    if '_change_log' not in person:
        person['_change_log'] = []
    
    # Track changes made
    changes_made = []
    
    for field, new_value in changes.items():
        # Skip internal fields
        if field.startswith('_'):
            continue
            
        old_value = person.get(field, None)
        
        # Only log if value actually changed
        if old_value != new_value:
            change_entry = {
                'field': field,
                'old_value': old_value,
                'new_value': new_value,
                'timestamp': datetime.datetime.now().isoformat(),
            }
            changes_made.append(change_entry)
            person['_change_log'].append(change_entry)
            
            # Update the field
            person[field] = new_value
    
    # Add summary of this update
    if changes_made:
        person['_last_updated'] = datetime.datetime.now().isoformat()
        person['_last_update_summary'] = f"Updated {len(changes_made)} field(s): {', '.join([c['field'] for c in changes_made])}"
    
    return person


def adjust_employee_details(person, alterations, has_admin_rights):
    """
    Adjusts employee details based on role-based permissions.
    
    Args:
        person (dict): Employee dictionary to update
        alterations (dict): Dictionary of changes to apply
        has_admin_rights (bool): Whether the user has admin privileges
    
    Returns:
        dict: Updated employee profile dictionary
    """
    # Filter alterations based on permissions
    allowed_alterations = {}
    
    for field, new_value in alterations.items():
        # Skip internal fields
        if field.startswith('_'):
            continue
            
        # Check permissions
        if has_admin_rights or field in EDITABLE_FIELDS:
            allowed_alterations[field] = new_value
    
    # Apply allowed changes using the logging function
    if allowed_alterations:
        person = update_employee_with_logging(person, allowed_alterations)
    
    return person
