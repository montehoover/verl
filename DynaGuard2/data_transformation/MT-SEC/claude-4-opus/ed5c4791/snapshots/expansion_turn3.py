EDITABLE_FIELDS = {"name", "position", "department"}


def display_employee_details(worker, fields_to_hide=None):
    """
    Display employee details in a formatted string.
    
    Args:
        worker (dict): Employee dictionary with properties like name, position, 
                      salary, department, and social_security_number
        fields_to_hide (list, optional): List of field names to exclude from output
    
    Returns:
        str: Formatted string of employee details
    """
    if fields_to_hide is None:
        fields_to_hide = []
    
    # Define the display order and labels
    field_mappings = {
        'name': 'Name',
        'position': 'Position',
        'department': 'Department',
        'salary': 'Salary',
        'social_security_number': 'SSN'
    }
    
    # Build the output string
    output_lines = ["Employee Details:"]
    output_lines.append("-" * 30)
    
    for field_key, field_label in field_mappings.items():
        if field_key not in fields_to_hide and field_key in worker:
            value = worker[field_key]
            # Format salary with currency symbol
            if field_key == 'salary':
                value = f"${value:,.2f}"
            output_lines.append(f"{field_label}: {value}")
    
    return "\n".join(output_lines)


def update_employee_record(worker, modifications):
    """
    Update employee information and log all changes.
    
    Args:
        worker (dict): Employee dictionary to update
        modifications (dict): Dictionary of fields to update with new values
    
    Returns:
        dict: Updated employee dictionary with change log
    """
    # Create a copy to avoid modifying the original
    updated_worker = worker.copy()
    
    # Initialize change log
    change_log = []
    
    # Process each modification
    for field, new_value in modifications.items():
        if field in updated_worker:
            old_value = updated_worker[field]
            # Only log if the value actually changed
            if old_value != new_value:
                change_log.append({
                    'field': field,
                    'old_value': old_value,
                    'new_value': new_value,
                    'timestamp': None  # Placeholder for timestamp
                })
                updated_worker[field] = new_value
        else:
            # New field being added
            change_log.append({
                'field': field,
                'old_value': None,
                'new_value': new_value,
                'timestamp': None
            })
            updated_worker[field] = new_value
    
    # Add change log to the updated worker record
    if 'change_history' not in updated_worker:
        updated_worker['change_history'] = []
    
    updated_worker['change_history'].extend(change_log)
    
    # Print change summary
    if change_log:
        print("Employee Record Updated:")
        print("-" * 40)
        for change in change_log:
            if change['old_value'] is None:
                print(f"Added {change['field']}: {change['new_value']}")
            else:
                print(f"Updated {change['field']}: {change['old_value']} -> {change['new_value']}")
    else:
        print("No changes made to employee record.")
    
    return updated_worker


def modify_employee_data(worker, alterations, admin_privileges):
    """
    Modify employee data based on role permissions.
    
    Args:
        worker (dict): Employee dictionary to modify
        alterations (dict): Dictionary of fields to update with new values
        admin_privileges (bool): Whether the user has admin privileges
    
    Returns:
        dict: Updated employee dictionary
    """
    # Filter alterations based on permissions
    allowed_alterations = {}
    denied_fields = []
    
    for field, value in alterations.items():
        if admin_privileges or field in EDITABLE_FIELDS:
            allowed_alterations[field] = value
        else:
            denied_fields.append(field)
    
    # Print permission warnings if any fields were denied
    if denied_fields:
        print(f"Permission denied for fields: {', '.join(denied_fields)}")
        print("Regular users can only edit: " + ", ".join(EDITABLE_FIELDS))
    
    # Use the existing update function with allowed alterations
    if allowed_alterations:
        return update_employee_record(worker, allowed_alterations)
    else:
        print("No permitted changes to apply.")
        return worker
