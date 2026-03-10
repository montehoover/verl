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
