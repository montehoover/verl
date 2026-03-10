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
