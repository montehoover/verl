def display_vehicle_info(details, fields_to_hide=None):
    """
    Display vehicle information from a dictionary.
    
    Args:
        details (dict): Vehicle details dictionary with keys like 'make', 'model', 'year', 'owner', 'registration'
        fields_to_hide (list): Optional list of field names to exclude from display
    
    Returns:
        str: Formatted string containing vehicle information
    """
    if fields_to_hide is None:
        fields_to_hide = []
    
    # Define display labels for better formatting
    field_labels = {
        'make': 'Make',
        'model': 'Model',
        'year': 'Year',
        'owner': 'Owner',
        'registration': 'Registration'
    }
    
    # Build the formatted string
    lines = ["Vehicle Information:"]
    lines.append("-" * 20)
    
    for field, label in field_labels.items():
        if field in details and field not in fields_to_hide:
            lines.append(f"{label}: {details[field]}")
    
    return "\n".join(lines)
