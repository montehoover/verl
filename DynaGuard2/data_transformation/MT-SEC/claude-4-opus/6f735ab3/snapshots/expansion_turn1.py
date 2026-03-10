def display_vehicle_info(vehicle, fields_to_exclude=None):
    """
    Display vehicle information in a formatted string.
    
    Args:
        vehicle (dict): Dictionary containing vehicle information with keys like
                       'make', 'model', 'year', 'owner', 'registration'
        fields_to_exclude (list, optional): List of field names to exclude from display
    
    Returns:
        str: Formatted string with vehicle details
    """
    if fields_to_exclude is None:
        fields_to_exclude = []
    
    # Define display labels for each field
    field_labels = {
        'make': 'Make',
        'model': 'Model',
        'year': 'Year',
        'owner': 'Owner',
        'registration': 'Registration'
    }
    
    # Build the formatted output
    lines = ["Vehicle Information:"]
    lines.append("-" * 30)
    
    for field, label in field_labels.items():
        if field in vehicle and field not in fields_to_exclude:
            value = vehicle[field]
            lines.append(f"{label}: {value}")
    
    return "\n".join(lines)
