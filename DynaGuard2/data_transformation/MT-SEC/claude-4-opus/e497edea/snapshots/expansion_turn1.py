def display_vehicle_info(auto, fields_to_hide=None):
    """
    Display vehicle information in a formatted string.
    
    Args:
        auto (dict): Dictionary containing vehicle information with keys like
                     'make', 'model', 'year', 'owner', 'registration'
        fields_to_hide (list, optional): List of field names to exclude from display
    
    Returns:
        str: Formatted string containing vehicle details
    """
    if fields_to_hide is None:
        fields_to_hide = []
    
    # Define the display order and labels
    field_mapping = {
        'make': 'Make',
        'model': 'Model',
        'year': 'Year',
        'owner': 'Owner',
        'registration': 'Registration'
    }
    
    # Build the formatted output
    lines = ["Vehicle Information", "-" * 20]
    
    for field_key, field_label in field_mapping.items():
        if field_key not in fields_to_hide and field_key in auto:
            lines.append(f"{field_label}: {auto[field_key]}")
    
    return "\n".join(lines)
