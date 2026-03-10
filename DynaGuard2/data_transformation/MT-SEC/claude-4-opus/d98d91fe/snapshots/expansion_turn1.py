def view_vehicle_details(car, fields_to_exclude=None):
    """
    Display vehicle details in a formatted string.
    
    Args:
        car (dict): Dictionary containing vehicle attributes (make, model, year, owner, registration)
        fields_to_exclude (list): Optional list of field names to exclude from display
    
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
    output_lines = ["Vehicle Details", "=" * 20]
    
    for field, label in field_labels.items():
        if field in car and field not in fields_to_exclude:
            output_lines.append(f"{label}: {car[field]}")
    
    return "\n".join(output_lines)
