ALLOWED_FIELDS = ["make", "model", "year", "registration"]


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


def log_vehicle_changes(car, changes):
    """
    Update vehicle details and log changes for auditing.
    
    Args:
        car (dict): Dictionary containing current vehicle attributes
        changes (dict): Dictionary containing fields to update with new values
    
    Returns:
        dict: Updated vehicle dictionary
    """
    import datetime
    
    # Create a copy of the car dictionary to avoid modifying the original
    updated_car = car.copy()
    
    # Initialize change log if it doesn't exist
    if '_change_log' not in updated_car:
        updated_car['_change_log'] = []
    
    # Process each change
    for field, new_value in changes.items():
        # Only log if the field exists and value is different
        if field in updated_car and updated_car[field] != new_value:
            # Create log entry
            log_entry = {
                'timestamp': datetime.datetime.now().isoformat(),
                'field': field,
                'old_value': updated_car[field],
                'new_value': new_value
            }
            
            # Add to change log
            updated_car['_change_log'].append(log_entry)
            
            # Update the field value
            updated_car[field] = new_value
        elif field not in updated_car:
            # New field being added
            log_entry = {
                'timestamp': datetime.datetime.now().isoformat(),
                'field': field,
                'old_value': None,
                'new_value': new_value
            }
            
            # Add to change log
            updated_car['_change_log'].append(log_entry)
            
            # Add the new field
            updated_car[field] = new_value
    
    return updated_car


def modify_car_details(car, changes):
    """
    Modify vehicle details with restrictions on allowed fields.
    
    Args:
        car (dict): Dictionary containing current vehicle attributes
        changes (dict): Dictionary containing fields to update with new values
    
    Returns:
        dict: Updated vehicle dictionary with only allowed changes applied
    """
    # Create a copy of the car dictionary to avoid modifying the original
    updated_car = car.copy()
    
    # Filter changes to only include allowed fields
    allowed_changes = {}
    for field, value in changes.items():
        if field in ALLOWED_FIELDS:
            allowed_changes[field] = value
    
    # Apply allowed changes
    for field, new_value in allowed_changes.items():
        updated_car[field] = new_value
    
    return updated_car
