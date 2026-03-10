ALLOWED_FIELDS = ['make', 'model', 'year', 'registration']


def update_dictionary(original, new_data):
    """
    Update a dictionary with new data.
    
    Args:
        original: The original dictionary to update
        new_data: Dictionary containing keys and values to update
        
    Returns:
        The updated dictionary
    """
    # Create a copy to avoid modifying the original
    updated = original.copy()
    
    # Update with new data
    updated.update(new_data)
    
    return updated


def restricted_update(original, new_data, allowed_fields):
    """
    Update a dictionary with new data, but only for allowed fields.
    
    Args:
        original: The original dictionary to update
        new_data: Dictionary containing keys and values to update
        allowed_fields: List of field names that are allowed to be updated
        
    Returns:
        The updated dictionary with only allowed fields modified
    """
    # Create a copy to avoid modifying the original
    updated = original.copy()
    
    # Only update fields that are in the allowed_fields list
    for field, value in new_data.items():
        if field in allowed_fields:
            updated[field] = value
    
    return updated


def modify_car_attributes(car, updates):
    """
    Modify car attributes with specific constraints.
    
    Args:
        car: Dictionary of vehicle attributes
        updates: Dictionary of proposed modifications
        
    Returns:
        The updated vehicle dictionary with only allowed modifications
    """
    # Create a copy to avoid modifying the original
    updated_car = car.copy()
    
    # Only update fields that are in ALLOWED_FIELDS
    for field, value in updates.items():
        if field in ALLOWED_FIELDS:
            updated_car[field] = value
    
    return updated_car
