def merge_dicts(original, updates):
    """
    Merge two dictionaries, with updates overwriting original values.
    
    Args:
        original: The original dictionary
        updates: Dictionary containing updates to apply
        
    Returns:
        A new dictionary with updates applied to original
    """
    # Create a copy of the original dictionary
    result = original.copy()
    
    # Apply updates
    result.update(updates)
    
    return result


def restricted_update(original, updates, allowed_keys):
    """
    Update a dictionary but only for specified allowed keys.
    
    Args:
        original: The original dictionary
        updates: Dictionary containing updates to apply
        allowed_keys: List of keys that are allowed to be updated
        
    Returns:
        A new dictionary with restricted updates applied
    """
    # Create a copy of the original dictionary
    result = original.copy()
    
    # Apply only allowed updates
    for key in allowed_keys:
        if key in updates:
            result[key] = updates[key]
    
    return result


ALLOWED_FIELDS = ['make', 'model', 'year', 'registration']


def update_car_information(car_details, changes):
    """
    Update car information with only allowed fields.
    
    Args:
        car_details: Dictionary containing current car details
        changes: Dictionary containing fields to update
        
    Returns:
        Dictionary with the updates applied to allowed fields only
    """
    # Create a copy of the car details
    updated_car = car_details.copy()
    
    # Apply only allowed changes
    for field in ALLOWED_FIELDS:
        if field in changes:
            updated_car[field] = changes[field]
    
    return updated_car
