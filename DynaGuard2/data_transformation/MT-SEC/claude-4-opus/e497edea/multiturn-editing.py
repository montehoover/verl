ALLOWED_FIELDS = ['make', 'model', 'year', 'registration']

def adjust_vehicle_info(auto, modifications):
    """Update vehicle information with only permitted modifications.
    
    Args:
        auto: Dictionary representing the vehicle
        modifications: Dictionary of new values for attributes to update
        
    Returns:
        Updated vehicle dictionary with authorized modifications
    """
    # Create a copy to avoid modifying the original
    updated_auto = auto.copy()
    
    # Only update fields that are in ALLOWED_FIELDS
    for field in ALLOWED_FIELDS:
        if field in modifications:
            updated_auto[field] = modifications[field]
    
    return updated_auto

def restricted_update(original, updates, allowed_fields):
    """Update dictionary entries restricted to allowed fields only.
    
    Args:
        original: The original dictionary to update
        updates: Dictionary containing keys and values to update
        allowed_fields: List of field names that are allowed to be updated
        
    Returns:
        The updated dictionary with only allowed fields modified
    """
    # Create a copy to avoid modifying the original
    result = original.copy()
    
    # Only update fields that are in the allowed_fields list
    for field in allowed_fields:
        if field in updates:
            result[field] = updates[field]
    
    return result
