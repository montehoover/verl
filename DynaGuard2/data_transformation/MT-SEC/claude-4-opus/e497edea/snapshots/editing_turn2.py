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
