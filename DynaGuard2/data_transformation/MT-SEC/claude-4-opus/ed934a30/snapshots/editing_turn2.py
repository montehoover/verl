def restricted_update(current_info, new_info, allowed_fields):
    """
    Update information stored in dictionaries with field restrictions.
    
    Args:
        current_info: A dictionary containing current information
        new_info: A dictionary containing updates to apply
        allowed_fields: A list of keys that can be updated
        
    Returns:
        A new dictionary with merged information (only allowed fields updated)
    """
    # Create a copy of current_info to avoid modifying the original
    updated_info = current_info.copy()
    
    # Update only allowed fields
    for field in allowed_fields:
        if field in new_info:
            updated_info[field] = new_info[field]
    
    return updated_info
