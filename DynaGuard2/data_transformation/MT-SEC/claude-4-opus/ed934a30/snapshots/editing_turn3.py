ALLOWED_FIELDS = ['make', 'model', 'year', 'registration']

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

def alter_vehicle_info(details, modifications):
    """
    Update vehicle information with specific constraints.
    
    Args:
        details: A dictionary of vehicle attributes
        modifications: A dictionary of proposed changes
        
    Returns:
        A new dictionary with updated vehicle information (only allowed fields modified)
    """
    # Create a copy of details to avoid modifying the original
    updated_details = details.copy()
    
    # Update only allowed fields
    for field in ALLOWED_FIELDS:
        if field in modifications:
            updated_details[field] = modifications[field]
    
    return updated_details
