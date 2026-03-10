def update_record(data, modifications):
    """
    Update values in a dictionary with modifications from another dictionary.
    
    Args:
        data: A dictionary to be updated
        modifications: A dictionary with keys and values to update in data
    
    Returns:
        The updated dictionary
    """
    # Create a copy to avoid modifying the original
    updated_data = data.copy()
    
    # Update with all modifications
    updated_data.update(modifications)
    
    return updated_data


def restricted_update(data, modifications, has_permission):
    """
    Update values in a dictionary with restrictions based on permission.
    
    Args:
        data: A dictionary to be updated
        modifications: A dictionary with keys and values to update in data
        has_permission: Boolean indicating if all fields can be updated
    
    Returns:
        The updated dictionary with restricted modifications
    """
    # Define allowed fields for restricted updates
    allowed_fields = {'name', 'email', 'phone', 'address'}
    
    # Create a copy to avoid modifying the original
    updated_data = data.copy()
    
    if has_permission:
        # Full permission: update all requested fields
        updated_data.update(modifications)
    else:
        # Restricted permission: only update allowed fields
        for key, value in modifications.items():
            if key in allowed_fields:
                updated_data[key] = value
    
    return updated_data
