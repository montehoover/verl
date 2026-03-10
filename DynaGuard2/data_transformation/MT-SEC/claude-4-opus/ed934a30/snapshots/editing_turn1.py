def update_info(current_info, new_info):
    """
    Update information stored in dictionaries.
    
    Args:
        current_info: A dictionary containing current information
        new_info: A dictionary containing updates to apply
        
    Returns:
        A new dictionary with merged information
    """
    # Create a copy of current_info to avoid modifying the original
    updated_info = current_info.copy()
    
    # Update with new information
    updated_info.update(new_info)
    
    return updated_info
