def update_values(original, new_data):
    """
    Update values in a dictionary with new data.
    
    Args:
        original: The original dictionary to update
        new_data: Dictionary containing keys and values to update in the original
        
    Returns:
        The updated dictionary
    """
    # Create a copy of the original to avoid modifying it
    updated = original.copy()
    
    # Update with new data
    updated.update(new_data)
    
    return updated
