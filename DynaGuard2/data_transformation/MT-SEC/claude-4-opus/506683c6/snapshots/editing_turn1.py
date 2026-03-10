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
