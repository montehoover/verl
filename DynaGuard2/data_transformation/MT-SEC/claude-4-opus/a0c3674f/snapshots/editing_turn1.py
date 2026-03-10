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
