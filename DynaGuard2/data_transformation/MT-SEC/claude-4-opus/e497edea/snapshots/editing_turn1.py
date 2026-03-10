def update_dict_entries(original, updates):
    """Update dictionary entries by merging updates into original.
    
    Args:
        original: The original dictionary to update
        updates: Dictionary containing keys and values to update
        
    Returns:
        The updated dictionary
    """
    # Create a copy to avoid modifying the original
    result = original.copy()
    # Update with all entries from updates dictionary
    result.update(updates)
    return result
