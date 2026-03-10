def merge_dicts(original, updates):
    """
    Merge two dictionaries, with updates overwriting original values.
    
    Args:
        original: The original dictionary
        updates: Dictionary containing updates to apply
        
    Returns:
        A new dictionary with updates applied to original
    """
    # Create a copy of the original dictionary
    result = original.copy()
    
    # Apply updates
    result.update(updates)
    
    return result
