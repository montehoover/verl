def is_valid_string(s):
    """
    Check if a string is non-empty and doesn't contain any spaces.
    
    Args:
        s: The string to validate
        
    Returns:
        bool: True if the string is non-empty and contains no spaces, False otherwise
    """
    return len(s) > 0 and ' ' not in s
