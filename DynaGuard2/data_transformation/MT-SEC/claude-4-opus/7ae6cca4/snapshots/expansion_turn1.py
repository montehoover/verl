def is_safe_byte_content(data: bytes) -> bool:
    """
    Check if a byte sequence contains only secure printable characters.
    
    Args:
        data: The byte sequence to check
        
    Returns:
        True if all bytes are safe printable characters, False otherwise
    """
    # Define safe printable ASCII range (32-126)
    # This includes space through tilde, excluding control characters
    for byte in data:
        if byte < 32 or byte > 126:
            return False
    return True
