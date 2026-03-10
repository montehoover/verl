def validate_byte_data(data: bytes) -> bool:
    """
    Validate if the given bytes contain only valid UTF-8 characters.
    
    Args:
        data: The byte stream to validate
        
    Returns:
        bool: True if the bytes are valid UTF-8, False otherwise
    """
    try:
        data.decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False
