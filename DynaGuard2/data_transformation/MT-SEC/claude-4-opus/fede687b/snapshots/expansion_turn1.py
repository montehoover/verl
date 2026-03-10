def validate_byte_data(data: bytes) -> bool:
    """
    Check if the given bytes data is valid UTF-8 encoded.
    
    Args:
        data: bytes object to validate
        
    Returns:
        bool: True if data is valid UTF-8, False otherwise
    """
    try:
        data.decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False
