def validate_byte_data(data: bytes) -> bool:
    """
    Validates if the given bytes data is valid UTF-8 encoded.
    
    Args:
        data: The bytes data to validate
        
    Returns:
        bool: True if the data is valid UTF-8, False otherwise
    """
    try:
        data.decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False
