def validate_binary_data(data: bytes) -> bool:
    """
    Validates if the given binary data is valid UTF-8 encoded.
    
    Args:
        data: Binary data to validate
        
    Returns:
        bool: True if data is valid UTF-8, False otherwise
    """
    try:
        data.decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False
