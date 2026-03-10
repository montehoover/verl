def validate_byte_stream(byte_stream):
    """
    Validates if a byte stream contains valid UTF-8 encoded data.
    
    Args:
        byte_stream (bytes): The byte stream to validate
        
    Returns:
        bool: True if the byte stream is valid UTF-8, False otherwise
    """
    try:
        byte_stream.decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False
