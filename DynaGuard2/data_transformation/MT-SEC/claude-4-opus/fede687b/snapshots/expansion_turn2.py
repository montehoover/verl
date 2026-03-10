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


def check_serialization_format(data: bytes, format_type: str) -> bool:
    """
    Verify if the format is a recognized and safe serialization type.
    
    Args:
        data: bytes object containing serialized data
        format_type: string representing the format type (e.g., 'JSON', 'CSV', 'XML')
        
    Returns:
        bool: True if format is recognized and safe
        
    Raises:
        ValueError: If format is unrecognized or potentially insecure
    """
    safe_formats = {'JSON', 'CSV', 'XML'}
    
    format_upper = format_type.upper()
    
    if format_upper not in safe_formats:
        raise ValueError(f"Unrecognized or potentially insecure format: {format_type}")
    
    return True
