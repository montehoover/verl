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


def detect_data_format(data: bytes) -> str:
    """
    Detect the format of the given byte data based on common markers.
    
    Args:
        data: The byte stream to analyze
        
    Returns:
        str: The detected format ('json', 'xml', or 'html')
        
    Raises:
        ValueError: If the format is unrecognized or potentially unsafe
    """
    if not data:
        raise ValueError("Empty data provided")
    
    # Try to decode as UTF-8 first
    try:
        text = data.decode('utf-8').strip()
    except UnicodeDecodeError:
        raise ValueError("Data is not valid UTF-8")
    
    if not text:
        raise ValueError("Empty text after decoding")
    
    # Check for JSON
    if (text.startswith('{') and text.endswith('}')) or \
       (text.startswith('[') and text.endswith(']')):
        return 'json'
    
    # Check for XML
    if text.startswith('<?xml') or \
       (text.startswith('<') and text.endswith('>') and '</' in text):
        return 'xml'
    
    # Check for HTML
    if text.lower().startswith('<!doctype html') or \
       text.lower().startswith('<html') or \
       ('<html' in text.lower() and '</html>' in text.lower()):
        return 'html'
    
    raise ValueError("Unrecognized or potentially unsafe format")
