def validate_byte_stream(byte_stream):
    """
    Validates if a byte stream contains only valid UTF-8 characters.
    
    Args:
        byte_stream: bytes object to validate
        
    Returns:
        bool: True if the byte stream is valid UTF-8, False otherwise
    """
    try:
        byte_stream.decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False


def detect_data_format(byte_stream):
    """
    Detects the format of a byte stream based on common markers or headers.
    
    Args:
        byte_stream: bytes object to analyze
        
    Returns:
        str: The detected format ('JSON', 'XML', or 'HTML')
        
    Raises:
        ValueError: If the format is unrecognized or potentially unsafe
    """
    # First validate UTF-8
    if not validate_byte_stream(byte_stream):
        raise ValueError("Invalid UTF-8 byte stream")
    
    # Convert to string for analysis
    try:
        content = byte_stream.decode('utf-8').strip()
    except:
        raise ValueError("Unable to decode byte stream")
    
    if not content:
        raise ValueError("Empty byte stream")
    
    # Check for JSON
    if (content.startswith('{') and content.endswith('}')) or \
       (content.startswith('[') and content.endswith(']')):
        return 'JSON'
    
    # Check for XML
    if content.startswith('<?xml') or \
       (content.startswith('<') and content.endswith('>') and 
        not content.lower().startswith('<!doctype html') and
        not content.lower().startswith('<html')):
        return 'XML'
    
    # Check for HTML
    if content.lower().startswith('<!doctype html') or \
       content.lower().startswith('<html') or \
       (content.lower().find('<html') != -1 and 
        content.lower().find('</html>') != -1):
        return 'HTML'
    
    # Format not recognized
    raise ValueError("Unrecognized or potentially unsafe data format")
