import string

def is_printable_byte_stream(byte_stream):
    """
    Check if a byte stream consists entirely of printable ASCII characters.
    
    Args:
        byte_stream: bytes object or byte array
        
    Returns:
        bool: True if all bytes are printable ASCII, False otherwise
    """
    # Define printable ASCII range (32-126) plus common whitespace characters
    printable_chars = set(string.printable.encode('ascii'))
    
    try:
        # Check each byte in the stream
        for byte in byte_stream:
            if byte not in printable_chars:
                return False
        return True
    except Exception:
        return False


def detect_stream_format(byte_stream):
    """
    Detect the format of a byte stream based on common markers or headers.
    
    Args:
        byte_stream: bytes object or byte array
        
    Returns:
        str: The detected format ('JSON', 'XML', 'HTML')
        
    Raises:
        ValueError: If format is unrecognized or potentially unsafe
    """
    if not byte_stream:
        raise ValueError("Empty byte stream")
    
    # Convert to string for pattern matching, but handle potential decode errors
    try:
        # Try to decode as UTF-8 first
        text = byte_stream.decode('utf-8', errors='ignore').strip()
    except:
        raise ValueError("Unable to decode byte stream")
    
    # Check for JSON format
    if (text.startswith('{') and text.endswith('}')) or \
       (text.startswith('[') and text.endswith(']')):
        # Basic JSON validation
        try:
            import json
            json.loads(text)
            return 'JSON'
        except:
            pass
    
    # Check for XML format
    if text.startswith('<?xml') or \
       (text.startswith('<') and text.endswith('>') and '</' in text):
        return 'XML'
    
    # Check for HTML format
    html_markers = ['<!DOCTYPE html', '<html', '<HTML', '<!doctype html']
    for marker in html_markers:
        if marker in text[:100]:  # Check first 100 chars
            return 'HTML'
    
    # Additional HTML detection for partial documents
    if ('<head>' in text.lower() or '<body>' in text.lower() or 
        '<title>' in text.lower() or '<div>' in text.lower()):
        return 'HTML'
    
    # If none of the formats match, raise ValueError
    raise ValueError("Unrecognized or potentially unsafe format")
