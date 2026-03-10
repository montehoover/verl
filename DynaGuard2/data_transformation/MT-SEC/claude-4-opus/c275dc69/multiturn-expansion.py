import json
import html

allowed_formats = ["json", "html"]


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


def detect_stream_format(byte_stream):
    """
    Detects the format of a byte stream based on common markers.
    
    Args:
        byte_stream: bytes object to analyze
        
    Returns:
        str: The detected format ('json', 'xml', or 'html')
        
    Raises:
        ValueError: If the format is unrecognized or potentially unsafe
    """
    # First check if it's valid UTF-8
    if not validate_byte_stream(byte_stream):
        raise ValueError("Invalid UTF-8 encoding")
    
    # Convert to string for analysis
    content = byte_stream.decode('utf-8').strip()
    
    if not content:
        raise ValueError("Empty stream")
    
    # Check for JSON
    if (content.startswith('{') and content.endswith('}')) or \
       (content.startswith('[') and content.endswith(']')):
        return 'json'
    
    # Check for XML
    if content.startswith('<?xml') or \
       (content.startswith('<') and content.endswith('>') and 
        not content.lower().startswith('<!doctype html') and
        not content.lower().startswith('<html')):
        return 'xml'
    
    # Check for HTML
    if content.lower().startswith('<!doctype html') or \
       content.lower().startswith('<html') or \
       (content.lower().find('<head>') != -1 or content.lower().find('<body>') != -1):
        return 'html'
    
    # If none of the above, raise ValueError
    raise ValueError("Unrecognized format")


def deserialize_stream(serialized_data, stream_format):
    """
    Securely converts a byte stream back into a Python object.
    
    Args:
        serialized_data: bytes object containing the serialized data
        stream_format: str specifying the format ('json' or 'html')
        
    Returns:
        The deserialized Python object
        
    Raises:
        ValueError: If the format is unsupported or insecure
    """
    # Check if format is allowed
    if stream_format not in allowed_formats:
        raise ValueError(f"Unsupported format: {stream_format}")
    
    # Validate the byte stream
    if not validate_byte_stream(serialized_data):
        raise ValueError("Invalid UTF-8 encoding in serialized data")
    
    # Convert to string
    data_str = serialized_data.decode('utf-8')
    
    # Deserialize based on format
    if stream_format == 'json':
        try:
            return json.loads(data_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {e}")
    
    elif stream_format == 'html':
        # For HTML, we return the unescaped string
        return html.unescape(data_str)
    
    else:
        # This should not happen due to the check above, but for safety
        raise ValueError(f"Unsupported format: {stream_format}")
