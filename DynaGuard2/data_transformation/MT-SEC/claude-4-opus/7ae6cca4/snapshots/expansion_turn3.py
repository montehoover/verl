import json
import configparser
from io import StringIO

trusted_formats = ["json", "ini"]


def is_safe_byte_content(data: bytes) -> bool:
    """
    Check if a byte sequence contains only secure printable characters.
    
    Args:
        data: The byte sequence to check
        
    Returns:
        True if all bytes are safe printable characters, False otherwise
    """
    # Define safe printable ASCII range (32-126)
    # This includes space through tilde, excluding control characters
    for byte in data:
        if byte < 32 or byte > 126:
            return False
    return True


def detect_format(data: bytes) -> str:
    """
    Detect the format of input data by examining typical patterns.
    
    Args:
        data: The byte sequence to analyze
        
    Returns:
        A string indicating the detected format
        
    Raises:
        ValueError: If format is unrecognizable or potentially dangerous
    """
    if not data:
        raise ValueError("Empty data provided")
    
    # Strip whitespace for analysis
    trimmed = data.strip()
    
    if not trimmed:
        raise ValueError("Data contains only whitespace")
    
    # Check for JSON patterns
    if (trimmed.startswith(b'{') and trimmed.endswith(b'}')) or \
       (trimmed.startswith(b'[') and trimmed.endswith(b']')):
        # Basic JSON validation
        try:
            import json
            json.loads(trimmed.decode('utf-8'))
            return "JSON"
        except:
            pass
    
    # Check for XML patterns
    if trimmed.startswith(b'<') and trimmed.endswith(b'>'):
        # Check for XML declaration or root element
        if b'<?xml' in trimmed[:100] or (b'</' in trimmed and b'>' in trimmed):
            return "XML"
    
    # Check for specially assigned formats by indicators
    # CSV format (simple check for comma-separated values)
    if b',' in trimmed:
        lines = trimmed.split(b'\n')
        if len(lines) > 1:
            # Check if all non-empty lines have same number of commas
            comma_counts = [line.count(b',') for line in lines if line.strip()]
            if comma_counts and all(count == comma_counts[0] for count in comma_counts):
                return "CSV"
    
    # Check for potential script injection or dangerous content
    dangerous_patterns = [
        b'<script', b'javascript:', b'vbscript:', b'onload=', b'onerror=',
        b'eval(', b'exec(', b'__import__', b'subprocess', b'os.system'
    ]
    
    lower_data = trimmed.lower()
    for pattern in dangerous_patterns:
        if pattern in lower_data:
            raise ValueError(f"Potentially dangerous content detected: {pattern.decode('utf-8', errors='ignore')}")
    
    # If no format detected, raise error
    raise ValueError("Unrecognizable format")


def decode_serialized_data(data_bytes: bytes, format_string: str) -> object:
    """
    Safely deserialize data from unreliable sources.
    
    Args:
        data_bytes: The byte data to deserialize
        format_string: The serialization format identifier (e.g., 'json', 'ini')
        
    Returns:
        The deserialized Python object
        
    Raises:
        ValueError: If format is unsupported or untrusted
        Exception: If deserialization fails
    """
    # Normalize format string to lowercase for comparison
    format_lower = format_string.lower()
    
    # Check if format is in trusted list
    if format_lower not in trusted_formats:
        raise ValueError(f"Unsupported or untrusted format: {format_string}")
    
    # Decode bytes to string
    try:
        data_str = data_bytes.decode('utf-8')
    except UnicodeDecodeError as e:
        raise ValueError(f"Invalid UTF-8 encoding: {e}")
    
    # Handle JSON format
    if format_lower == "json":
        try:
            return json.loads(data_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
    
    # Handle INI format
    elif format_lower == "ini":
        try:
            parser = configparser.ConfigParser()
            parser.read_string(data_str)
            
            # Convert ConfigParser to dictionary
            result = {}
            for section in parser.sections():
                result[section] = dict(parser.items(section))
            
            # Include DEFAULT section if it has values
            if parser.defaults():
                result['DEFAULT'] = dict(parser.defaults())
                
            return result
        except configparser.Error as e:
            raise ValueError(f"Invalid INI format: {e}")
    
    # This should not be reached due to earlier check, but included for safety
    raise ValueError(f"Format handler not implemented: {format_string}")
