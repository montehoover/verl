import json
import csv
import io

trusted_serializations = ["json", "csv"]


def validate_byte_stream(byte_stream: bytes) -> bool:
    """
    Validates if a byte stream contains valid UTF-8 encoded data.
    
    Args:
        byte_stream: The bytes input to validate
        
    Returns:
        bool: True if the byte stream is valid UTF-8, False otherwise
    """
    try:
        byte_stream.decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False


def detect_serialization_format(byte_stream: bytes) -> str:
    """
    Detects the serialization format of a byte stream.
    
    Args:
        byte_stream: The bytes input to analyze
        
    Returns:
        str: The detected format ('json', 'csv', 'xml')
        
    Raises:
        ValueError: If the format is unrecognized or potentially harmful
    """
    if not byte_stream:
        raise ValueError("Empty byte stream")
    
    # Try to decode as UTF-8 first
    try:
        text = byte_stream.decode('utf-8').strip()
    except UnicodeDecodeError:
        raise ValueError("Invalid UTF-8 encoding")
    
    if not text:
        raise ValueError("Empty content after decoding")
    
    # Check for JSON
    if (text.startswith('{') and text.endswith('}')) or \
       (text.startswith('[') and text.endswith(']')):
        return 'json'
    
    # Check for XML
    if text.startswith('<?xml') or \
       (text.startswith('<') and text.endswith('>') and '</' in text):
        return 'xml'
    
    # Check for CSV - look for comma-separated values with optional quotes
    lines = text.split('\n', 2)
    if lines:
        first_line = lines[0]
        # Simple CSV detection: contains commas and doesn't look like other formats
        if ',' in first_line and not first_line.startswith('<'):
            # Additional check: see if it has consistent comma count across lines
            if len(lines) > 1:
                comma_count = first_line.count(',')
                if lines[1].count(',') == comma_count:
                    return 'csv'
            else:
                # Single line with commas
                return 'csv'
    
    raise ValueError("Unrecognized serialization format")


def restore_object(serialized_bytes: bytes, serialization_method: str):
    """
    Securely deserializes a byte stream into a Python object.
    
    Args:
        serialized_bytes: The byte stream to deserialize
        serialization_method: The format used for serialization ('json' or 'csv')
        
    Returns:
        The restored Python object
        
    Raises:
        ValueError: If the method is unrecognized or insecure
    """
    if serialization_method not in trusted_serializations:
        raise ValueError(f"Unrecognized or insecure serialization method: {serialization_method}")
    
    if not serialized_bytes:
        raise ValueError("Empty byte stream")
    
    # Decode bytes to string
    try:
        text = serialized_bytes.decode('utf-8')
    except UnicodeDecodeError:
        raise ValueError("Invalid UTF-8 encoding")
    
    if serialization_method == 'json':
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
    
    elif serialization_method == 'csv':
        # Parse CSV and return as list of dictionaries
        csv_file = io.StringIO(text)
        reader = csv.DictReader(csv_file)
        result = list(reader)
        
        # If no headers, return as list of lists
        if not result and text.strip():
            csv_file.seek(0)
            reader = csv.reader(csv_file)
            result = list(reader)
        
        return result
