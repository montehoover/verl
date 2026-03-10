import json
from html.parser import HTMLParser

allowed_formats = ["json", "html"]

def deserialize_content_stream(stream, stream_format):
    """
    Deserialize a byte stream into a Python object.
    
    Args:
        stream: bytes, containing serialized information
        stream_format: str, indicating the format of the serialized data
        
    Returns:
        The deserialized Python object in the corresponding format
        
    Raises:
        ValueError: for unsupported or insecure formats
    """
    # Check if the format is allowed
    if stream_format not in allowed_formats:
        raise ValueError(f"Unsupported or insecure format: {stream_format}")
    
    # Deserialize based on format
    if stream_format == "json":
        try:
            # Decode bytes to string and parse JSON
            return json.loads(stream.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Invalid JSON data: {str(e)}")
            
    elif stream_format == "html":
        try:
            # For HTML, we'll just decode it to a string
            # since HTML isn't typically "deserialized" into Python objects
            # but rather parsed for specific purposes
            return stream.decode('utf-8')
        except UnicodeDecodeError as e:
            raise ValueError(f"Invalid HTML data: {str(e)}")
