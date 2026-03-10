import json
from html.parser import HTMLParser

allowed_formats = ["json", "html"]

def deserialize_stream(serialized_data, stream_format):
    """
    Deserialize a byte stream back into a Python object.
    
    Args:
        serialized_data: bytes, the serialized data that needs to be deserialized
        stream_format: str, a string indicating the deserialization format
        
    Returns:
        The corresponding Python object reconstructed from the serialized content
        
    Raises:
        ValueError: if the specified format is either unsupported or considered insecure
    """
    if stream_format not in allowed_formats:
        raise ValueError(f"Unsupported or insecure format: {stream_format}")
    
    if stream_format == "json":
        try:
            # Decode bytes to string and parse JSON
            return json.loads(serialized_data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to deserialize JSON data: {str(e)}")
    
    elif stream_format == "html":
        try:
            # For HTML, we'll return the decoded string
            # In a real application, you might want to parse it further
            return serialized_data.decode('utf-8')
        except UnicodeDecodeError as e:
            raise ValueError(f"Failed to deserialize HTML data: {str(e)}")
