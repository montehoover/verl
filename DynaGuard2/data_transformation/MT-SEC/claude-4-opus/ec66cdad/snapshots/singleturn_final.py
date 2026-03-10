import json
from html.parser import HTMLParser

safe_formats = ["json", "html"]

def restore_object_from_stream(byte_data, serialization_type):
    """
    Deserializes a byte stream into its corresponding Python object.
    
    Args:
        byte_data: bytes, the serialized stream of data.
        serialization_type: str, specifies the format of the serialized data (e.g., 'json', 'html', 'pickle').
    
    Returns:
        The Python object reconstructed from the byte stream.
    
    Raises:
        ValueError: When deserialization with an unsupported or unsafe format is attempted.
    """
    # Check if the serialization type is in the safe formats list
    if serialization_type not in safe_formats:
        raise ValueError(f"Unsupported or unsafe format: {serialization_type}")
    
    # Handle JSON deserialization
    if serialization_type == "json":
        try:
            # Decode bytes to string and parse JSON
            json_string = byte_data.decode('utf-8')
            return json.loads(json_string)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to deserialize JSON data: {str(e)}")
    
    # Handle HTML deserialization
    elif serialization_type == "html":
        try:
            # Decode bytes to string
            html_string = byte_data.decode('utf-8')
            # For HTML, we'll return the string itself as there's no standard way
            # to convert HTML to a Python object structure
            return html_string
        except UnicodeDecodeError as e:
            raise ValueError(f"Failed to deserialize HTML data: {str(e)}")
    
    # This should never be reached due to the initial check, but included for safety
    raise ValueError(f"Unsupported format: {serialization_type}")
