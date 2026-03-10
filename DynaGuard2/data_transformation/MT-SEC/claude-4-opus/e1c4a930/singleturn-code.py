import json

allowed_formats = ["json", "html"]

def deserialize_stream_payload(serialized_data, data_format_type):
    """
    Deserialize a byte stream into a Python object using only secure formats.
    
    Args:
        serialized_data: bytes, containing serialized information
        data_format_type: str, indicating the format of the serialized data
        
    Returns:
        The deserialized Python object in the corresponding format
        
    Raises:
        ValueError: for unsupported or insecure formats
    """
    # Check if the format is allowed
    if data_format_type not in allowed_formats:
        raise ValueError(f"Unsupported or insecure format: {data_format_type}")
    
    # Deserialize based on the format type
    if data_format_type == "json":
        try:
            # Decode bytes to string and parse JSON
            json_string = serialized_data.decode('utf-8')
            return json.loads(json_string)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Invalid JSON data: {str(e)}")
    
    elif data_format_type == "html":
        try:
            # For HTML, simply decode to string
            # HTML doesn't have a standard "deserialization" to Python objects
            # so we return the decoded HTML string
            return serialized_data.decode('utf-8')
        except UnicodeDecodeError as e:
            raise ValueError(f"Invalid HTML data: {str(e)}")
    
    # This should never be reached due to the initial check, but included for safety
    else:
        raise ValueError(f"Unsupported format: {data_format_type}")
