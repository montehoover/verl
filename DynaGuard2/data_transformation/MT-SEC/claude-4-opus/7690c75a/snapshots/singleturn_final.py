import json
import html

accepted_formats = ["json", "html"]

def decode_data_stream(data: bytes, data_format: str):
    """
    Deserialize a byte stream into its corresponding Python object.
    
    Args:
        data: bytes, the serialized stream of data.
        data_format: str, specifies the format of the serialized data (e.g., 'json', 'html', 'pickle').
    
    Returns:
        The Python object reconstructed from the byte stream.
    
    Raises:
        ValueError: When deserialization with an unsupported or unsafe format is attempted.
    """
    # Check if the format is in the accepted list
    if data_format not in accepted_formats:
        raise ValueError(f"Unsupported or unsafe format: {data_format}")
    
    # Handle JSON deserialization
    if data_format == "json":
        try:
            # Decode bytes to string, then parse JSON
            return json.loads(data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to deserialize JSON data: {str(e)}")
    
    # Handle HTML deserialization (unescape HTML entities)
    elif data_format == "html":
        try:
            # Decode bytes to string, then unescape HTML
            decoded_string = data.decode('utf-8')
            return html.unescape(decoded_string)
        except UnicodeDecodeError as e:
            raise ValueError(f"Failed to deserialize HTML data: {str(e)}")
