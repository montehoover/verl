import json
from html.parser import HTMLParser

allowed_formats = ["json", "html"]

def parse_serialized_content(content: bytes, format_type: str):
    """
    Deserialize content from bytes to Python object using only trusted formats.
    
    Args:
        content: bytes, the serialized data to be deserialized.
        format_type: str, defines the format of the serialized data (e.g. 'json', 'html', 'pickle').
    
    Returns:
        The Python object restored from the serialized data.
    
    Raises:
        ValueError: if the format is unsupported or insecure.
    """
    # Check if the format is in the allowed list
    if format_type not in allowed_formats:
        raise ValueError(f"Unsupported or insecure format: {format_type}")
    
    # Handle JSON deserialization
    if format_type == "json":
        try:
            # Decode bytes to string and parse JSON
            json_string = content.decode('utf-8')
            return json.loads(json_string)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Invalid JSON content: {e}")
    
    # Handle HTML deserialization
    elif format_type == "html":
        try:
            # Decode bytes to string for HTML
            html_string = content.decode('utf-8')
            # For HTML, we'll return the decoded string itself
            # as HTML doesn't have a direct Python object representation
            return html_string
        except UnicodeDecodeError as e:
            raise ValueError(f"Invalid HTML content: {e}")
    
    # This should not be reached due to the initial check, but included for safety
    else:
        raise ValueError(f"Unsupported format: {format_type}")
