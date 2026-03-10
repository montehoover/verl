import json

def check_header_for_json(headers: dict) -> bool:
    """
    Check if the Content-Type header is set to 'application/json'.
    
    Args:
        headers: Dictionary representing request headers
        
    Returns:
        bool: True if Content-Type is 'application/json', False otherwise
    """
    content_type = headers.get('Content-Type', '').lower()
    return content_type == 'application/json'


def decode_json_body(json_string: str) -> dict:
    """
    Decode a JSON-encoded string into a dictionary.
    
    Args:
        json_string: JSON-encoded string to parse
        
    Returns:
        dict: Parsed JSON data as a dictionary
        
    Raises:
        ValueError: If the JSON string is improperly formatted
    """
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
