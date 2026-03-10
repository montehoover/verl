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


def validate_json_request(input_data: dict) -> dict:
    """
    Validate a JSON request by checking headers and parsing the body.
    
    Args:
        input_data: Dictionary containing 'headers' and 'body' keys
        
    Returns:
        dict: Parsed JSON body as a dictionary
        
    Raises:
        ValueError: If content type is not 'application/json' or JSON is invalid
    """
    headers = input_data.get('headers', {})
    body = input_data.get('body', '')
    
    if not check_header_for_json(headers):
        raise ValueError("Content-Type must be 'application/json'")
    
    return decode_json_body(body)
