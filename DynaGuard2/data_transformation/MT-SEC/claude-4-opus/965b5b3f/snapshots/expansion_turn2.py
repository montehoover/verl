import json

def validate_content_type(headers: dict) -> bool:
    """
    Validates if the Content-Type header is 'application/json'.
    
    Args:
        headers: Dictionary representing request headers
        
    Returns:
        bool: True if Content-Type is 'application/json', False otherwise
    """
    content_type = headers.get('Content-Type', '')
    return content_type.lower() == 'application/json'


def decode_json_body(body: str) -> dict:
    """
    Decodes JSON data from the request body.
    
    Args:
        body: String representing the JSON body
        
    Returns:
        dict: The decoded JSON data as a dictionary
        
    Raises:
        ValueError: If the JSON is malformed
    """
    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")
