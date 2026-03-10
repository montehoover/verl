import json


def validate_content_type(headers: dict) -> bool:
    """
    Validates if the Content-Type header is application/json.
    
    Args:
        headers: Dictionary representing HTTP headers
        
    Returns:
        bool: True if Content-Type is application/json, False otherwise
    """
    content_type = headers.get('Content-Type', '')
    return content_type == 'application/json'


def extract_request_body(request: dict) -> str:
    """
    Extracts the body from an incoming request if it has the correct content type.
    
    Args:
        request: Dictionary containing 'headers' and 'body'
        
    Returns:
        str: The body as a string if content type is valid, empty string otherwise
    """
    headers = request.get('headers', {})
    if not validate_content_type(headers):
        return ''
    
    body = request.get('body', '')
    return str(body)


def process_json_input(request: dict) -> dict:
    """
    Parses the JSON body of an incoming request.
    
    Args:
        request: Dictionary containing 'headers' and 'body'
        
    Returns:
        dict: The parsed JSON object
        
    Raises:
        ValueError: If content type is incorrect or JSON decoding fails
    """
    headers = request.get('headers', {})
    if not validate_content_type(headers):
        raise ValueError("Invalid content type: expected application/json")
    
    body = request.get('body', '')
    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON decoding failed: {str(e)}")
