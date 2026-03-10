import json


def validate_content_type(headers: dict) -> bool:
    """
    Validate that the Content-Type header is application/json.
    
    Args:
        headers: Dictionary representing request headers
        
    Returns:
        bool: True if Content-Type is application/json, False otherwise
    """
    content_type = headers.get('Content-Type', '')
    return content_type.lower() == 'application/json'


def extract_request_body(request: dict) -> str:
    """
    Extract the body from an incoming request.
    
    Args:
        request: Dictionary representing the entire request (headers and body)
        
    Returns:
        str: The request body as a string
    """
    return request.get('body', '')


def analyze_json_request(request: dict) -> dict:
    """
    Analyze a JSON request by validating content type and parsing the body.
    
    Args:
        request: Dictionary representing the incoming request
        
    Returns:
        dict: Parsed JSON body
        
    Raises:
        ValueError: If content type is not application/json or if JSON is malformed
    """
    headers = request.get('headers', {})
    
    if not validate_content_type(headers):
        raise ValueError("Invalid content type. Expected 'application/json'")
    
    body = extract_request_body(request)
    
    if not body:
        raise ValueError("Empty request body")
    
    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON: {str(e)}")
