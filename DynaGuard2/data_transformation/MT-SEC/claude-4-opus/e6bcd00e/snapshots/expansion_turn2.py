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
