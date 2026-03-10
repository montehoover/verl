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
