def validate_content_type(headers):
    """
    Validates if the Content-Type header is application/json.
    
    Args:
        headers (dict): Dictionary representing request headers
        
    Returns:
        bool: True if Content-Type is application/json, False otherwise
    """
    content_type = headers.get('Content-Type', '').lower()
    return content_type == 'application/json'
