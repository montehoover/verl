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
