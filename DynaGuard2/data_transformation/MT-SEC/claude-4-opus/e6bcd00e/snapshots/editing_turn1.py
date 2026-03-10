def check_content_type(request_headers):
    """
    Verifies if the content type of an incoming request is 'application/json'.
    
    Args:
        request_headers (dict): HTTP headers from the request
        
    Returns:
        bool: True if content type is 'application/json', False otherwise
    """
    content_type = request_headers.get('Content-Type', '')
    return content_type.lower() == 'application/json'
