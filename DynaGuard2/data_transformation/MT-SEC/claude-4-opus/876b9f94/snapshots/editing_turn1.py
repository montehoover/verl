def check_content_type(headers):
    """
    Verifies if the content type of an HTTP request is 'application/json'.
    
    Args:
        headers (dict): Dictionary containing HTTP headers
        
    Returns:
        bool: True if content type is 'application/json', False otherwise
    """
    content_type = headers.get('Content-Type', '')
    return content_type.lower() == 'application/json'
