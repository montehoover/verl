def check_content_type_and_length(headers):
    """
    Verifies if the content type is 'application/json' and Content-Length is a positive integer.
    
    Args:
        headers (dict): HTTP headers from the request
        
    Returns:
        bool: True if content type is 'application/json' and Content-Length is a positive integer, False otherwise
    """
    content_type = headers.get('Content-Type', '')
    if content_type.lower() != 'application/json':
        return False
    
    content_length = headers.get('Content-Length', '')
    try:
        length = int(content_length)
        return length > 0
    except (ValueError, TypeError):
        return False
