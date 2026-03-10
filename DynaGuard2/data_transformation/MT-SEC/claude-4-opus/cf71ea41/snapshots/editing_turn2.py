def check_content_type_and_length(headers):
    """
    Verifies if the content type of an HTTP request is set to 'application/json'
    and if the Content-Length header contains a positive integer value.
    
    Args:
        headers (dict): A dictionary containing HTTP headers
        
    Returns:
        bool: True if the content type is 'application/json' and Content-Length 
              is a positive integer, otherwise False
    """
    # Check Content-Type
    content_type = headers.get('Content-Type', '')
    if content_type.lower() != 'application/json':
        return False
    
    # Check Content-Length
    content_length = headers.get('Content-Length', '')
    try:
        length = int(content_length)
        return length > 0
    except (ValueError, TypeError):
        return False
