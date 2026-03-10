def check_content_type_and_length(headers):
    """
    Verifies if the content type of an incoming request is 'application/json'
    and if it has a valid Content-Length header with a positive integer value.
    
    Args:
        headers (dict): A dictionary containing request headers.
        
    Returns:
        bool: True if the content type is 'application/json' and Content-Length 
              is a positive integer, otherwise False.
    """
    content_type = headers.get('Content-Type', '')
    if content_type.lower() != 'application/json':
        return False
    
    content_length = headers.get('Content-Length', '')
    try:
        length = int(content_length)
        return length > 0
    except ValueError:
        return False
