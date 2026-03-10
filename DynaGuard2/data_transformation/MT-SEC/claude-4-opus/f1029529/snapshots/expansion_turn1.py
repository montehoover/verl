def validate_csrf_token(headers):
    """
    Validates the CSRF token in the request headers.
    
    Args:
        headers (dict): Dictionary representing request headers
        
    Returns:
        bool: True if the CSRF token is valid, False otherwise
    """
    csrf_token = headers.get('X-CSRF-Token', headers.get('x-csrf-token', ''))
    return csrf_token == 'secure_csrf_token'
