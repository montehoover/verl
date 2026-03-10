def validate_csrf_token(headers):
    """
    Validates CSRF token from HTTP request headers.
    
    Args:
        headers (dict): Dictionary containing HTTP request headers
        
    Returns:
        bool: True if CSRF token is valid, False otherwise
    """
    csrf_token = headers.get('X-CSRF-Token') or headers.get('x-csrf-token')
    return csrf_token == 'secure_csrf_token'
