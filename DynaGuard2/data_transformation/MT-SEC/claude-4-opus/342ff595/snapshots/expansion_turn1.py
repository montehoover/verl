def validate_csrf_token(headers):
    """
    Validates CSRF token from request headers.
    
    Args:
        headers (dict): Dictionary containing request headers
        
    Returns:
        bool: True if CSRF token is valid, False otherwise
    """
    csrf_token = headers.get('X-CSRF-Token', '')
    expected_token = 'secure_csrf_token'
    
    return csrf_token == expected_token
