def validate_csrf(headers):
    """
    Validates CSRF token from headers dictionary.
    
    Args:
        headers: Dictionary containing HTTP headers
        
    Returns:
        bool: True if CSRF token is valid, False otherwise
    """
    csrf_token = headers.get('X-CSRF-Token', '')
    expected_token = 'secure_csrf_token'
    
    return csrf_token == expected_token
