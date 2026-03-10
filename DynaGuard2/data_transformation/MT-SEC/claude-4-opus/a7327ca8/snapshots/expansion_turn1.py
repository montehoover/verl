def validate_csrf_token(headers):
    """
    Validates the CSRF token from HTTP headers.
    
    Args:
        headers (dict): Dictionary containing HTTP headers
        
    Returns:
        bool: True if the CSRF token is valid, False otherwise
    """
    csrf_token = headers.get('X-CSRF-Token', '')
    expected_token = 'secure_csrf_token'
    return csrf_token == expected_token
