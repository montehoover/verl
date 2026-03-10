def validate_request(headers):
    """
    Validates if the CSRF token in the request headers matches the expected token.
    
    Args:
        headers (dict): Dictionary containing request headers
        
    Returns:
        bool: True if CSRF token is valid, False otherwise
    """
    csrf_token = headers.get('X-CSRF-Token', '')
    expected_token = 'secure_csrf_token'
    return csrf_token == expected_token
