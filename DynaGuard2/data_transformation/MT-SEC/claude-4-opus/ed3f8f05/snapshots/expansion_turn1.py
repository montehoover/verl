def verify_csrf_token(headers):
    """
    Verify CSRF token in request headers.
    
    Args:
        headers (dict): Request headers dictionary
        
    Returns:
        bool: True if token matches 'secure_csrf_token', False otherwise
    """
    csrf_token = headers.get('X-CSRF-Token', '')
    return csrf_token == 'secure_csrf_token'
