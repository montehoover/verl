def validate_csrf_token(request, known_csrf_token):
    """
    Validates the CSRF token from the request headers against a known token.
    
    Args:
        request: Dictionary containing request data with headers
        known_csrf_token: The expected CSRF token value
        
    Returns:
        bool: True if the token is valid, False otherwise
    """
    # Get the CSRF token from request headers
    csrf_token = request.get('headers', {}).get('X-CSRF-Token', '')
    
    # Compare with the known token
    return csrf_token == known_csrf_token
