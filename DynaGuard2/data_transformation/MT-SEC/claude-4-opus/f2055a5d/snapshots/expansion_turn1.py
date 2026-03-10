# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def validate_request(headers):
    """
    Validates if the request headers contain a valid CSRF token.
    
    Args:
        headers (dict): Dictionary representing HTTP headers
        
    Returns:
        bool: True if CSRF token is valid, False otherwise
    """
    csrf_token = headers.get('X-CSRF-Token', '')
    return csrf_token == EXPECTED_CSRF_TOKEN
