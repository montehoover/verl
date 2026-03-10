EXPECTED_CSRF_TOKEN = 'secure_csrf_token'

def validate_request(headers: dict) -> bool:
    """
    Validates an incoming request by checking the CSRF token.

    Args:
        headers: A dictionary representing the request headers.

    Returns:
        True if the CSRF token is valid, False otherwise.
    """
    token = headers.get('X-CSRF-Token')
    return token == EXPECTED_CSRF_TOKEN
