EXPECTED_CSRF_TOKEN = 'secure_csrf_token'
CSRF_HEADER_NAME = 'X-CSRF-Token'

def validate_csrf_token(request_headers: dict) -> bool:
    """
    Validates the CSRF token from request headers.

    Args:
        request_headers: A dictionary representing the request headers.

    Returns:
        True if the CSRF token is valid, False otherwise.
    """
    actual_token = request_headers.get(CSRF_HEADER_NAME)
    return actual_token == EXPECTED_CSRF_TOKEN
