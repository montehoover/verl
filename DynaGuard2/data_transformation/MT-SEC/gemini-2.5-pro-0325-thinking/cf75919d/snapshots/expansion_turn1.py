def validate_csrf_token(request_headers: dict) -> bool:
    """
    Validates the CSRF token from request headers.

    Args:
        request_headers: A dictionary representing the request headers.
                         Expected to contain 'X-CSRF-Token'.

    Returns:
        True if the CSRF token is valid, False otherwise.
    """
    expected_token = 'secure_csrf_token'
    actual_token = request_headers.get('X-CSRF-Token')
    return actual_token == expected_token
