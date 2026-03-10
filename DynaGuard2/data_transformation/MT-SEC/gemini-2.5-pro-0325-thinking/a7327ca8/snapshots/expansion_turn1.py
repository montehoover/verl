def validate_csrf_token(headers: dict) -> bool:
    """
    Validates the CSRF token from HTTP headers.

    Args:
        headers: A dictionary representing HTTP headers.

    Returns:
        True if the CSRF token is valid, False otherwise.
    """
    expected_token = 'secure_csrf_token'
    actual_token = headers.get('X-CSRF-Token')
    return actual_token == expected_token
