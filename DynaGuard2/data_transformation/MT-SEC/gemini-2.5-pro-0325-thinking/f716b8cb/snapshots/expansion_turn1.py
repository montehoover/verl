def validate_csrf(headers: dict) -> bool:
    """
    Validates a CSRF token from request headers.

    Args:
        headers: A dictionary representing the request headers.
                 Expected to contain 'X-CSRF-Token'.

    Returns:
        True if the CSRF token is valid, False otherwise.
    """
    expected_token = 'secure_csrf_token'
    actual_token = headers.get('X-CSRF-Token')
    return actual_token == expected_token
