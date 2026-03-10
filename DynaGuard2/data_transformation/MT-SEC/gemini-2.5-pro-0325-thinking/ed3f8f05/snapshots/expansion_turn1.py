def verify_csrf_token(request_headers: dict) -> bool:
    """
    Checks whether a CSRF token in a given request headers dictionary
    matches the expected token.

    Args:
        request_headers: A dictionary representing the request headers.

    Returns:
        True if the token matches, False otherwise.
    """
    expected_token = 'secure_csrf_token'
    actual_token = request_headers.get('X-CSRF-Token')
    return actual_token == expected_token
