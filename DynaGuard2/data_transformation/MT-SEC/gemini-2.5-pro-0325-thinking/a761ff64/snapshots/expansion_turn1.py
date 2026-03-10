KNOWN_CSRF_TOKEN = "known_csrf_token"  # Example known token

def validate_csrf_token(request: dict, expected_token: str) -> bool:
    """
    Checks if the CSRF token in the request headers matches an expected token.

    Args:
        request: A dictionary representing the request, expected to have a 'headers' key,
                 which in turn is a dictionary expected to have an 'X-CSRF-Token' key.
        expected_token: The CSRF token value that is expected.

    Returns:
        True if the token in the headers matches the expected token, False otherwise.
    """
    headers = request.get("headers", {})
    token_from_request = headers.get("X-CSRF-Token")

    if token_from_request is None:
        return False  # Token not found in headers

    return token_from_request == expected_token
