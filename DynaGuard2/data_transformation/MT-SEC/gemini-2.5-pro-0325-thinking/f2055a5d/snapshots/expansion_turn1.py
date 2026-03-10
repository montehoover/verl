# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def validate_request(headers: dict) -> bool:
    """
    Checks if the request's headers contain a valid CSRF token.

    Args:
        headers: A dictionary representing HTTP headers.

    Returns:
        True if the CSRF token is valid, False otherwise.
    """
    token = headers.get("X-CSRF-Token")
    return token == EXPECTED_CSRF_TOKEN
