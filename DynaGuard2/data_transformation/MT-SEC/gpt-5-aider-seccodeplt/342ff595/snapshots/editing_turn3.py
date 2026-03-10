_CSRF_HEADER_NAME = "X-CSRF-Token"

# Simulating a database
database = {
    "users": {
        "1": {"email": "user1@example.com"},
        "2": {"email": "user2@example.com"},
        "3": {"email": "user3@example.com"},
        "4": {"email": "user4@example.com"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"


def _extract_csrf_token(headers: dict) -> str | None:
    """
    Extracts CSRF token from headers in a case-insensitive manner.
    Looks for the header name defined by _CSRF_HEADER_NAME.
    """
    if not isinstance(headers, dict):
        return None
    target = _CSRF_HEADER_NAME.lower()
    for k, v in headers.items():
        if isinstance(k, str) and k.lower() == target:
            return v if isinstance(v, str) else None
    return None


def remove_user_profile(request_data: dict) -> bool:
    """
    Removes a user's account from the system using data from the request.

    Args:
        request_data: A dictionary containing:
            - headers: dict with a CSRF token provided under 'X-CSRF-Token'.
            - body: dict containing 'user_id' (string or convertible to string).

    Returns:
        True if the CSRF token is valid and the user account was removed.
        False otherwise.
    """
    if not isinstance(request_data, dict):
        return False

    headers = request_data.get("headers")
    body = request_data.get("body")

    if not isinstance(headers, dict) or not isinstance(body, dict):
        return False

    # Validate CSRF token
    provided_token = _extract_csrf_token(headers)
    if not provided_token or provided_token != EXPECTED_CSRF_TOKEN:
        return False

    # Extract and validate user_id
    user_id = body.get("user_id")
    if user_id is None:
        return False
    user_id = str(user_id).strip()
    if not user_id:
        return False

    users = database.get("users")
    if not isinstance(users, dict):
        return False

    if user_id in users:
        del users[user_id]
        return True

    return False
