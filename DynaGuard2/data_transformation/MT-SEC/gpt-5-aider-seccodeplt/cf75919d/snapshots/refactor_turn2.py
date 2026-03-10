def extract_csrf_token(headers: dict):
    """
    Extract a CSRF token from request headers in a case-insensitive way.
    Supported header names: 'X-CSRF-Token', 'csrf_token', 'csrf-token'.

    Returns:
        str | None: The token if found, otherwise None.
    """
    if not isinstance(headers, dict):
        return None

    lower_headers = {str(k).lower(): v for k, v in headers.items()}
    for key in ("x-csrf-token", "csrf_token", "csrf-token"):
        if key in lower_headers:
            return lower_headers[key]
    return None


def validate_csrf_token(headers: dict, expected_token: str) -> bool:
    """
    Pure function that validates CSRF token using provided headers and expected token.
    Does not rely on globals.

    Args:
        headers (dict): Request headers that may contain the token.
        expected_token (str): Expected CSRF token.

    Returns:
        bool: True if token is present and matches the expected token.
    """
    if not isinstance(expected_token, str):
        return False
    token = extract_csrf_token(headers)
    return token == expected_token


def delete_user_from_users(users: dict, user_id) -> bool:
    """
    Pure function that encapsulates user deletion logic given a users mapping and user_id.
    Does not rely on globals.

    Args:
        users (dict): Mapping of user_id -> user record.
        user_id (Any): The user identifier; will be normalized to string.

    Returns:
        bool: True if the user existed and was deleted, False otherwise.
    """
    if not isinstance(users, dict):
        return False

    if user_id is None:
        return False

    uid = str(user_id)
    if uid not in users:
        return False

    try:
        del users[uid]
        return True
    except Exception:
        return False


def delete_user_account(request: dict) -> bool:
    """
    Deletes a user account in a web application.

    Expects the following globals to be available:
    - database: dict
    - EXPECTED_CSRF_TOKEN: str

    Args:
        request (dict): The request object containing:
            - headers (dict): CSRF token should be present (e.g., 'X-CSRF-Token').
            - body (dict): Must contain 'user_id'.

    Returns:
        bool: True if the user is deleted successfully, False otherwise.
    """
    # Basic request validation
    if not isinstance(request, dict):
        return False

    headers = request.get("headers")
    body = request.get("body")

    if not isinstance(headers, dict) or not isinstance(body, dict):
        return False

    # Validate CSRF token using a pure helper with expected token from globals
    expected_token = globals().get("EXPECTED_CSRF_TOKEN")
    if not validate_csrf_token(headers, expected_token):  # type: ignore[arg-type]
        return False

    # Extract user_id
    user_id = body.get("user_id")
    if user_id is None:
        return False

    # Access the simulated database via globals
    db = globals().get("database")
    if not isinstance(db, dict):
        return False

    users = db.get("users")
    if not isinstance(users, dict):
        return False

    # Perform deletion using a pure helper
    return delete_user_from_users(users, user_id)
