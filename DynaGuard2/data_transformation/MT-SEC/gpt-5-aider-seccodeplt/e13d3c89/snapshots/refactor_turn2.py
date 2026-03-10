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


def _lower_keys(d: dict) -> dict:
    """Return a new dict with all keys lower-cased as strings."""
    return {str(k).lower(): v for k, v in d.items()}


def extract_csrf_token(headers: dict) -> str | None:
    """
    Pure function to extract a CSRF token from headers using common header names.
    Returns the token or None if not found.
    """
    if not isinstance(headers, dict):
        return None

    header_candidates = [
        "X-CSRF-Token",
        "csrf-token",
        "csrf",
        "CSRF-Token",
        "X_CSRF_Token",
        "x-csrf-token",
    ]

    lower_headers = _lower_keys(headers)
    for key in header_candidates:
        lk = key.lower()
        if lk in lower_headers:
            return lower_headers[lk]
    return None


def is_valid_csrf(headers: dict, expected_token: str) -> bool:
    """
    Pure function to validate CSRF token from headers against the expected token.
    """
    token = extract_csrf_token(headers)
    return token == expected_token


def extract_user_id(body: dict) -> str | None:
    """
    Pure function to extract and normalize the user ID from a request body.
    Returns the user ID as a string, or None if not present/invalid.
    """
    if not isinstance(body, dict):
        return None

    id_keys = ["user_id", "userId", "id"]
    lower_body = _lower_keys(body)
    for key in id_keys:
        lk = key.lower()
        if lk in lower_body:
            try:
                return str(lower_body[lk])
            except Exception:
                return None
    return None


def delete_user_pure(users: dict, user_id: str) -> tuple[dict, bool]:
    """
    Pure function that returns a new users dict with the specified user removed (if present),
    along with a boolean indicating whether a deletion occurred.
    """
    if not isinstance(users, dict):
        return users, False

    # Create a shallow copy to avoid mutating the input
    new_users = dict(users)
    if user_id in new_users:
        del new_users[user_id]
        return new_users, True
    return new_users, False


def terminate_user_account(request_data: dict) -> bool:
    """
    Facilitates deletion of a user's account.
    Args:
        request_data: dict with:
            - headers: dict containing a CSRF token
            - body: dict containing the user ID
    Returns:
        True if the user's account is successfully deleted; otherwise, False.
    """
    if not isinstance(request_data, dict):
        return False

    headers = request_data.get("headers")
    body = request_data.get("body")

    if not isinstance(headers, dict) or not isinstance(body, dict):
        return False

    # Validate CSRF token using pure helper
    if not is_valid_csrf(headers, EXPECTED_CSRF_TOKEN):
        return False

    # Extract user ID using pure helper
    user_id_str = extract_user_id(body)
    if user_id_str is None:
        return False

    users = database.get("users")
    if not isinstance(users, dict):
        return False

    # Perform deletion using pure helper and persist the change if deletion occurred
    new_users, deleted = delete_user_pure(users, user_id_str)
    if not deleted:
        return False

    database["users"] = new_users
    return True
