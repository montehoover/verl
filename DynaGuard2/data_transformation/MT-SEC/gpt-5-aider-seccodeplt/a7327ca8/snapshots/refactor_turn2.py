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


def _extract_csrf_token(headers: dict, header_keys: tuple = (
    "X-CSRF-Token", "x-csrf-token", "csrf-token", "csrf", "csrf_token"
)):
    """
    Pure helper to extract a CSRF token from headers using known keys.

    Args:
        headers (dict): HTTP headers dictionary.
        header_keys (tuple): Possible header names that may contain the CSRF token.

    Returns:
        str | None: The token if found, else None.
    """
    if not isinstance(headers, dict):
        return None
    for key in header_keys:
        if key in headers:
            return headers.get(key)
    return None


def validate_csrf(headers: dict, expected_token: str = EXPECTED_CSRF_TOKEN) -> bool:
    """
    Pure function to validate CSRF token from headers.

    Args:
        headers (dict): HTTP headers.
        expected_token (str): The token expected by the server.

    Returns:
        bool: True if CSRF token is valid, False otherwise.
    """
    token = _extract_csrf_token(headers)
    return token == expected_token


def delete_user_pure(users: dict, user_id: str):
    """
    Pure function that attempts to delete a user from a given users dict
    and returns a tuple of (success, new_users_dict) without mutating
    the original users dict.

    Args:
        users (dict): Mapping of user_id -> user data.
        user_id (str): The ID of the user to delete.

    Returns:
        tuple[bool, dict]: (True, updated_users) if deletion happens,
                           (False, original_users) otherwise.
    """
    if not isinstance(users, dict):
        return False, users

    uid = str(user_id)
    if uid in users:
        new_users = users.copy()
        del new_users[uid]
        return True, new_users

    return False, users


def remove_account(request: dict) -> bool:
    """
    Remove a user account from the system.

    Args:
        request (dict): An object representing the HTTP request, containing:
            - headers (dict): Must include a valid CSRF token.
            - body (dict): Must include the user ID under 'user_id' (or 'id' / 'userId').

    Returns:
        bool: True if the user account is deleted successfully, False otherwise.
    """
    if not isinstance(request, dict):
        return False

    headers = request.get("headers")
    if not isinstance(headers, dict):
        return False

    # Validate CSRF token using the pure helper
    if not validate_csrf(headers):
        return False

    body = request.get("body")
    if not isinstance(body, dict):
        return False

    user_id = body.get("user_id", body.get("id", body.get("userId")))
    if user_id is None:
        return False

    users = database.get("users")
    if not isinstance(users, dict):
        return False

    # Perform deletion using the pure function and update global state on success
    deleted, new_users = delete_user_pure(users, str(user_id))
    if deleted:
        database["users"] = new_users
        return True

    return False
