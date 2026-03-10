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

# Common CSRF header names to check
CSRF_HEADER_CANDIDATES = (
    "X-CSRF-Token",
    "x-csrf-token",
    "CSRF-Token",
    "csrf-token",
    "csrf_token",
    "Csrf-Token",
)


def is_valid_csrf_token(headers: dict, expected_token: str) -> bool:
    """
    Pure function: Validate CSRF token from headers against an expected value.

    Args:
        headers (dict): HTTP headers possibly containing a CSRF token.
        expected_token (str): The expected CSRF token value.

    Returns:
        bool: True if a token is present in known header names and matches expected_token.
    """
    if not isinstance(headers, dict):
        return False

    csrf_token = None
    for key in CSRF_HEADER_CANDIDATES:
        if key in headers:
            csrf_token = headers.get(key)
            break

    return csrf_token == expected_token


def delete_user_immutable(users: dict, user_id) -> tuple:
    """
    Pure function: Return a new users dict with the given user removed (if present).

    Args:
        users (dict): The users mapping (id -> user record).
        user_id: The user ID to remove.

    Returns:
        (bool, dict): Tuple of (deleted_flag, new_users_dict).
    """
    if not isinstance(users, dict):
        return (False, users)

    if user_id is None:
        return (False, users)

    user_id_str = str(user_id)

    if user_id_str not in users:
        return (False, users)

    new_users = users.copy()
    del new_users[user_id_str]
    return (True, new_users)


def remove_user_profile(request_data: dict) -> bool:
    """
    Remove a user's account from the system using their user ID.

    Args:
        request_data (dict): A representation of an HTTP request containing headers and body
                             (with the user ID in body["user_id"]).

    Returns:
        bool: True if the account was removed successfully, otherwise False.
    """
    if not isinstance(request_data, dict):
        return False

    headers = request_data.get("headers") or {}
    body = request_data.get("body") or {}

    if not isinstance(headers, dict) or not isinstance(body, dict):
        return False

    if not is_valid_csrf_token(headers, EXPECTED_CSRF_TOKEN):
        return False

    user_id = body.get("user_id")
    if user_id is None:
        return False

    users = database.get("users")
    if not isinstance(users, dict):
        return False

    deleted, new_users = delete_user_immutable(users, user_id)
    if deleted:
        database["users"] = new_users
        return True

    return False
