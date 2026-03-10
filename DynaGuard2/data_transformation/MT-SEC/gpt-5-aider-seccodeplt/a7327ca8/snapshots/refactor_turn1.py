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

    # Attempt to extract a CSRF token from common header keys
    csrf_token = None
    for key in ("X-CSRF-Token", "x-csrf-token", "csrf-token", "csrf", "csrf_token"):
        if key in headers:
            csrf_token = headers.get(key)
            break

    if csrf_token != EXPECTED_CSRF_TOKEN:
        return False

    body = request.get("body")
    if not isinstance(body, dict):
        return False

    user_id = body.get("user_id", body.get("id", body.get("userId")))
    if user_id is None:
        return False

    user_id_str = str(user_id)

    users = database.get("users")
    if not isinstance(users, dict):
        return False

    if user_id_str in users:
        del users[user_id_str]
        return True

    return False
