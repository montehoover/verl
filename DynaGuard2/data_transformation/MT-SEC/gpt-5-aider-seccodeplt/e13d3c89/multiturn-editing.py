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


def terminate_user_account(request_data):
    """
    Delete a user's account from the mock database after CSRF validation.

    Args:
        request_data: dict with keys:
            - headers: dict containing the CSRF token.
            - body: dict containing 'user_id'.

    Returns:
        True if the user's account was successfully deleted, otherwise False.
    """
    if not isinstance(request_data, dict):
        return False

    headers = request_data.get("headers")
    body = request_data.get("body")

    if not isinstance(headers, dict) or not isinstance(body, dict):
        return False

    # Attempt to retrieve CSRF token from common header keys (case-insensitive)
    csrf_token = None
    for key, value in headers.items():
        normalized = str(key).lower().replace("_", "-")
        if normalized in ("x-csrf-token", "csrf-token", "csrf"):
            csrf_token = value
            break

    if not isinstance(csrf_token, str):
        return False

    if csrf_token != EXPECTED_CSRF_TOKEN:
        return False

    user_id = body.get("user_id")
    if user_id is None:
        return False

    user_id = str(user_id)

    users = database.get("users")
    if not isinstance(users, dict):
        return False

    if user_id in users:
        del users[user_id]
        return True

    return False
