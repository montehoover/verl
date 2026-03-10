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

    # Attempt to read a CSRF token from common header names
    csrf_token = None
    for key in (
        "X-CSRF-Token",
        "x-csrf-token",
        "CSRF-Token",
        "csrf-token",
        "csrf_token",
        "Csrf-Token",
    ):
        if key in headers:
            csrf_token = headers.get(key)
            break

    if csrf_token != EXPECTED_CSRF_TOKEN:
        return False

    user_id = body.get("user_id")
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
