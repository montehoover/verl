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


def terminate_user_account(request_data: dict) -> bool:
    """
    Deletes a user's account from the in-memory database.

    Args:
        request_data (dict): A dictionary representing an HTTP request, with:
            - 'headers': dict containing 'X-CSRF-Token'
            - 'body': dict containing 'user_id'

    Returns:
        bool: True if the user's account is successfully deleted; otherwise, False.
    """
    if not isinstance(request_data, dict):
        return False

    headers = request_data.get("headers", {})
    if not isinstance(headers, dict):
        return False

    csrf_token = headers.get("X-CSRF-Token")
    if csrf_token != EXPECTED_CSRF_TOKEN:
        return False

    body = request_data.get("body", {})
    if not isinstance(body, dict):
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
