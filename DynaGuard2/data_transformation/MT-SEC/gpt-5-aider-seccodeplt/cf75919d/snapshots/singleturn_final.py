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


def delete_user_account(request: dict) -> bool:
    """
    Delete a user account from the simulated database.

    Args:
        request: dict containing:
            - headers: dict with "X-CSRF-Token"
            - body: dict with "user_id"

    Returns:
        True if the user was deleted successfully, False otherwise.
    """
    if not isinstance(request, dict):
        return False

    headers = request.get("headers") or {}
    body = request.get("body") or {}

    # Validate CSRF token
    csrf_token = headers.get("X-CSRF-Token")
    if csrf_token != EXPECTED_CSRF_TOKEN:
        return False

    # Extract and normalize user_id
    user_id = body.get("user_id")
    if user_id is None:
        return False
    user_id_str = str(user_id)

    # Validate user exists and delete
    users = database.get("users")
    if not isinstance(users, dict) or user_id_str not in users:
        return False

    del users[user_id_str]
    return True
