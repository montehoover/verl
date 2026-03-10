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
            - headers (dict): Should include "X-CSRF-Token".
            - body (dict): Should include "user_id".

    Returns:
        bool: True if the user account is deleted successfully, False otherwise.
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
    user_id = str(user_id)

    users = database.get("users")
    if not isinstance(users, dict):
        return False

    if user_id in users:
        del users[user_id]
        return True

    return False
