from typing import Any, Dict

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


def remove_account(request: Dict[str, Any]) -> bool:
    """
    Remove a user account from the system.

    The request dict is expected to have:
      - headers: dict with a CSRF token (header name 'X-CSRF-Token')
      - body: dict with 'user_id' key

    Returns True if the user was deleted and CSRF token is valid, else False.
    """
    if not isinstance(request, dict):
        return False

    headers = request.get("headers")
    body = request.get("body")
    if not isinstance(headers, dict) or not isinstance(body, dict):
        return False

    # Normalize header keys to lowercase to make lookup case-insensitive
    normalized_headers = {str(k).lower(): v for k, v in headers.items()}
    csrf_token = normalized_headers.get("x-csrf-token")
    if csrf_token != EXPECTED_CSRF_TOKEN:
        return False

    user_id = body.get("user_id")
    if user_id is None:
        return False
    user_id = str(user_id)

    users = database.get("users")
    if not isinstance(users, dict):
        return False

    if user_id not in users:
        return False

    # Perform deletion
    del users[user_id]
    return True
