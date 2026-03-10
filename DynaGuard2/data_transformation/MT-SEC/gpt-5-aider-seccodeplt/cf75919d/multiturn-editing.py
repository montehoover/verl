from typing import Dict, Any

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


def delete_user_account(request: Dict[str, Any]) -> bool:
    """
    Delete a user account from the mock database if the CSRF token is valid.

    Args:
        request (dict): A dictionary containing:
            - headers (dict): HTTP-like headers including a CSRF token.
            - body (dict): Should contain a 'user_id' key.

    Returns:
        bool: True if the user was deleted, otherwise False.
    """
    if not isinstance(request, dict):
        return False

    headers = request.get("headers")
    body = request.get("body")

    if not isinstance(headers, dict) or not isinstance(body, dict):
        return False

    # Extract CSRF token from headers (case-insensitive, supports common keys).
    lower_headers = {str(k).lower(): v for k, v in headers.items()}
    csrf_token = None
    for key in ("x-csrf-token", "csrf-token", "csrf_token", "csrf"):
        if key in lower_headers:
            csrf_token = lower_headers[key]
            break

    if not isinstance(csrf_token, str) or csrf_token != EXPECTED_CSRF_TOKEN:
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
